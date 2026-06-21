"""Reference lookup and ABNT formatting helpers for the review handler.

Covers all reference-related user actions: resolving numbered citations,
listing all document references, formatting user-provided entries, and
enriching metadata via MongoDB, DOI resolution, CrossRef, and Tavily.
"""

from __future__ import annotations

import os
import re

from revisao_agents.agents.reference_extractor_agent import run_reference_extractor_agent
from revisao_agents.agents.reference_formatter_agent import run_reference_formatter_agent
from revisao_agents.tools.tavily_web_search import extract_tavily, search_tavily_incremental
from revisao_agents.utils.vector_utils.vector_store import search_chunk_records

from ..base import _detect_user_language, _localized_text
from .document import (
    _split_sections,
)
from .intent import (
    _build_phrase_reference_query_seed,
    _extract_citation_number,
    _extract_provided_reference_items,
    _extract_requested_citation_numbers,
)


def _search_reference_in_mongo_by_phrase(
    user_text: str, missing_numbers: list[int]
) -> tuple[str, dict]:
    """Search the MongoDB vector store for source metadata matching a phrase citation.

    Args:
        user_text: The user's original message, used to extract a query seed
            phrase via :func:`_build_phrase_reference_query_seed`.
        missing_numbers: Citation numbers that were absent from the reference
            list and triggered this lookup.

    Returns:
        A 2-tuple of:

        - A localized markdown reply string with the candidate reference
          details (title, DOI, URL, and file path when present).
        - A stats dict with keys ``found`` (bool), ``mongo_queries`` (int),
          and ``mongo_hits`` (int).
    """
    language = _detect_user_language(user_text)
    query_seed = _build_phrase_reference_query_seed(user_text)
    if not query_seed:
        return (
            _localized_text(
                language,
                "Não consegui extrair um trecho para busca no MongoDB.",
                "I couldn't extract a phrase to search in MongoDB.",
            ),
            {"found": False, "mongo_queries": 0, "mongo_hits": 0},
        )

    records = search_chunk_records(query_seed[:500], k=5)
    if not records:
        return (
            _localized_text(
                language,
                "Não encontrei candidato no MongoDB para essa frase.",
                "I couldn't find a MongoDB candidate for that phrase.",
            ),
            {"found": False, "mongo_queries": 1, "mongo_hits": 0},
        )

    best = records[0]
    title = str(best.get("source_title") or "").strip()
    doi = str(best.get("doi") or "").strip()
    url = str(best.get("source_url") or "").strip()
    file_path = str(best.get("file_path") or "").strip()

    lines = [
        _localized_text(
            language,
            f"### Referência candidata para {', '.join(f'[{n}]' for n in missing_numbers)} (MongoDB)",
            f"### Candidate reference for {', '.join(f'[{n}]' for n in missing_numbers)} (MongoDB)",
        ),
        "",
        f"- {_localized_text(language, 'Título', 'Title')}: {title or _localized_text(language, '(não identificado)', '(not identified)')}",
    ]
    if doi:
        lines.append(f"- DOI: {doi}")
    if url:
        lines.append(f"- URL: {url}")
    if file_path:
        lines.append(f"- {_localized_text(language, 'Arquivo', 'File')}: {file_path}")

    return "\n".join(lines), {"found": True, "mongo_queries": 1, "mongo_hits": 1}


def _search_reference_on_web_by_phrase(
    user_text: str, missing_numbers: list[int]
) -> tuple[str, dict]:
    """Search the internet via Tavily for source metadata matching a phrase citation.

    Args:
        user_text: The user's original message, used to extract a query seed
            phrase via :func:`_build_phrase_reference_query_seed`.
        missing_numbers: Citation numbers that were absent from the reference
            list and triggered this lookup.

    Returns:
        A 2-tuple of:

        - A localized markdown reply string with the first extracted page
          title and URL.
        - A stats dict with keys ``found`` (bool), ``web_queries`` (int),
          and ``web_hits`` (int).
    """
    language = _detect_user_language(user_text)
    query_seed = _build_phrase_reference_query_seed(user_text)
    if not query_seed:
        return (
            _localized_text(
                language,
                "Não consegui extrair um trecho para busca na internet.",
                "I couldn't extract a phrase to search on the internet.",
            ),
            {"found": False, "web_queries": 0, "web_hits": 0},
        )

    web = search_tavily_incremental(query=query_seed[:400], previous_urls=[], max_results=3)
    urls = web.get("new_urls", [])[:3]
    if not urls:
        return (
            _localized_text(
                language,
                "Não encontrei resultados web para essa frase.",
                "I couldn't find web results for that phrase.",
            ),
            {"found": False, "web_queries": 1, "web_hits": 0},
        )

    extracted = extract_tavily.invoke({"urls": urls, "include_images": False})
    items = extracted.get("extracted", []) if isinstance(extracted, dict) else []
    if not items:
        return (
            _localized_text(
                language,
                "Encontrei URLs, mas não consegui extrair metadados suficientes.",
                "I found URLs, but I couldn't extract enough metadata.",
            ),
            {"found": False, "web_queries": 1, "web_hits": 0},
        )

    first = items[0]
    title = str(first.get("title") or "").strip()
    url = str(first.get("url") or "").strip()

    lines = [
        _localized_text(
            language,
            f"### Referência candidata para {', '.join(f'[{n}]' for n in missing_numbers)} (Internet)",
            f"### Candidate reference for {', '.join(f'[{n}]' for n in missing_numbers)} (Internet)",
        ),
        "",
        f"- {_localized_text(language, 'Título', 'Title')}: {title or _localized_text(language, '(não identificado)', '(not identified)')}",
    ]
    if url:
        lines.append(f"- URL: {url}")

    return "\n".join(lines), {"found": True, "web_queries": 1, "web_hits": 1}


def _build_reference_confirmation_prompt(
    intent: str, user_text: str, allow_web: bool = True
) -> tuple[str, dict]:
    """Build a user-facing confirmation prompt for a pending reference action.

    Args:
        intent: Reference action label; one of ``"list_all"`` or
            ``"format_provided"``.
        user_text: The original user message, used for language detection
            and for extracting provided reference items when intent is
            ``"format_provided"``.
        allow_web: Whether external web search is enabled.  When ``False``
            and incomplete items exist, the prompt warns the user to enable
            it before confirming.

    Returns:
        A 2-tuple of:

        - A localized prompt string asking the user to confirm or cancel.
        - A pending-action payload dict with keys ``intent``,
          ``original_message``, ``requires_web``, and ``incomplete_items``.
    """
    language = _detect_user_language(user_text)
    pending: dict = {"intent": intent, "original_message": user_text}

    if intent == "list_all":
        prompt = _localized_text(
            language,
            "Posso listar todas as referências do documento e formatar em ABNT. Deseja continuar? Responda **sim** ou **não**.",
            "I can list all references from the document and format them in ABNT. Do you want to continue? Reply **yes** or **no**.",
        )
        if not allow_web:
            prompt += "\n\n" + _localized_text(
                language,
                "Observação: a busca na web está desativada, então algumas referências podem ficar incompletas.",
                "Note: web search is disabled, so some references may remain incomplete.",
            )
        pending.update({"requires_web": False, "incomplete_items": []})
        return prompt, pending

    if intent == "format_provided":
        items = _extract_provided_reference_items(user_text)
        incomplete_items: list[int] = []
        for idx, item in enumerate(items, start=1):
            metadata = _metadata_from_raw_reference(idx, item)
            if not _is_metadata_complete(metadata):
                incomplete_items.append(idx)

        requires_web = bool(incomplete_items)
        pending.update({"requires_web": requires_web, "incomplete_items": incomplete_items})

        base_prompt = _localized_text(
            language,
            f"Posso formatar {len(items)} referência(s) fornecida(s) em ABNT. Deseja continuar? Responda **sim** ou **não**.",
            f"I can format the {len(items)} provided reference(s) in ABNT. Do you want to continue? Reply **yes** or **no**.",
        )

        if requires_web and not allow_web:
            warning = _localized_text(
                language,
                "Algumas referências parecem incompletas e podem exigir busca na web para completar.",
                "Some references look incomplete and may require web search to complete.",
            )
            item_list = ", ".join(f"[{idx}]" for idx in incomplete_items)
            detail = _localized_text(
                language,
                f"Itens incompletos: {item_list}. Ative **Allow web search** e confirme novamente com **sim**.",
                f"Incomplete items: {item_list}. Enable **Allow web search** and confirm again with **yes**.",
            )
            return f"{base_prompt}\n\n{warning}\n{detail}", pending

        if requires_web:
            note = _localized_text(
                language,
                "Algumas referências parecem incompletas; vou usar a web para complementar os dados.",
                "Some references look incomplete; I'll use the web to enrich the data.",
            )
            return f"{base_prompt}\n\n{note}", pending

        return base_prompt, pending

    fallback = _localized_text(
        language,
        "Não consegui identificar a ação de referências. Refaça o pedido.",
        "I couldn't identify the reference action. Please resend the request.",
    )
    pending.update({"requires_web": False, "incomplete_items": []})
    return fallback, pending


def _normalize_reference_key(raw: str) -> str:
    """Normalize a raw reference string for deduplication comparisons.

    Strips leading ``[n]`` number prefixes, DOI fragments, bare URLs, and
    punctuation, then lowercases and collapses whitespace.

    Args:
        raw: Raw reference text, optionally prefixed with a numbered tag
            such as ``"[1] Author et al."``.

    Returns:
        A lowercase, punctuation-stripped string suitable for set-based
        deduplication.
    """
    text = re.sub(r"^\[\d+\]\s*", "", raw or "")
    text = re.sub(r"\s+", " ", text).strip().lower()
    text = re.sub(r"doi:\s*10\.[^\s,;]+", "", text)
    text = re.sub(r"https?://\S+", "", text)
    return re.sub(r"[^\w\s]", "", text).strip()


def _title_from_file_path(path: str) -> str:
    """Derive a human-readable title from a local PDF file path.

    Removes the ``.pdf`` extension, replaces underscores and hyphens with
    spaces, and collapses consecutive whitespace.

    Args:
        path: Absolute or relative path to a PDF file.

    Returns:
        A title string derived from the filename.  Returns an empty string
        when *path* is empty.
    """
    base = os.path.basename(path or "")
    base = re.sub(r"\.pdf$", "", base, flags=re.IGNORECASE)
    base = re.sub(r"[_+\-]", " ", base)
    base = re.sub(r"\s+", " ", base).strip()
    return base


def _metadata_from_raw_reference(number: int | None, raw_reference: str) -> dict:
    """Extract structured metadata from a free-form reference string.

    Attempts to parse DOI, URL, PDF path, and publication year from the raw
    text.  When a PDF path is found, the title is derived from the filename;
    otherwise the title is approximated from the remaining text after
    stripping identifiers and noise tokens.

    Args:
        number: Display position in the reference list.  Pass ``None`` when
            the position is unknown.
        raw_reference: Free-form reference text, optionally prefixed with a
            numbered tag such as ``"[1]"``.

    Returns:
        A metadata dict with keys: ``number``, ``raw``, ``title``, ``doi``,
        ``url``, ``year``, ``file_path``, ``derived_from_path``.
    """
    raw = (raw_reference or "").strip()
    body = re.sub(r"^\[\d+\]\s*", "", raw).strip()

    doi_match = re.search(r"(10\.\d{4,9}/[^\s,;]+)", body, flags=re.IGNORECASE)
    url_match = re.search(r"(https?://\S+)", body, flags=re.IGNORECASE)
    path_match = re.search(r"(/[^\n]*?\.pdf)", body, flags=re.IGNORECASE)
    year_match = re.search(r"\b(19|20)\d{2}\b", body)

    file_path = path_match.group(1).strip() if path_match else ""
    title_guess = ""

    if file_path:
        title_guess = _title_from_file_path(file_path)
    else:
        text_no_url = re.sub(r"https?://\S+", "", body)
        text_no_doi = re.sub(r"10\.\d{4,9}/[^\s,;]+", "", text_no_url, flags=re.IGNORECASE)
        text_no_path = re.sub(r"/[^\n]*?\.pdf", "", text_no_doi, flags=re.IGNORECASE)
        text_no_labels = re.sub(
            r"\b(?:dispon[ií]vel em|arquivo local|citado em)\b:?.*",
            "",
            text_no_path,
            flags=re.IGNORECASE,
        )
        title_guess = re.sub(r"\s+", " ", text_no_labels).strip(" .;,")

    title_guess = re.sub(
        r"\bDOI\b\s*:?\s*10\.\d{4,9}/[^\s,;]+", "", title_guess, flags=re.IGNORECASE
    )
    title_guess = re.sub(r"https?://\S+", "", title_guess)
    title_guess = re.sub(r"\s+", " ", title_guess).strip(" .;,")

    return {
        "number": number,
        "raw": raw,
        "title": title_guess,
        "doi": doi_match.group(1).rstrip(".)],;") if doi_match else "",
        "url": (url_match.group(1).rstrip(".)],;")) if url_match else "",
        "year": year_match.group(0) if year_match else "",
        "file_path": file_path,
        "derived_from_path": bool(file_path),
    }


def _is_metadata_complete(metadata: dict) -> bool:
    """Determine whether a metadata dict has enough information for ABNT output.

    A DOI alone is considered sufficient.  Otherwise both a non-path-derived
    title **and** at least one of year or URL are required.

    Args:
        metadata: Metadata dict as returned by :func:`_metadata_from_raw_reference`
            or similar helpers.

    Returns:
        ``True`` when the entry is considered complete for ABNT formatting,
        ``False`` otherwise.
    """
    title = (metadata.get("title") or "").strip()
    year = (metadata.get("year") or "").strip()
    doi = (metadata.get("doi") or "").strip()
    url = (metadata.get("url") or "").strip()
    derived_from_path = bool(metadata.get("derived_from_path"))

    if doi:
        return True
    return bool(title and not derived_from_path and (year or url))


def _extract_non_numbered_mentions(markdown: str) -> list[str]:
    """Extract in-text author-year citations that are not numbered references.

    Scans the markdown for parenthesised ``(Author, Year)`` patterns and
    for bare reference lines inside section-level reference blocks that lack
    a leading ``[n]`` number tag.

    Args:
        markdown: Full markdown document text.

    Returns:
        Deduplicated list of citation strings in author-year or raw-text
        format.  Each entry has been normalised and bounded to 180
        characters.
    """
    mentions: list[str] = []

    patterns = [
        r"\(([A-Z][A-Za-zÀ-ÿ'’\-]+(?:\s+et\s+al\.)?(?:\s*&\s*[A-Z][A-Za-zÀ-ÿ'’\-]+)?\s*,\s*(?:19|20)\d{2})\)",
        r"\(([A-Z][^()\n]{6,120},\s*(?:19|20)\d{2})\)",
    ]

    for pattern in patterns:
        for match in re.findall(pattern, markdown):
            text = re.sub(r"\s+", " ", match).strip(" .;,")
            if text:
                mentions.append(text)

    lines = markdown.splitlines()
    in_refs = False
    for line in lines:
        stripped = line.strip()
        if re.match(
            r"^###\s+(?:References\s+for\s+this\s+section|Refer[êe]ncias\s+desta\s+se[çc][ãa]o)\s*$",
            stripped,
            flags=re.IGNORECASE,
        ):
            in_refs = True
            continue
        if in_refs and re.match(r"^##\s+", stripped):
            in_refs = False
        if not in_refs:
            continue
        if not stripped or stripped.startswith("<!--"):
            continue
        if re.match(r"^\[\d+\]", stripped):
            continue
        cleaned = re.sub(r"^[-*]\s+", "", stripped)
        if (
            cleaned
            and len(cleaned) <= 180
            and "http" not in cleaned.lower()
            and "doi" not in cleaned.lower()
        ):
            mentions.append(cleaned)

    dedup: list[str] = []
    seen = set()
    for mention in mentions:
        key = _normalize_reference_key(mention)
        if not key or key in seen:
            continue
        seen.add(key)
        dedup.append(mention)
    return dedup


def _collect_reference_inventory(markdown: str) -> dict:
    """Build a structured inventory of all references and citations in the document.

    Args:
        markdown: Full markdown document text.

    Returns:
        A dict with five keys:

        - ``references_by_number`` (dict[int, str]): Maps each citation
          number to its raw reference line.
        - ``citation_paragraphs`` (dict[int, list[str]]): Maps each citation
          number to the body paragraphs that contain it.
        - ``unique_references`` (list[str]): Deduplicated reference lines
          ordered by citation number.
        - ``cited_numbers`` (list[int]): Sorted list of all citation numbers
          found in body paragraphs.
        - ``non_numbered_mentions`` (list[str]): Author-year citations not
          expressed as numbered references.
    """
    sections = _split_sections(markdown)
    references_by_number: dict[int, str] = {}
    citation_paragraphs: dict[int, list[str]] = {}

    for section in sections:
        for ref in section.get("references", []):
            match = re.match(r"^\[(\d+)\]\s*(.+)$", ref.strip())
            if not match:
                continue
            number = int(match.group(1))
            text = f"[{number}] {match.group(2).strip()}"
            references_by_number[number] = text

        for paragraph in section.get("paragraphs", []):
            p_text = paragraph.get("text", "")
            for number_token in re.findall(r"\[(\d+)\]", p_text):
                number = int(number_token)
                citation_paragraphs.setdefault(number, []).append(p_text)

    unique_refs: list[str] = []
    seen_keys: set[str] = set()
    for number in sorted(references_by_number.keys()):
        ref = references_by_number[number]
        key = _normalize_reference_key(ref)
        if key and key in seen_keys:
            continue
        if key:
            seen_keys.add(key)
        unique_refs.append(ref)

    cited_numbers = sorted(citation_paragraphs.keys())
    non_numbered_mentions = _extract_non_numbered_mentions(markdown)
    return {
        "references_by_number": references_by_number,
        "citation_paragraphs": citation_paragraphs,
        "unique_references": unique_refs,
        "cited_numbers": cited_numbers,
        "non_numbered_mentions": non_numbered_mentions,
    }


def _collect_all_raw_references_text(markdown: str) -> list[str]:
    """Extract every non-empty line from all reference sections in a markdown document.

    Detects reference section headings (e.g. *References*, *Referências*,
    *Bibliography*) and collects lines until the next heading of the same or
    higher level.

    Args:
        markdown: Full markdown document text.

    Returns:
        List of raw reference lines preserving order of appearance.
    """
    ref_heading_re = re.compile(
        r"^(#+)\s+(?:[\d]+[\s\.]+)?(refer[eê]ncias|references|bibliography|bibliograf\w+|bibliog\w+)\b",
        re.IGNORECASE,
    )
    any_heading_re = re.compile(r"^(#+)\s+")

    lines = markdown.splitlines()
    collected: list[str] = []
    collecting = False
    current_depth = 0

    for line in lines:
        stripped = line.strip()
        ref_match = ref_heading_re.match(stripped)
        if ref_match:
            collecting = True
            current_depth = len(ref_match.group(1))
            continue

        if collecting:
            any_match = any_heading_re.match(stripped)
            if any_match and len(any_match.group(1)) <= current_depth:
                collecting = False
            elif stripped:
                collected.append(stripped)

    return collected


def _collect_all_citation_paragraphs(markdown: str) -> dict[int, list[str]]:
    """Scan the document body for paragraphs that contain numbered citation tokens.

    Non-paragraph lines (headings, HTML comments) are skipped.  At most two
    paragraphs per citation number are stored to keep memory bounded.

    Args:
        markdown: Full markdown document text.

    Returns:
        Dict mapping each citation number ``n`` to a list of up to two body
        lines that contain the token ``[n]``.
    """
    result: dict[int, list[str]] = {}
    for line in markdown.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("<!--"):
            continue
        nums = {int(n) for n in re.findall(r"\[(\d+)\]", stripped)}
        for num in nums:
            paragraphs = result.setdefault(num, [])
            if len(paragraphs) < 2:
                paragraphs.append(stripped)
    return result


def _handle_resolve_numbers_request(
    markdown: str, user_text: str, allow_web: bool = True
) -> tuple[str, dict]:
    """Resolve specific numbered references using the extractor-to-formatter agent pipeline.

    When no specific numbers are requested, all references found in the
    document are processed.

    Args:
        markdown: Working-copy document text from which references are read.
        user_text: User's request message, parsed for citation numbers and
            used for language detection.
        allow_web: Whether external web search is permitted during extraction
            and formatting.

    Returns:
        A 2-tuple of:

        - A localized markdown reply with an ABNT-formatted reference list.
        - A metadata dict with keys ``intent`` (``"resolve_numbers"``),
          ``count`` (int), and ``agent`` (str).
    """
    language = _detect_user_language(user_text)
    requested = _extract_requested_citation_numbers(user_text)
    inventory = _collect_reference_inventory(markdown)
    references_by_number: dict[int, str] = inventory.get("references_by_number", {})

    entries = (
        {n: references_by_number[n] for n in requested if n in references_by_number}
        if requested
        else references_by_number
    )

    if not entries:
        msg = _localized_text(
            language,
            "Nenhuma referência encontrada para os números solicitados.",
            "No references found for the requested numbers.",
        )
        return msg, {"intent": "resolve_numbers", "count": 0, "agent": "none"}

    raw_block = "\n".join(entries.values())
    citation_context = _collect_all_citation_paragraphs(markdown)

    enriched = run_reference_extractor_agent(
        raw_block, citation_context=citation_context, allow_web=allow_web
    )
    abnt_list = run_reference_formatter_agent(enriched, allow_web=allow_web)

    heading = _localized_text(language, "### Referências (ABNT)", "### References (ABNT)")
    reply = f"{heading}\n\n{abnt_list}"
    return reply, {
        "intent": "resolve_numbers",
        "count": len(entries),
        "agent": "reference_extractor+reference_formatter",
    }


def _handle_list_all_references_request(
    markdown: str, user_text: str, allow_web: bool = True
) -> tuple[str, dict]:
    """Collect every reference from the document and format them in ABNT.

    Combines numbered references from the inventory with any additional raw
    lines found in reference section blocks.  All entries are normalised to
    ``[n]`` prefixes before being passed to the agent pipeline.

    Args:
        markdown: Working-copy document text from which references are read.
        user_text: User's request message, used for language detection.
        allow_web: Whether external web search is permitted during extraction
            and formatting.

    Returns:
        A 2-tuple of:

        - A localized markdown reply with the ABNT-formatted reference list.
        - A metadata dict with keys ``intent`` (``"list_all"``), ``count``
          (int), and ``agent`` (str).
    """
    language = _detect_user_language(user_text)

    inventory = _collect_reference_inventory(markdown)
    primary_refs: list[str] = list(inventory.get("references_by_number", {}).values())

    extra_lines = _collect_all_raw_references_text(markdown)
    primary_refs.extend(extra_lines)

    primary_refs = [r for r in primary_refs if not r.strip().startswith("<!--")]

    if not primary_refs:
        msg = _localized_text(
            language,
            "Nenhuma referência encontrada no documento. Verifique se o arquivo contém seções de referências.",
            "No references found in the document. Check that the file contains reference sections.",
        )
        return msg, {"intent": "list_all", "count": 0, "agent": "none"}

    numbered_lines: list[str] = []
    counter = 1
    for ref in primary_refs:
        if re.match(r"^\[\d+\]", ref):
            numbered_lines.append(ref)
        else:
            numbered_lines.append(f"[{counter}] {ref}")
        counter += 1
    raw_block = "\n".join(numbered_lines)

    citation_context: dict[int, list[str]] = _collect_all_citation_paragraphs(markdown)
    for num, paras in inventory.get("citation_paragraphs", {}).items():
        existing = citation_context.setdefault(num, [])
        for para in paras:
            if para not in existing and len(existing) < 2:
                existing.append(para)

    enriched = run_reference_extractor_agent(
        raw_block, citation_context=citation_context, allow_web=allow_web
    )

    abnt_list = run_reference_formatter_agent(enriched, allow_web=allow_web)

    heading = _localized_text(
        language,
        "### Referências do documento (ABNT)",
        "### Document references (ABNT)",
    )
    reply = f"{heading}\n\n{abnt_list}"
    meta = {
        "intent": "list_all",
        "count": len(primary_refs),
        "agent": "reference_extractor+reference_formatter",
    }
    return reply, meta


def _handle_format_provided_references_request(user_text: str, allow_web: bool) -> tuple[str, dict]:
    """Format a user-provided reference list in ABNT using the agent pipeline.

    Delegates directly to :func:`run_reference_extractor_agent` followed by
    :func:`run_reference_formatter_agent` without additional enrichment steps.

    Args:
        user_text: User's message containing the reference list to format.
        allow_web: Whether external web search is permitted during extraction
            and formatting.

    Returns:
        A 2-tuple of:

        - A localized markdown reply with the ABNT-formatted sources.
        - A metadata dict with keys ``intent`` (``"format_provided"``) and
          ``agent`` (str).
    """
    language = _detect_user_language(user_text)

    enriched = run_reference_extractor_agent(user_text, allow_web=allow_web)

    abnt_list = run_reference_formatter_agent(enriched, allow_web=allow_web)

    heading = _localized_text(
        language,
        "### Fontes formatadas (ABNT)",
        "### Formatted sources (ABNT)",
    )
    reply = f"{heading}\n\n{abnt_list}"
    meta = {
        "intent": "format_provided",
        "agent": "reference_extractor+reference_formatter",
    }
    return reply, meta


def _list_paragraphs_using_citation(markdown: str, user_text: str) -> str:
    """List all paragraphs in the working copy that contain a specific citation token.

    Args:
        markdown: Working-copy document text.
        user_text: User's query; must contain a citation number such as
            ``[2]`` for the lookup to succeed.

    Returns:
        A localized markdown string with section-level paragraph matches and
        any matching reference lines.  Returns an error message when no
        citation number can be extracted from *user_text*, or when no
        paragraphs match.
    """
    language = _detect_user_language(user_text)
    citation_number = _extract_citation_number(user_text)
    if citation_number is None:
        return _localized_text(
            language,
            "Não consegui identificar a citação pedida. Use algo como [2].",
            "I couldn't identify the requested citation. Use something like [2].",
        )

    sections = _split_sections(markdown)
    token = f"[{citation_number}]"
    matches: list[str] = []
    reference_hits: list[str] = []

    for section in sections:
        refs = section.get("references", [])
        for ref in refs:
            if ref.startswith(token):
                reference_hits.append(f"- **{section['title']}**: {ref}")

        for paragraph_index, paragraph in enumerate(section.get("paragraphs", []), start=1):
            text = paragraph.get("text", "")
            if token not in text:
                continue
            snippet = re.sub(r"\s+", " ", text).strip()
            if len(snippet) > 280:
                snippet = snippet[:277].rstrip() + "..."
            matches.append(
                _localized_text(
                    language,
                    f"- **{section['title']}**, parágrafo **{paragraph_index}**: {snippet}",
                    f"- **{section['title']}**, paragraph **{paragraph_index}**: {snippet}",
                )
            )

    if not matches:
        return _localized_text(
            language,
            f"Nenhum parágrafo na cópia de trabalho usa a citação **{token}**.",
            f"No paragraph in the working copy uses citation **{token}**.",
        )

    lines = [
        _localized_text(
            language,
            f"### Parágrafos que usam {token}",
            f"### Paragraphs using {token}",
        ),
        "",
        *matches,
    ]
    if reference_hits:
        lines += [
            "",
            _localized_text(language, "### Referência detectada", "### Detected reference"),
            "",
            *reference_hits[:8],
        ]
    return "\n".join(lines)
