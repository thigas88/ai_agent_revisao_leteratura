"""
snippet_evaluators.py - MLflow judges for evaluating search snippets in the academic review workflow.

Impelements three judges:
1. Relevance Judge: Evaluates how relevant a search snippet is to the user's query.
2. Academic Quality Judge: Assesses whether the snippet contains technically solid information from reputable sources.
3. Citation Potential Judge: Determines if the snippet is suitable for use as a citation in an
"""

import logging

from mlflow.genai.judges import make_judge
from mlflow.genai.judges.base import Judge
from mlflow.genai.scorers import RelevanceToQuery

logger = logging.getLogger(__name__)


def get_relevance_judge() -> RelevanceToQuery:
    """Return judge built-in RelevanceToQuery for evaluating snippet relevance.

    Returns:
        An instance of RelevanceToQuery judge to evaluate the relevance of search snippets to user queries.

    Note:
        - Returns "yes" or "no"
        - Don't requires a cutomized model or instructions, as it's a built-in judge with predefined behavior.
        - Optmized for search and retrieval evaluation, providing consistent relevance assessments based on the query and snippet content."""
    return RelevanceToQuery(name="search_relevance")


def get_academic_quality_judge() -> Judge:
    """Create a judge to evaluate the academic quality of a search snippet based on technical soundness, source credibility, and alignment with user goals.

    Criteria:
        Content + Source = Solid techinical information from reputable sources.

    Returns:
        An instance of a judge to evaluate the academic quality of search snippets.
    """
    return make_judge(
        name="academic_quality",
        instructions="""Evaluate if the snippet contains technically solid information and is from respectable sources.

        Snippet: {{ outputs['snippet'] }}
        URL Domain: {{ outputs['domain'] }}
        Interview Goals: {{ outputs['user_goals'] }}

        Criteria:
        - Technical Soundness: Is the information accurate, well-explained, and follows the best practices?
        - Source Credibility: Is the information from a reputable source (e.g., academic papers, official documentation, well-known experts)?
        - Alignment with User Goals: Does the information align with the user's interview goals and help them achieve their objectives?

        Return JSON with:
        {
            "academic_quality": true|false,
            "technical_soundness": "excellent|good|poor|weak",
            "source_credibility": "high|medium|low",
            "alignment_with_user_goals": "high|medium|low",
            "reason": "A rationale for the academic quality assessment, citing specific aspects of the snippet and its source."
        }

        Don't include markdown formatting in the response, just return the JSON object as specified.
        """,
        model="openai:/gpt-4-mini",
        description="Judge to evaluate the academic quality of search snippets based on technical soundness, source credibility, and alignment with user goals.",
    )


def get_citation_potential_judge() -> Judge:
    """Create a judge to evaluate if a snippet can be used as a basis for a claim.

    Main question: Can I use this specific snippet to support/sustain an academic claim in a research paper?

    Returns:
        An instance of a judge to evaluate the citation potential of search snippets based on criteria such as
        specificity, source attribution, authority, and context preservation.
    """
    return make_judge(
        name="citation_potential",
        instructions="""Evaluate if the snippet is suitable for use as citation/evidence.

        Snippet: {{ outputs['snippet'] }}
        User Goals: {{ outputs['user_goals'] }}

        Can i use this specific snippet to support/sustain an academic claim in a research paper?

        Criteria:
        - Specificity: Does it contain concrete, quotable information?
        - Source Attribution: Is authos/publication clear?
        - Authority: Is the source authoritaive of this topic?
        - Context Preservation: Will the snippet make sense when cited?

        Return JSON with:
        {
            "citation_potential": true|false,
            "specificity_score": 0-10,
            "attribution_clarity": "clear|mediocre|absent",
            "authority_level": "high|medium|low",
            "reason": "A rationale for the citation potential assessment, citing specific aspects of the snippet and its source."
        }

        Don't include markdown formatting in the response, just return the JSON object as specified.
        """,
        model="openai:/gpt-4-mini",
        description="Judge to evaluate if a search snippet is suitable for use as a citation in an academic paper based on criteria such as specificity, source attribution, authority, and context preservation.",
    )


# Singleton instances of judges to be reused across evaluations
_relevance_judge: RelevanceToQuery | None = None
_academic_quality_judge: Judge | None = None
_citation_potential_judge: Judge | None = None


def get_or_create_relevance_judge() -> RelevanceToQuery:
    """Get or create a singleton instance of the relevance judge.

    This function ensures that only one instance of the relevance judge is created and reused across evaluations,
    optimizing resource usage and maintaining consistency in relevance assessments.

    Returns:
        An instance of the RelevanceToQuery judge, either newly created or reused from a previous
        instantiation.
    """
    global _relevance_judge
    if _relevance_judge is None:
        _relevance_judge = get_relevance_judge()
        logger.debug("Created new relevance judge instance.")
    return _relevance_judge


def get_or_create_academic_quality_judge() -> Judge:
    global _academic_quality_judge
    if _academic_quality_judge is None:
        _academic_quality_judge = get_academic_quality_judge()
        logger.debug("Created new academic quality judge instance.")
    return _academic_quality_judge


def get_or_create_citation_potential_judge() -> Judge:
    global _citation_potential_judge
    if _citation_potential_judge is None:
        _citation_potential_judge = get_citation_potential_judge()
        logger.debug("Created new citation potential judge instance.")
    return _citation_potential_judge
