"""Evaluation module for search snippet assessment."""

from .evaluators import evaluate_search_snippets, log_snippet_evaluations_to_mlflow
from .snippet_evaluators import (
    get_or_create_academic_quality_judge,
    get_or_create_citation_potential_judge,
    get_or_create_relevance_judge,
)
from .types import SnippetEvaluation

__all__ = [
    "get_or_create_relevance_judge",
    "get_or_create_academic_quality_judge",
    "get_or_create_citation_potential_judge",
    "SnippetEvaluation",
    "log_snippet_evaluations_to_mlflow",
    "evaluate_search_snippets",
]
