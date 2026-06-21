from typing import Literal

from pydantic import BaseModel, Field


class SnippetEvaluation(BaseModel):
    """Data model for evaluating a snippet."""

    snippet: str = Field(..., description="The snippet to be evaluated")

    # Relevance (Use RelevanceToQuery)
    relevance_level: Literal[
        "Perfectly relevant",
        "Partially relevant",
        "Not relevant",
    ] = Field(..., description="The relevance level of the snippet")
    relevance_score: float = Field(
        ...,
        description="A score from 0 to 1 indicating the relevance of the snippet",
        ge=0.0,
        le=1.0,
    )
    relevance_rationale: str = Field(..., description="A rationale for the relevance score")

    # Academic quality
    academic_quality: bool = Field(..., description="Whether the snippet has academic quality")
    academic_quality_rationale: str = Field(
        ..., description="A rationale for the academic quality assessment"
    )

    # Citation potential
    citation_potential: bool = Field(
        ..., description="Whether the snippet has potential to be cited in an academic paper"
    )
    citation_potential_rationale: str = Field(
        ..., description="A rationale for the citation potential assessment"
    )

    # Metadata
    url: str = Field(..., description="The URL from which the snippet was extracted")
    domain: str = Field(..., description="The domain of the URL")
