"""
models/schemas.py
Pydantic schemas for all structured agent inputs and outputs.

Using Pydantic v2 models ensures that data passed between agents is
type-checked, serialisable to JSON, and self-documenting. Each agent
has a dedicated output model that defines exactly what it produces.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Source Models (produced by Search Agent)
# ---------------------------------------------------------------------------

class ResearchSource(BaseModel):
    """A single web source discovered by the Search Agent."""

    source_id: str = Field(..., description="Unique identifier, e.g. 'S1'")
    title: str = Field(default="", description="Page title")
    url: str = Field(..., description="Source URL")
    snippet: str = Field(default="", description="Brief excerpt from Tavily")
    published_date: Optional[str] = Field(default=None, description="ISO date string if available")
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    # Tavily may return raw page content; store it as a fallback for the Reader Agent
    tavily_content: Optional[str] = Field(default=None, description="Raw content from Tavily (fallback)")


class SearchAgentOutput(BaseModel):
    """Full output of the Search Agent."""

    agent: str = Field(default="search_agent")
    research_topic: str
    search_queries: list[str] = Field(default_factory=list)
    sources: list[ResearchSource] = Field(default_factory=list)
    status: str = Field(default="completed")
    error: Optional[str] = Field(default=None)


# ---------------------------------------------------------------------------
# Extracted Document Models (produced by Reader Agent)
# ---------------------------------------------------------------------------

class ExtractedDocument(BaseModel):
    """Content extracted from a single URL by the Reader Agent."""

    source_id: str
    title: str = Field(default="")
    url: str
    extracted_text: str = Field(default="")
    headings: list[str] = Field(default_factory=list)
    extraction_status: str = Field(
        default="success",
        description="success | failed | partial | fallback"
    )
    failure_reason: Optional[str] = Field(default=None)
    content_length: int = Field(default=0)


class ReaderAgentOutput(BaseModel):
    """Full output of the Reader Agent."""

    agent: str = Field(default="reader_agent")
    extracted_documents: list[ExtractedDocument] = Field(default_factory=list)
    failed_sources: list[str] = Field(default_factory=list, description="List of source IDs that failed")
    total_sources: int = Field(default=0)
    successful_extractions: int = Field(default=0)
    status: str = Field(default="completed")
    error: Optional[str] = Field(default=None)


# ---------------------------------------------------------------------------
# Writer Agent Models
# ---------------------------------------------------------------------------

class WriterAgentOutput(BaseModel):
    """Full output of the Writer Agent (initial or revised report)."""

    agent: str = Field(default="writer_agent")
    research_topic: str
    report_markdown: str = Field(..., description="Full report in Markdown format")
    revision_number: int = Field(default=0, description="0 = initial, 1+ = revisions")
    sources_used: list[str] = Field(default_factory=list, description="Source IDs referenced in report")
    word_count: int = Field(default=0)
    status: str = Field(default="completed")
    error: Optional[str] = Field(default=None)


# ---------------------------------------------------------------------------
# Critic Agent Models
# ---------------------------------------------------------------------------

class CategoryScores(BaseModel):
    """Detailed scoring breakdown from the Critic Agent."""

    factual_accuracy: int = Field(default=0, ge=0, le=25, description="Max 25")
    research_depth: int = Field(default=0, ge=0, le=20, description="Max 20")
    source_quality: int = Field(default=0, ge=0, le=20, description="Max 20")
    citation_quality: int = Field(default=0, ge=0, le=15, description="Max 15")
    organization: int = Field(default=0, ge=0, le=10, description="Max 10")
    readability: int = Field(default=0, ge=0, le=10, description="Max 10")

    @property
    def total(self) -> int:
        return (
            self.factual_accuracy
            + self.research_depth
            + self.source_quality
            + self.citation_quality
            + self.organization
            + self.readability
        )


class CriticAgentOutput(BaseModel):
    """Full output of the Critic Agent."""

    agent: str = Field(default="critic_agent")
    overall_score: int = Field(default=0, ge=0, le=100)
    category_scores: CategoryScores = Field(default_factory=CategoryScores)
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    revision_recommendations: list[str] = Field(default_factory=list)
    approved: bool = Field(default=False)
    approval_threshold: int = Field(default=70)
    error: Optional[str] = Field(default=None)


# ---------------------------------------------------------------------------
# Research Configuration
# ---------------------------------------------------------------------------

class ResearchConfiguration(BaseModel):
    """User-specified research parameters."""

    research_topic: str
    search_depth: str = Field(default="standard", pattern="^(quick|standard|deep)$")
    max_sources: int = Field(default=8, ge=2, le=20)
    max_revisions: int = Field(default=2, ge=0, le=5)
    approval_threshold: int = Field(default=70, ge=0, le=100)
    date_range: Optional[str] = Field(default=None, description="e.g. '2023-01-01 to 2024-01-01'")

    @property
    def num_queries(self) -> int:
        """Generate more search queries for deeper research depth."""
        return {"quick": 2, "standard": 4, "deep": 6}.get(self.search_depth, 4)

    @property
    def tavily_search_depth(self) -> str:
        """Map internal depth to Tavily API depth parameter."""
        return "advanced" if self.search_depth == "deep" else "basic"


# ---------------------------------------------------------------------------
# Final Research Result
# ---------------------------------------------------------------------------

class ResearchResult(BaseModel):
    """
    The complete result of a research session, persisted to SQLite
    and returned to the frontend.
    """

    research_id: str
    research_topic: str
    configuration: ResearchConfiguration
    search_output: Optional[SearchAgentOutput] = None
    reader_output: Optional[ReaderAgentOutput] = None
    writer_output: Optional[WriterAgentOutput] = None
    critic_output: Optional[CriticAgentOutput] = None
    final_report: str = Field(default="")
    final_score: int = Field(default=0)
    revision_count: int = Field(default=0)
    status: str = Field(default="pending")  # pending | running | completed | failed
    error_message: Optional[str] = None
    started_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    execution_logs: list[str] = Field(default_factory=list)
