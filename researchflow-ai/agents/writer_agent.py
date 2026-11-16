"""
agents/writer_agent.py
Writer Agent — synthesises extracted content into a structured research report.

Architecture:
    Initial:   writer_initial_prompt | llm | StrOutputParser()
    Revision:  writer_revision_prompt | llm | StrOutputParser()

The Writer Agent operates in two modes:
1. Initial mode: Generates the first draft from scratch.
2. Revision mode: Receives Critic feedback and produces an improved report.

The LLM receives carefully framed context that:
- Provides structured source content (not raw HTML)
- Explicitly labels content as evidence from untrusted web sources
- Instructs the LLM to cite using [S1], [S2] etc.
- Forbids fabrication of facts not present in the sources
"""

from __future__ import annotations

import re
from typing import Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from models.schemas import (
    CriticAgentOutput,
    ExtractedDocument,
    ReaderAgentOutput,
    ResearchSource,
    SearchAgentOutput,
    WriterAgentOutput,
)
from prompts.writer_prompts import writer_initial_prompt, writer_revision_prompt
from utils.config import config
from utils.logging_config import setup_logging

logger = setup_logging("researchflow.writer_agent")


def _format_sources_for_llm(
    sources: list[ResearchSource],
    documents: list[ExtractedDocument],
) -> str:
    """
    Build a structured text block that presents each source's content to the LLM.

    This function carefully labels content as untrusted web evidence to guard
    against prompt injection attacks embedded in scraped web pages.
    """
    doc_map = {d.source_id: d for d in documents}
    lines: list[str] = []

    for source in sources:
        doc = doc_map.get(source.source_id)
        lines.append(f"=== SOURCE [{source.source_id}] ===")
        lines.append(f"Title: {source.title}")
        lines.append(f"URL: {source.url}")
        if source.published_date:
            lines.append(f"Published: {source.published_date}")
        lines.append(f"Relevance Score: {source.relevance_score:.2f}")

        if doc:
            status = doc.extraction_status
            if status == "failed":
                lines.append(f"Content Status: FAILED — {doc.failure_reason}")
                lines.append("NOTE: Do not cite this source for factual claims.")
            elif status == "fallback":
                lines.append(
                    f"Content Status: FALLBACK (direct scrape failed: {doc.failure_reason})"
                )
                lines.append("--- BEGIN UNTRUSTED WEB CONTENT (Tavily fallback) ---")
                lines.append(doc.extracted_text[:3000] if doc.extracted_text else "")
                lines.append("--- END UNTRUSTED WEB CONTENT ---")
            else:
                lines.append("--- BEGIN UNTRUSTED WEB CONTENT (scraped) ---")
                lines.append(doc.extracted_text[:5000] if doc.extracted_text else "")
                lines.append("--- END UNTRUSTED WEB CONTENT ---")
        else:
            lines.append(f"Content: {source.snippet}")

        lines.append("")  # blank line between sources

    return "\n".join(lines)


def _count_words(text: str) -> int:
    return len(re.findall(r'\b\w+\b', text))


def _extract_cited_sources(report: str, sources: list[ResearchSource]) -> list[str]:
    """Extract which source IDs are actually cited in the report."""
    cited = []
    for source in sources:
        if source.source_id in report:
            cited.append(source.source_id)
    return cited


class WriterAgent:
    """
    Writer Agent — generates and revises research reports using LCEL chains.
    """

    def __init__(self, model_name: Optional[str] = None):
        # Writers benefit from a slightly higher temperature for natural prose
        self.llm = ChatOpenAI(
            model=model_name or config.OPENAI_MODEL,
            temperature=0.4,
            openai_api_key=config.OPENAI_API_KEY,
            max_tokens=4096,
        )
        self._initial_chain = writer_initial_prompt | self.llm | StrOutputParser()
        self._revision_chain = writer_revision_prompt | self.llm | StrOutputParser()

    def generate_initial_report(
        self,
        research_topic: str,
        search_output: SearchAgentOutput,
        reader_output: ReaderAgentOutput,
    ) -> WriterAgentOutput:
        """
        Generate the first draft research report.

        Args:
            research_topic:  The user's research question.
            search_output:   Discovered sources from the Search Agent.
            reader_output:   Extracted content from the Reader Agent.

        Returns:
            WriterAgentOutput with the initial report.
        """
        logger.info("WriterAgent generating initial report for: '%s'", research_topic)

        formatted_sources = _format_sources_for_llm(
            search_output.sources,
            reader_output.extracted_documents,
        )

        try:
            report_markdown = self._initial_chain.invoke({
                "research_topic": research_topic,
                "search_depth": search_output.sources[0].relevance_score if search_output.sources else "standard",
                "num_sources": len(search_output.sources),
                "formatted_sources": formatted_sources,
            })
        except Exception as exc:
            logger.error("WriterAgent initial generation failed: %s", exc)
            return WriterAgentOutput(
                research_topic=research_topic,
                report_markdown="",
                revision_number=0,
                status="failed",
                error=str(exc),
            )

        cited_sources = _extract_cited_sources(report_markdown, search_output.sources)
        word_count = _count_words(report_markdown)

        logger.info(
            "WriterAgent initial report: %d words, %d sources cited",
            word_count, len(cited_sources)
        )

        return WriterAgentOutput(
            research_topic=research_topic,
            report_markdown=report_markdown,
            revision_number=0,
            sources_used=cited_sources,
            word_count=word_count,
            status="completed",
        )

    def revise_report(
        self,
        research_topic: str,
        current_report: str,
        critic_feedback: CriticAgentOutput,
        search_output: SearchAgentOutput,
        reader_output: ReaderAgentOutput,
        revision_number: int,
    ) -> WriterAgentOutput:
        """
        Produce a revised report incorporating Critic feedback.

        Args:
            research_topic:    Original research topic.
            current_report:    The current (pre-revision) report markdown.
            critic_feedback:   Structured feedback from the Critic Agent.
            search_output:     Source metadata for new citations.
            reader_output:     Extracted content for adding missing information.
            revision_number:   Which revision this is (1, 2, …).

        Returns:
            WriterAgentOutput with the revised report.
        """
        logger.info(
            "WriterAgent revision %d for: '%s' (Critic score: %d)",
            revision_number, research_topic, critic_feedback.overall_score
        )

        formatted_sources = _format_sources_for_llm(
            search_output.sources,
            reader_output.extracted_documents,
        )

        try:
            report_markdown = self._revision_chain.invoke({
                "research_topic": research_topic,
                "revision_number": revision_number,
                "current_report": current_report,
                "critic_score": critic_feedback.overall_score,
                "strengths": "\n".join(f"- {s}" for s in critic_feedback.strengths),
                "weaknesses": "\n".join(f"- {w}" for w in critic_feedback.weaknesses),
                "unsupported_claims": "\n".join(
                    f"- {c}" for c in critic_feedback.unsupported_claims
                ),
                "missing_information": "\n".join(
                    f"- {m}" for m in critic_feedback.missing_information
                ),
                "revision_recommendations": "\n".join(
                    f"- {r}" for r in critic_feedback.revision_recommendations
                ),
                "formatted_sources": formatted_sources,
            })
        except Exception as exc:
            logger.error("WriterAgent revision %d failed: %s", revision_number, exc)
            return WriterAgentOutput(
                research_topic=research_topic,
                report_markdown=current_report,  # Return previous version on failure
                revision_number=revision_number,
                status="failed",
                error=str(exc),
            )

        cited_sources = _extract_cited_sources(report_markdown, search_output.sources)
        word_count = _count_words(report_markdown)

        logger.info(
            "WriterAgent revision %d complete: %d words, %d sources cited",
            revision_number, word_count, len(cited_sources)
        )

        return WriterAgentOutput(
            research_topic=research_topic,
            report_markdown=report_markdown,
            revision_number=revision_number,
            sources_used=cited_sources,
            word_count=word_count,
            status="completed",
        )
