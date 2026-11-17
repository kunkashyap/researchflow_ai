"""
agents/reader_agent.py
Reader Agent — scrapes and extracts content from discovered source URLs.

Architecture:
    scrape_sources_concurrently() [BeautifulSoup + requests]
        ↓ (ExtractedDocument list with potential fallback status)
    Optional: LLM summarisation for oversized documents
        ↓
    ReaderAgentOutput (Pydantic model)

Security note:
    All scraped content is treated as UNTRUSTED DATA throughout.
    The LLM summarisation prompt explicitly frames the content as evidence,
    not instructions, to mitigate prompt injection attacks from web pages.
"""

from __future__ import annotations

from typing import Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from models.schemas import (
    ExtractedDocument,
    ReaderAgentOutput,
    SearchAgentOutput,
)
from prompts.reader_prompts import reader_summarisation_prompt
from tools.web_scraper import scrape_sources_concurrently
from utils.config import config
from utils.helpers import clean_text
from utils.llm_factory import create_llm
from utils.logging_config import setup_logging

logger = setup_logging("researchflow.reader_agent")

# Documents longer than this threshold are condensed with an LLM call
# to prevent the Writer Agent's context window from being exhausted.
SUMMARISE_THRESHOLD = 8_000  # characters


class ReaderAgent:
    """
    Reader Agent — visits URLs, extracts content, optionally summarises.

    The core extraction is done by BeautifulSoup (no LLM required).
    LLM summarisation is only triggered for excessively long documents.
    """

    def __init__(self, model_name: Optional[str] = None):
        self.llm = create_llm(temperature=0.0, model_name=model_name)
        # LCEL summarisation chain
        self._summarisation_chain = (
            reader_summarisation_prompt | self.llm | StrOutputParser()
        )

    def run(
        self,
        search_output: SearchAgentOutput,
        research_topic: str,
    ) -> ReaderAgentOutput:
        """
        Execute the full content extraction pipeline.

        Args:
            search_output:    Output from the Search Agent (contains source list).
            research_topic:   The original research topic (for LLM summarisation context).

        Returns:
            ReaderAgentOutput with extracted documents and failed source IDs.
        """
        sources = search_output.sources
        if not sources:
            logger.warning("ReaderAgent received no sources to process")
            return ReaderAgentOutput(
                extracted_documents=[],
                failed_sources=[],
                total_sources=0,
                successful_extractions=0,
                status="completed",
            )

        logger.info("ReaderAgent starting: %d sources to process", len(sources))

        # --- Step 1: Concurrent BeautifulSoup extraction ---
        extracted_docs, failed_ids = scrape_sources_concurrently(sources)

        # --- Step 2: Optional LLM summarisation for oversized content ---
        processed_docs: list[ExtractedDocument] = []
        for doc in extracted_docs:
            if (
                doc.extraction_status in ("success", "fallback")
                and len(doc.extracted_text) > SUMMARISE_THRESHOLD
            ):
                doc = self._summarise_document(doc, research_topic)
            processed_docs.append(doc)

        successful = sum(
            1 for d in processed_docs
            if d.extraction_status in ("success", "fallback")
        )

        logger.info(
            "ReaderAgent complete: %d/%d successful, %d failed",
            successful, len(sources), len(failed_ids)
        )

        return ReaderAgentOutput(
            extracted_documents=processed_docs,
            failed_sources=failed_ids,
            total_sources=len(sources),
            successful_extractions=successful,
            status="completed",
        )

    def _summarise_document(
        self,
        doc: ExtractedDocument,
        research_topic: str,
    ) -> ExtractedDocument:
        """
        Use an LLM to condense an oversized document.

        The summarisation prompt explicitly frames the content as untrusted
        evidence to mitigate prompt injection attacks embedded in web pages.
        """
        logger.info(
            "Summarising oversized document %s (%d chars)",
            doc.source_id, len(doc.extracted_text)
        )
        try:
            # Truncate input to avoid LLM context overflow
            truncated_text = doc.extracted_text[:20_000]
            summary = self._summarisation_chain.invoke({
                "research_topic": research_topic,
                "source_title": doc.title or doc.url,
                "source_url": doc.url,
                "extracted_text": truncated_text,
            })
            # Return updated doc with summarised text
            return doc.model_copy(update={
                "extracted_text": summary,
                "content_length": len(summary),
            })
        except Exception as exc:
            logger.warning(
                "LLM summarisation failed for %s: %s — keeping truncated raw text",
                doc.source_id, exc
            )
            # Keep truncated raw text rather than failing completely
            return doc.model_copy(update={
                "extracted_text": doc.extracted_text[:SUMMARISE_THRESHOLD],
                "content_length": SUMMARISE_THRESHOLD,
            })
