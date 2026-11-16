"""
agents/critic_agent.py
Critic Agent — rigorously evaluates the research report and provides scored feedback.

Architecture:
    critic_prompt | llm | StrOutputParser()
        → _parse_critic_output()  (JSON parsing with fallback)
        → CriticAgentOutput (Pydantic model)

The Critic Agent:
1. Receives the full report and all source content.
2. Scores the report on six weighted criteria (total 100 points).
3. Identifies specific weaknesses with evidence.
4. Returns revision recommendations if the report doesn't meet the threshold.
5. Sets `approved = True` only if the score meets or exceeds the threshold.
"""

from __future__ import annotations

import json
import re
from typing import Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from models.schemas import (
    CategoryScores,
    CriticAgentOutput,
    ExtractedDocument,
    ResearchSource,
    WriterAgentOutput,
)
from prompts.critic_prompts import critic_prompt
from utils.config import config
from utils.logging_config import setup_logging

logger = setup_logging("researchflow.critic_agent")


def _format_sources_brief(
    sources: list[ResearchSource],
    documents: list[ExtractedDocument],
    max_chars_per_doc: int = 2000,
) -> str:
    """
    Format sources for the Critic's context window.
    Uses shorter excerpts than the Writer's format to stay within token limits.
    """
    doc_map = {d.source_id: d for d in documents}
    lines: list[str] = []

    for source in sources:
        doc = doc_map.get(source.source_id)
        lines.append(f"[{source.source_id}] {source.title}")
        lines.append(f"URL: {source.url}")
        if doc and doc.extraction_status != "failed" and doc.extracted_text:
            excerpt = doc.extracted_text[:max_chars_per_doc]
            lines.append(f"Content excerpt: {excerpt}")
        else:
            lines.append(f"Content: {source.snippet}")
        lines.append("")

    return "\n".join(lines)


def _parse_critic_output(raw_json: str, approval_threshold: int) -> CriticAgentOutput:
    """
    Parse the LLM's JSON response into a CriticAgentOutput.

    Implements multiple fallback strategies because LLMs sometimes produce
    slightly malformed JSON (trailing commas, missing braces, etc.).
    """
    # Strip markdown code fences
    cleaned = re.sub(r"```(?:json)?", "", raw_json).strip().strip("`")

    # Find the outermost JSON object
    start = cleaned.find("{")
    end = cleaned.rfind("}") + 1
    if start != -1 and end > start:
        cleaned = cleaned[start:end]

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse Critic JSON: %s\nRaw: %s", exc, raw_json[:500])
        # Return a default failing evaluation rather than crashing
        return CriticAgentOutput(
            overall_score=0,
            category_scores=CategoryScores(),
            strengths=[],
            weaknesses=["Critic evaluation failed due to malformed JSON output"],
            unsupported_claims=[],
            missing_information=[],
            revision_recommendations=["Re-evaluate the report manually"],
            approved=False,
            approval_threshold=approval_threshold,
            error=f"JSON parse error: {exc}",
        )

    # Parse category scores
    raw_cats = data.get("category_scores", {})
    category_scores = CategoryScores(
        factual_accuracy=min(int(raw_cats.get("factual_accuracy", 0)), 25),
        research_depth=min(int(raw_cats.get("research_depth", 0)), 20),
        source_quality=min(int(raw_cats.get("source_quality", 0)), 20),
        citation_quality=min(int(raw_cats.get("citation_quality", 0)), 15),
        organization=min(int(raw_cats.get("organization", 0)), 10),
        readability=min(int(raw_cats.get("readability", 0)), 10),
    )

    # Use the computed total (sum of categories) rather than the LLM's self-reported total
    # to prevent the LLM from inflating its own score
    overall_score = category_scores.total
    approved = overall_score >= approval_threshold

    return CriticAgentOutput(
        overall_score=overall_score,
        category_scores=category_scores,
        strengths=data.get("strengths", []),
        weaknesses=data.get("weaknesses", []),
        unsupported_claims=data.get("unsupported_claims", []),
        missing_information=data.get("missing_information", []),
        revision_recommendations=data.get("revision_recommendations", []),
        approved=approved,
        approval_threshold=approval_threshold,
    )


class CriticAgent:
    """
    Critic Agent — evaluates research report quality against source material.
    """

    def __init__(self, model_name: Optional[str] = None):
        # Critics should be strict and analytical — lower temperature
        self.llm = ChatOpenAI(
            model=model_name or config.OPENAI_MODEL,
            temperature=0.1,
            openai_api_key=config.OPENAI_API_KEY,
            max_tokens=2048,
        )
        self._chain = critic_prompt | self.llm | StrOutputParser()

    def evaluate(
        self,
        writer_output: WriterAgentOutput,
        sources: list[ResearchSource],
        documents: list[ExtractedDocument],
        approval_threshold: int = None,
        revision_number: int = 0,
    ) -> CriticAgentOutput:
        """
        Evaluate a research report and return a structured critique.

        Args:
            writer_output:        The WriterAgent's output containing the report.
            sources:              Original source metadata (for fact-checking).
            documents:            Extracted content (for verifying claims).
            approval_threshold:   Minimum score to approve (defaults to config).
            revision_number:      Which revision this evaluation is for.

        Returns:
            CriticAgentOutput with scores, feedback and approval decision.
        """
        threshold = approval_threshold or config.APPROVAL_THRESHOLD

        logger.info(
            "CriticAgent evaluating revision %d (threshold: %d)",
            revision_number, threshold
        )

        formatted_sources = _format_sources_brief(sources, documents)

        try:
            raw_output = self._chain.invoke({
                "research_topic": writer_output.research_topic,
                "formatted_sources": formatted_sources,
                "report_markdown": writer_output.report_markdown,
                "revision_number": revision_number,
                "approval_threshold": threshold,
            })
        except Exception as exc:
            logger.error("CriticAgent evaluation failed: %s", exc)
            return CriticAgentOutput(
                overall_score=0,
                approved=False,
                approval_threshold=threshold,
                error=str(exc),
                weaknesses=[f"Evaluation failed: {exc}"],
                revision_recommendations=["Retry the evaluation"],
            )

        result = _parse_critic_output(raw_output, threshold)

        logger.info(
            "CriticAgent complete: score=%d/%d, approved=%s",
            result.overall_score, threshold, result.approved
        )

        return result
