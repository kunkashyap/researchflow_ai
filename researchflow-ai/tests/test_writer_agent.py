"""
tests/test_writer_agent.py
Unit tests for the Writer Agent.
"""

import pytest
from unittest.mock import MagicMock, patch

from models.schemas import (
    CriticAgentOutput,
    CategoryScores,
    ExtractedDocument,
    ReaderAgentOutput,
    ResearchSource,
    SearchAgentOutput,
    WriterAgentOutput,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_sources():
    return [
        ResearchSource(source_id="S1", title="Source 1", url="https://example.com/1",
                       snippet="Quantum computing content", relevance_score=0.9),
        ResearchSource(source_id="S2", title="Source 2", url="https://example.com/2",
                       snippet="Cryptography research content", relevance_score=0.85),
    ]


@pytest.fixture
def sample_documents():
    return [
        ExtractedDocument(source_id="S1", title="Source 1", url="https://example.com/1",
                          extracted_text="Detailed quantum computing information from source 1.",
                          extraction_status="success", content_length=100),
        ExtractedDocument(source_id="S2", title="Source 2", url="https://example.com/2",
                          extracted_text="Detailed cryptography research from source 2.",
                          extraction_status="success", content_length=100),
    ]


@pytest.fixture
def sample_search_output(sample_sources):
    return SearchAgentOutput(
        research_topic="Quantum computing and cryptography",
        search_queries=["quantum cryptography", "post-quantum encryption"],
        sources=sample_sources,
        status="completed",
    )


@pytest.fixture
def sample_reader_output(sample_documents):
    return ReaderAgentOutput(
        extracted_documents=sample_documents,
        failed_sources=[],
        total_sources=2,
        successful_extractions=2,
        status="completed",
    )


SAMPLE_REPORT = """# Quantum Computing and Cryptography

## Executive Summary
Research shows quantum computing poses significant risks [S1]. Post-quantum standards are being developed [S2].

## Introduction
Quantum computing represents a paradigm shift [S1].

## References
[S1] Source 1 — https://example.com/1
[S2] Source 2 — https://example.com/2
"""


@pytest.fixture
def sample_critic_feedback():
    return CriticAgentOutput(
        overall_score=65,
        category_scores=CategoryScores(
            factual_accuracy=18,
            research_depth=12,
            source_quality=15,
            citation_quality=10,
            organization=5,
            readability=5,
        ),
        strengths=["Good citation format", "Clear introduction"],
        weaknesses=["Missing analysis section", "Insufficient depth on NIST standards"],
        unsupported_claims=["RSA-2048 can be broken in days"],
        missing_information=["NIST PQC algorithm details", "Timeline of quantum threat"],
        revision_recommendations=[
            "Add a detailed section on NIST post-quantum standards",
            "Cite sources for the RSA-2048 claim",
        ],
        approved=False,
        approval_threshold=70,
    )


# ---------------------------------------------------------------------------
# Tests: Source formatting
# ---------------------------------------------------------------------------

class TestSourceFormatting:

    def test_formats_all_sources(self, sample_sources, sample_documents):
        from agents.writer_agent import _format_sources_for_llm
        result = _format_sources_for_llm(sample_sources, sample_documents)
        assert "S1" in result
        assert "S2" in result
        assert "example.com/1" in result

    def test_labels_failed_sources(self, sample_documents):
        from agents.writer_agent import _format_sources_for_llm
        failed_doc = ExtractedDocument(
            source_id="S3", title="Failed", url="https://example.com/3",
            extraction_status="failed", failure_reason="Connection timeout",
        )
        sources = [ResearchSource(source_id="S3", title="Failed", url="https://example.com/3")]
        result = _format_sources_for_llm(sources, [failed_doc])
        assert "FAILED" in result
        assert "Do not cite" in result

    def test_labels_fallback_sources(self, sample_documents):
        from agents.writer_agent import _format_sources_for_llm
        fallback_doc = ExtractedDocument(
            source_id="S4", title="Fallback", url="https://example.com/4",
            extracted_text="Tavily provided content",
            extraction_status="fallback", failure_reason="robots.txt disallowed",
        )
        sources = [ResearchSource(source_id="S4", title="Fallback", url="https://example.com/4")]
        result = _format_sources_for_llm(sources, [fallback_doc])
        assert "FALLBACK" in result

    def test_marks_content_as_untrusted(self, sample_sources, sample_documents):
        from agents.writer_agent import _format_sources_for_llm
        result = _format_sources_for_llm(sample_sources, sample_documents)
        assert "UNTRUSTED" in result


# ---------------------------------------------------------------------------
# Tests: WriterAgent.generate_initial_report()
# ---------------------------------------------------------------------------

class TestWriterAgentInitialReport:

    @patch("agents.writer_agent.ChatOpenAI")
    def test_successful_generation(self, mock_llm_class, sample_search_output, sample_reader_output):
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = SAMPLE_REPORT

        from agents.writer_agent import WriterAgent
        agent = WriterAgent()
        agent._initial_chain = mock_chain

        result = agent.generate_initial_report(
            research_topic="Quantum computing and cryptography",
            search_output=sample_search_output,
            reader_output=sample_reader_output,
        )

        assert isinstance(result, WriterAgentOutput)
        assert result.status == "completed"
        assert result.revision_number == 0
        assert len(result.report_markdown) > 0

    @patch("agents.writer_agent.ChatOpenAI")
    def test_cites_sources_in_report(self, mock_llm_class, sample_search_output, sample_reader_output):
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = SAMPLE_REPORT

        from agents.writer_agent import WriterAgent
        agent = WriterAgent()
        agent._initial_chain = mock_chain

        result = agent.generate_initial_report(
            research_topic="Quantum computing",
            search_output=sample_search_output,
            reader_output=sample_reader_output,
        )

        assert "S1" in result.sources_used or "S2" in result.sources_used

    @patch("agents.writer_agent.ChatOpenAI")
    def test_llm_failure_returns_failed_output(self, mock_llm_class, sample_search_output, sample_reader_output):
        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = Exception("OpenAI API unavailable")

        from agents.writer_agent import WriterAgent
        agent = WriterAgent()
        agent._initial_chain = mock_chain

        result = agent.generate_initial_report(
            research_topic="Quantum computing",
            search_output=sample_search_output,
            reader_output=sample_reader_output,
        )

        assert result.status == "failed"
        assert "OpenAI API unavailable" in result.error


# ---------------------------------------------------------------------------
# Tests: WriterAgent.revise_report()
# ---------------------------------------------------------------------------

class TestWriterAgentRevision:

    @patch("agents.writer_agent.ChatOpenAI")
    def test_revision_incorporates_feedback(
        self, mock_llm_class, sample_search_output, sample_reader_output, sample_critic_feedback
    ):
        revised_report = SAMPLE_REPORT + "\n## NIST PQC Standards\nDetailed information added [S2]."
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = revised_report

        from agents.writer_agent import WriterAgent
        agent = WriterAgent()
        agent._revision_chain = mock_chain

        result = agent.revise_report(
            research_topic="Quantum computing",
            current_report=SAMPLE_REPORT,
            critic_feedback=sample_critic_feedback,
            search_output=sample_search_output,
            reader_output=sample_reader_output,
            revision_number=1,
        )

        assert result.status == "completed"
        assert result.revision_number == 1
        assert len(result.report_markdown) > len(SAMPLE_REPORT)

    @patch("agents.writer_agent.ChatOpenAI")
    def test_revision_failure_returns_previous_report(
        self, mock_llm_class, sample_search_output, sample_reader_output, sample_critic_feedback
    ):
        """When revision fails, the previous report should be returned (not empty string)."""
        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = Exception("API error")

        from agents.writer_agent import WriterAgent
        agent = WriterAgent()
        agent._revision_chain = mock_chain

        result = agent.revise_report(
            research_topic="Quantum computing",
            current_report=SAMPLE_REPORT,
            critic_feedback=sample_critic_feedback,
            search_output=sample_search_output,
            reader_output=sample_reader_output,
            revision_number=1,
        )

        # Should return the PREVIOUS report, not an empty string
        assert result.report_markdown == SAMPLE_REPORT
        assert result.status == "failed"


# ---------------------------------------------------------------------------
# Tests: Word count and citation extraction
# ---------------------------------------------------------------------------

class TestWordCountAndCitations:

    def test_word_count_calculation(self):
        from agents.writer_agent import _count_words
        text = "This is a test sentence with seven words."
        assert _count_words(text) == 8

    def test_citation_extraction(self):
        from agents.writer_agent import _extract_cited_sources
        sources = [
            ResearchSource(source_id="S1", url="https://a.com"),
            ResearchSource(source_id="S2", url="https://b.com"),
            ResearchSource(source_id="S3", url="https://c.com"),
        ]
        report = "According to [S1] and [S2], this is true."
        cited = _extract_cited_sources(report, sources)
        assert "S1" in cited
        assert "S2" in cited
        assert "S3" not in cited
