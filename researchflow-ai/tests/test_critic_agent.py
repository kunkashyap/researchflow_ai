"""
tests/test_critic_agent.py
Unit tests for the Critic Agent.
"""

import json
import pytest
from unittest.mock import MagicMock, patch

from models.schemas import (
    CategoryScores,
    CriticAgentOutput,
    ExtractedDocument,
    ResearchSource,
    WriterAgentOutput,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_sources():
    return [
        ResearchSource(source_id="S1", title="Quantum Paper", url="https://arxiv.org/1",
                       snippet="Quantum computing...", relevance_score=0.9),
        ResearchSource(source_id="S2", title="NIST Standards", url="https://nist.gov/pqc",
                       snippet="Post-quantum standards...", relevance_score=0.95),
    ]


@pytest.fixture
def sample_documents():
    return [
        ExtractedDocument(source_id="S1", title="Quantum Paper", url="https://arxiv.org/1",
                          extracted_text="Detailed quantum computing research content.",
                          extraction_status="success"),
        ExtractedDocument(source_id="S2", title="NIST Standards", url="https://nist.gov/pqc",
                          extracted_text="NIST post-quantum cryptography standards information.",
                          extraction_status="success"),
    ]


@pytest.fixture
def sample_writer_output():
    return WriterAgentOutput(
        research_topic="Quantum computing and cryptography",
        report_markdown="""# Quantum Computing and Cryptography

## Executive Summary
Quantum computing poses risks to RSA encryption [S1]. NIST has published new standards [S2].

## References
[S1] Quantum Paper — https://arxiv.org/1
[S2] NIST Standards — https://nist.gov/pqc
""",
        revision_number=0,
        sources_used=["S1", "S2"],
        word_count=350,
        status="completed",
    )


VALID_CRITIC_JSON = {
    "agent": "critic_agent",
    "overall_score": 72,
    "category_scores": {
        "factual_accuracy": 20,
        "research_depth": 14,
        "source_quality": 16,
        "citation_quality": 12,
        "organization": 7,
        "readability": 7,
    },
    "strengths": ["Good citation format", "Covers key topics"],
    "weaknesses": ["Lacks implementation details"],
    "unsupported_claims": [],
    "missing_information": ["Timeline of quantum threat"],
    "revision_recommendations": ["Add implementation timeline"],
    "approved": True,
    "approval_threshold": 70,
}


# ---------------------------------------------------------------------------
# Tests: JSON parsing
# ---------------------------------------------------------------------------

class TestCriticOutputParsing:

    def test_parse_valid_json(self):
        from agents.critic_agent import _parse_critic_output
        raw = json.dumps(VALID_CRITIC_JSON)
        result = _parse_critic_output(raw, approval_threshold=70)
        assert isinstance(result, CriticAgentOutput)
        assert result.overall_score == 76  # sum of categories: 20+14+16+12+7+7=76

    def test_score_computed_from_categories(self):
        """The overall score must be the SUM of category scores, not LLM self-report."""
        from agents.critic_agent import _parse_critic_output
        # LLM reports 72 but actual sum is 76
        raw = json.dumps(VALID_CRITIC_JSON)
        result = _parse_critic_output(raw, approval_threshold=70)
        expected_total = (
            VALID_CRITIC_JSON["category_scores"]["factual_accuracy"]
            + VALID_CRITIC_JSON["category_scores"]["research_depth"]
            + VALID_CRITIC_JSON["category_scores"]["source_quality"]
            + VALID_CRITIC_JSON["category_scores"]["citation_quality"]
            + VALID_CRITIC_JSON["category_scores"]["organization"]
            + VALID_CRITIC_JSON["category_scores"]["readability"]
        )
        assert result.overall_score == expected_total

    def test_approval_based_on_computed_score(self):
        from agents.critic_agent import _parse_critic_output
        low_score_json = dict(VALID_CRITIC_JSON)
        low_score_json["category_scores"] = {
            "factual_accuracy": 10, "research_depth": 8,
            "source_quality": 10, "citation_quality": 7,
            "organization": 4, "readability": 4,
        }
        raw = json.dumps(low_score_json)
        result = _parse_critic_output(raw, approval_threshold=70)
        assert result.overall_score == 43
        assert result.approved is False

    def test_parse_json_with_markdown_fence(self):
        from agents.critic_agent import _parse_critic_output
        raw = "```json\n" + json.dumps(VALID_CRITIC_JSON) + "\n```"
        result = _parse_critic_output(raw, approval_threshold=70)
        assert result.overall_score > 0

    def test_malformed_json_returns_fallback(self):
        from agents.critic_agent import _parse_critic_output
        result = _parse_critic_output("This is not JSON at all!", approval_threshold=70)
        assert isinstance(result, CriticAgentOutput)
        assert result.overall_score == 0
        assert result.approved is False
        assert result.error is not None

    def test_category_scores_capped_at_max(self):
        """LLM must not be able to exceed maximum scores by over-reporting."""
        from agents.critic_agent import _parse_critic_output
        inflated_json = dict(VALID_CRITIC_JSON)
        inflated_json["category_scores"] = {
            "factual_accuracy": 999,  # Must be capped at 25
            "research_depth": 100,    # Must be capped at 20
            "source_quality": 50,     # Must be capped at 20
            "citation_quality": 30,   # Must be capped at 15
            "organization": 20,       # Must be capped at 10
            "readability": 20,        # Must be capped at 10
        }
        raw = json.dumps(inflated_json)
        result = _parse_critic_output(raw, approval_threshold=70)
        assert result.category_scores.factual_accuracy <= 25
        assert result.category_scores.research_depth <= 20
        assert result.category_scores.source_quality <= 20
        assert result.category_scores.citation_quality <= 15
        assert result.category_scores.organization <= 10
        assert result.category_scores.readability <= 10
        assert result.overall_score <= 100


# ---------------------------------------------------------------------------
# Tests: CriticAgent.evaluate()
# ---------------------------------------------------------------------------

class TestCriticAgentEvaluate:

    @patch("agents.critic_agent.ChatOpenAI")
    def test_successful_evaluation(
        self, mock_llm_class, sample_writer_output, sample_sources, sample_documents
    ):
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = json.dumps(VALID_CRITIC_JSON)

        from agents.critic_agent import CriticAgent
        agent = CriticAgent()
        agent._chain = mock_chain

        result = agent.evaluate(
            writer_output=sample_writer_output,
            sources=sample_sources,
            documents=sample_documents,
            approval_threshold=70,
            revision_number=0,
        )

        assert isinstance(result, CriticAgentOutput)
        assert result.overall_score > 0
        assert isinstance(result.approved, bool)

    @patch("agents.critic_agent.ChatOpenAI")
    def test_evaluation_failure_returns_fallback(
        self, mock_llm_class, sample_writer_output, sample_sources, sample_documents
    ):
        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = Exception("LLM API down")

        from agents.critic_agent import CriticAgent
        agent = CriticAgent()
        agent._chain = mock_chain

        result = agent.evaluate(
            writer_output=sample_writer_output,
            sources=sample_sources,
            documents=sample_documents,
        )

        assert result.overall_score == 0
        assert result.approved is False
        assert result.error is not None

    @patch("agents.critic_agent.ChatOpenAI")
    def test_below_threshold_sets_approved_false(
        self, mock_llm_class, sample_writer_output, sample_sources, sample_documents
    ):
        low_json = dict(VALID_CRITIC_JSON)
        low_json["category_scores"] = {
            "factual_accuracy": 10, "research_depth": 5,
            "source_quality": 8, "citation_quality": 5,
            "organization": 3, "readability": 3,
        }
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = json.dumps(low_json)

        from agents.critic_agent import CriticAgent
        agent = CriticAgent()
        agent._chain = mock_chain

        result = agent.evaluate(
            writer_output=sample_writer_output,
            sources=sample_sources,
            documents=sample_documents,
            approval_threshold=70,
        )

        assert result.approved is False
