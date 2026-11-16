"""
tests/test_orchestrator.py
Integration tests for the Research Orchestrator.

Tests the complete pipeline with all external APIs mocked.
"""

import json
import pytest
from unittest.mock import MagicMock, patch, call

from models.schemas import (
    CategoryScores,
    CriticAgentOutput,
    ExtractedDocument,
    ReaderAgentOutput,
    ResearchConfiguration,
    ResearchResult,
    ResearchSource,
    SearchAgentOutput,
    WriterAgentOutput,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def research_config():
    return ResearchConfiguration(
        research_topic="Artificial intelligence in healthcare",
        search_depth="quick",
        max_sources=3,
        max_revisions=2,
        approval_threshold=70,
    )


@pytest.fixture
def mock_search_output():
    return SearchAgentOutput(
        research_topic="Artificial intelligence in healthcare",
        search_queries=["AI healthcare applications", "machine learning diagnostics"],
        sources=[
            ResearchSource(source_id="S1", title="AI in Healthcare", url="https://example.com/ai-health",
                           snippet="AI is transforming healthcare...", relevance_score=0.9),
            ResearchSource(source_id="S2", title="ML Diagnostics", url="https://example.com/ml-diag",
                           snippet="Machine learning for diagnostics...", relevance_score=0.85),
        ],
        status="completed",
    )


@pytest.fixture
def mock_reader_output():
    return ReaderAgentOutput(
        extracted_documents=[
            ExtractedDocument(source_id="S1", title="AI in Healthcare", url="https://example.com/ai-health",
                              extracted_text="AI is transforming healthcare delivery [S1].",
                              extraction_status="success", content_length=200),
            ExtractedDocument(source_id="S2", title="ML Diagnostics", url="https://example.com/ml-diag",
                              extracted_text="Machine learning improves diagnostic accuracy [S2].",
                              extraction_status="success", content_length=200),
        ],
        failed_sources=[],
        total_sources=2,
        successful_extractions=2,
        status="completed",
    )


@pytest.fixture
def mock_writer_output():
    return WriterAgentOutput(
        research_topic="Artificial intelligence in healthcare",
        report_markdown="# AI in Healthcare\n\n## Executive Summary\nAI is revolutionising medicine [S1].\n\n## References\n[S1] AI in Healthcare — https://example.com/ai-health",
        revision_number=0,
        sources_used=["S1"],
        word_count=800,
        status="completed",
    )


@pytest.fixture
def mock_critic_approved():
    return CriticAgentOutput(
        overall_score=78,
        category_scores=CategoryScores(
            factual_accuracy=20, research_depth=16, source_quality=16,
            citation_quality=12, organization=8, readability=6,
        ),
        strengths=["Well cited", "Clear structure"],
        weaknesses=[],
        unsupported_claims=[],
        missing_information=[],
        revision_recommendations=[],
        approved=True,
        approval_threshold=70,
    )


@pytest.fixture
def mock_critic_rejected():
    return CriticAgentOutput(
        overall_score=55,
        category_scores=CategoryScores(
            factual_accuracy=12, research_depth=10, source_quality=12,
            citation_quality=8, organization=7, readability=6,
        ),
        strengths=["Good citations"],
        weaknesses=["Lacks depth"],
        unsupported_claims=["AI will cure all diseases"],
        missing_information=["Regulatory challenges"],
        revision_recommendations=["Add regulatory section", "Remove unsupported claims"],
        approved=False,
        approval_threshold=70,
    )


# ---------------------------------------------------------------------------
# Tests: Complete pipeline
# ---------------------------------------------------------------------------

class TestOrchestratorPipeline:

    @patch("orchestrator.research_orchestrator.save_research_session")
    @patch("orchestrator.research_orchestrator.CriticAgent")
    @patch("orchestrator.research_orchestrator.WriterAgent")
    @patch("orchestrator.research_orchestrator.ReaderAgent")
    @patch("orchestrator.research_orchestrator.SearchAgent")
    def test_successful_pipeline_approved_first_try(
        self,
        MockSearchAgent, MockReaderAgent, MockWriterAgent, MockCriticAgent,
        mock_save,
        research_config, mock_search_output, mock_reader_output,
        mock_writer_output, mock_critic_approved,
    ):
        """Test the happy path: report approved on first evaluation."""
        # Setup mocks
        MockSearchAgent.return_value.run.return_value = mock_search_output
        MockReaderAgent.return_value.run.return_value = mock_reader_output
        MockWriterAgent.return_value.generate_initial_report.return_value = mock_writer_output
        MockCriticAgent.return_value.evaluate.return_value = mock_critic_approved

        from orchestrator.research_orchestrator import ResearchOrchestrator
        orch = ResearchOrchestrator()
        result = orch.run(research_config)

        assert isinstance(result, ResearchResult)
        assert result.status == "completed"
        assert result.final_score == mock_critic_approved.overall_score
        assert result.revision_count == 0  # No revisions needed
        assert len(result.final_report) > 0

        # Verify agents were called in order
        MockSearchAgent.return_value.run.assert_called_once()
        MockReaderAgent.return_value.run.assert_called_once()
        MockWriterAgent.return_value.generate_initial_report.assert_called_once()
        MockCriticAgent.return_value.evaluate.assert_called_once()

        # Revision writer should NOT have been called
        MockWriterAgent.return_value.revise_report.assert_not_called()

        # Database should be saved
        mock_save.assert_called_once()

    @patch("orchestrator.research_orchestrator.save_research_session")
    @patch("orchestrator.research_orchestrator.CriticAgent")
    @patch("orchestrator.research_orchestrator.WriterAgent")
    @patch("orchestrator.research_orchestrator.ReaderAgent")
    @patch("orchestrator.research_orchestrator.SearchAgent")
    def test_revision_loop_triggers_on_rejection(
        self,
        MockSearchAgent, MockReaderAgent, MockWriterAgent, MockCriticAgent,
        mock_save,
        research_config, mock_search_output, mock_reader_output,
        mock_writer_output, mock_critic_rejected, mock_critic_approved,
    ):
        """Test that the Writer is asked to revise when Critic rejects the report."""
        revised_output = WriterAgentOutput(
            research_topic="Artificial intelligence in healthcare",
            report_markdown="# Revised AI Report\n\n[S1] [S2]\n## Regulatory section added",
            revision_number=1,
            sources_used=["S1", "S2"],
            word_count=1200,
            status="completed",
        )

        MockSearchAgent.return_value.run.return_value = mock_search_output
        MockReaderAgent.return_value.run.return_value = mock_reader_output
        MockWriterAgent.return_value.generate_initial_report.return_value = mock_writer_output
        MockWriterAgent.return_value.revise_report.return_value = revised_output
        # First evaluation: reject. Second: approve.
        MockCriticAgent.return_value.evaluate.side_effect = [mock_critic_rejected, mock_critic_approved]

        from orchestrator.research_orchestrator import ResearchOrchestrator
        orch = ResearchOrchestrator()
        result = orch.run(research_config)

        assert result.status == "completed"
        assert result.revision_count == 1
        MockWriterAgent.return_value.revise_report.assert_called_once()
        assert MockCriticAgent.return_value.evaluate.call_count == 2

    @patch("orchestrator.research_orchestrator.save_research_session")
    @patch("orchestrator.research_orchestrator.CriticAgent")
    @patch("orchestrator.research_orchestrator.WriterAgent")
    @patch("orchestrator.research_orchestrator.ReaderAgent")
    @patch("orchestrator.research_orchestrator.SearchAgent")
    def test_revision_loop_stops_at_max_revisions(
        self,
        MockSearchAgent, MockReaderAgent, MockWriterAgent, MockCriticAgent,
        mock_save,
        mock_search_output, mock_reader_output, mock_writer_output, mock_critic_rejected,
    ):
        """Test that the revision loop never exceeds max_revisions."""
        config = ResearchConfiguration(
            research_topic="Test topic",
            max_revisions=2,
            max_sources=3,
        )

        revised_output = WriterAgentOutput(
            research_topic="Test topic",
            report_markdown="# Revised",
            revision_number=1,
            status="completed",
        )

        MockSearchAgent.return_value.run.return_value = mock_search_output
        MockReaderAgent.return_value.run.return_value = mock_reader_output
        MockWriterAgent.return_value.generate_initial_report.return_value = mock_writer_output
        MockWriterAgent.return_value.revise_report.return_value = revised_output
        # Always reject — should still stop at max_revisions
        MockCriticAgent.return_value.evaluate.return_value = mock_critic_rejected

        from orchestrator.research_orchestrator import ResearchOrchestrator
        orch = ResearchOrchestrator()
        result = orch.run(config)

        # Critic called once per round: initial + max_revisions
        assert MockCriticAgent.return_value.evaluate.call_count == config.max_revisions + 1
        # Writer revise called max_revisions times
        assert MockWriterAgent.return_value.revise_report.call_count == config.max_revisions
        # Still completes (not failed)
        assert result.status == "completed"

    @patch("orchestrator.research_orchestrator.save_research_session")
    @patch("orchestrator.research_orchestrator.SearchAgent")
    def test_search_failure_marks_research_failed(
        self, MockSearchAgent, mock_save, research_config
    ):
        """Test that a SearchAgent failure marks the entire research as failed."""
        from models.schemas import SearchAgentOutput
        MockSearchAgent.return_value.run.return_value = SearchAgentOutput(
            research_topic="test",
            status="failed",
            error="Tavily API down",
            sources=[],
        )

        from orchestrator.research_orchestrator import ResearchOrchestrator
        orch = ResearchOrchestrator()
        result = orch.run(research_config)

        assert result.status == "failed"
        assert "SearchAgent failed" in result.error_message
        # Database should still be saved (for debugging)
        mock_save.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: Shared memory isolation
# ---------------------------------------------------------------------------

class TestSharedMemoryIsolation:

    def test_separate_states_do_not_share_data(self):
        """Two ResearchState instances must not share any mutable data."""
        from memory.state import ResearchState
        from models.schemas import ResearchConfiguration

        config_a = ResearchConfiguration(research_topic="Topic A", max_sources=3)
        config_b = ResearchConfiguration(research_topic="Topic B", max_sources=3)

        state_a = ResearchState(research_id="A001", config=config_a)
        state_b = ResearchState(research_id="B001", config=config_b)

        state_a.log("Log entry for A")
        assert len(state_b.execution_logs) == 0

        state_a.agent_statuses["search_agent"] = "completed"
        assert state_b.agent_statuses["search_agent"] == "pending"

    def test_research_id_uniqueness(self):
        """generate_research_id should produce different IDs for different topics."""
        from utils.helpers import generate_research_id
        id1 = generate_research_id("Topic A")
        id2 = generate_research_id("Topic B")
        assert id1 != id2
        assert id1.startswith("RES_")
        assert id2.startswith("RES_")


# ---------------------------------------------------------------------------
# Tests: SQLite persistence
# ---------------------------------------------------------------------------

class TestSQLitePersistence:

    def test_save_and_retrieve_session(self, tmp_path, monkeypatch):
        """Test that a session can be saved and retrieved from SQLite."""
        from memory import database

        # Redirect DB to a temporary file for this test
        test_db = tmp_path / "test_history.db"
        monkeypatch.setattr(database, "DB_PATH", test_db)
        database.initialise_db()

        test_state = {
            "research_id": "TEST_001",
            "research_topic": "Test topic",
            "status": "completed",
            "final_score": 75,
            "revision_count": 1,
            "configuration": {"research_topic": "Test topic", "max_sources": 5},
            "search_output": None,
            "reader_output": None,
            "writer_output": None,
            "critic_output": None,
            "final_report": "# Test Report\n\nContent here.",
            "execution_logs": ["[00:00:00] [INFO] Started", "[00:01:00] [INFO] Done"],
            "error_message": None,
            "started_at": "2024-01-01T00:00:00",
            "completed_at": "2024-01-01T00:05:00",
        }

        database.save_research_session(test_state)
        retrieved = database.get_session_by_id("TEST_001")

        assert retrieved is not None
        assert retrieved["research_id"] == "TEST_001"
        assert retrieved["research_topic"] == "Test topic"
        assert retrieved["final_score"] == 75
        assert retrieved["final_report"] == "# Test Report\n\nContent here."
        assert isinstance(retrieved["execution_logs"], list)
        assert len(retrieved["execution_logs"]) == 2

    def test_get_all_sessions(self, tmp_path, monkeypatch):
        """Test that multiple sessions are returned in reverse chronological order."""
        from memory import database

        test_db = tmp_path / "test_history2.db"
        monkeypatch.setattr(database, "DB_PATH", test_db)
        database.initialise_db()

        for i in range(3):
            database.save_research_session({
                "research_id": f"TEST_00{i}",
                "research_topic": f"Topic {i}",
                "status": "completed",
                "final_score": 60 + i * 10,
                "revision_count": 0,
                "configuration": {},
                "search_output": None, "reader_output": None,
                "writer_output": None, "critic_output": None,
                "final_report": "", "execution_logs": [],
                "error_message": None,
                "started_at": f"2024-01-0{i+1}T00:00:00",
                "completed_at": None,
            })

        sessions = database.get_all_sessions()
        assert len(sessions) == 3

    def test_delete_session(self, tmp_path, monkeypatch):
        """Test session deletion."""
        from memory import database

        test_db = tmp_path / "test_history3.db"
        monkeypatch.setattr(database, "DB_PATH", test_db)
        database.initialise_db()

        database.save_research_session({
            "research_id": "DEL_001",
            "research_topic": "Delete me",
            "status": "completed",
            "final_score": 70,
            "revision_count": 0,
            "configuration": {}, "search_output": None, "reader_output": None,
            "writer_output": None, "critic_output": None,
            "final_report": "", "execution_logs": [], "error_message": None,
            "started_at": "2024-01-01T00:00:00", "completed_at": None,
        })

        assert database.get_session_by_id("DEL_001") is not None
        deleted = database.delete_session("DEL_001")
        assert deleted is True
        assert database.get_session_by_id("DEL_001") is None
