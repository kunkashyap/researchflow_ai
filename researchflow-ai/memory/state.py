"""
memory/state.py
Shared research state management.

ResearchState is the single container that is passed through the entire
agent pipeline. Each agent reads what it needs and returns updates that
the Orchestrator merges back into the shared state. This prevents agents
from silently sharing global mutable data while keeping the pipeline clean.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional, Callable

from models.schemas import (
    ResearchConfiguration,
    SearchAgentOutput,
    ReaderAgentOutput,
    WriterAgentOutput,
    CriticAgentOutput,
)


class ResearchState:
    """
    Centralised, mutable research session container.

    Design rationale:
    - Mutable Python class (not frozen Pydantic) so agents can update it
      in-place via the Orchestrator without full model replacement.
    - Each research session gets its own ResearchState instance so that
      concurrent sessions never share data.
    - The Orchestrator is the only component that directly mutates state.
      Agents receive a snapshot / subset and return structured outputs.
    """

    def __init__(self, research_id: str, config: ResearchConfiguration):
        # --- Identity ---
        self.research_id: str = research_id
        self.research_topic: str = config.research_topic
        self.configuration: ResearchConfiguration = config

        # --- Agent outputs (populated as pipeline progresses) ---
        self.search_output: Optional[SearchAgentOutput] = None
        self.reader_output: Optional[ReaderAgentOutput] = None
        self.writer_output: Optional[WriterAgentOutput] = None
        self.critic_output: Optional[CriticAgentOutput] = None

        # --- Revision tracking ---
        self.revision_count: int = 0
        self.revised_reports: list[WriterAgentOutput] = []
        self.critic_feedbacks: list[CriticAgentOutput] = []

        # --- Final outputs ---
        self.final_report: str = ""
        self.final_score: int = 0

        # --- Execution tracking ---
        self.status: str = "pending"  # pending | running | completed | failed
        self.error_message: Optional[str] = None
        self.execution_logs: list[str] = []
        self.agent_statuses: dict[str, str] = {
            "search_agent": "pending",
            "reader_agent": "pending",
            "writer_agent": "pending",
            "critic_agent": "pending",
        }

        # --- Timestamps ---
        self.started_at: datetime = datetime.now()
        self.completed_at: Optional[datetime] = None

        # --- Optional progress callback (used by Streamlit for real-time updates) ---
        self._progress_callback: Optional[Callable[[str, str, str], None]] = None

    # -----------------------------------------------------------------------
    # Logging helpers
    # -----------------------------------------------------------------------

    def log(self, message: str, level: str = "INFO") -> None:
        """Append a timestamped log entry to the execution log."""
        entry = f"[{datetime.now().strftime('%H:%M:%S')}] [{level}] {message}"
        self.execution_logs.append(entry)

    def set_agent_status(self, agent_name: str, status: str, message: str = "") -> None:
        """Update an individual agent's status and emit a progress callback if set."""
        self.agent_statuses[agent_name] = status
        log_msg = f"{agent_name}: {status}" + (f" — {message}" if message else "")
        self.log(log_msg)
        if self._progress_callback:
            try:
                self._progress_callback(agent_name, status, message)
            except Exception:
                pass  # Callback failures must never crash the pipeline

    def set_progress_callback(self, callback: Callable[[str, str, str], None]) -> None:
        """Register a callback function (agent_name, status, message) → None."""
        self._progress_callback = callback

    # -----------------------------------------------------------------------
    # Convenience properties
    # -----------------------------------------------------------------------

    @property
    def sources(self):
        """Shortcut to the discovered sources list."""
        if self.search_output:
            return self.search_output.sources
        return []

    @property
    def extracted_documents(self):
        """Shortcut to the extracted document list."""
        if self.reader_output:
            return self.reader_output.extracted_documents
        return []

    @property
    def current_report(self) -> str:
        """Return the most recent version of the report."""
        if self.revised_reports:
            return self.revised_reports[-1].report_markdown
        if self.writer_output:
            return self.writer_output.report_markdown
        return ""

    @property
    def latest_critic_feedback(self) -> Optional[CriticAgentOutput]:
        """Return the most recent critic evaluation."""
        if self.critic_feedbacks:
            return self.critic_feedbacks[-1]
        return self.critic_output

    def to_dict(self) -> dict:
        """Serialise the entire state to a plain dictionary for persistence."""
        return {
            "research_id": self.research_id,
            "research_topic": self.research_topic,
            "configuration": self.configuration.model_dump(),
            "search_output": self.search_output.model_dump() if self.search_output else None,
            "reader_output": self.reader_output.model_dump() if self.reader_output else None,
            "writer_output": self.writer_output.model_dump() if self.writer_output else None,
            "critic_output": self.critic_output.model_dump() if self.critic_output else None,
            "revision_count": self.revision_count,
            "final_report": self.final_report,
            "final_score": self.final_score,
            "status": self.status,
            "error_message": self.error_message,
            "execution_logs": self.execution_logs,
            "agent_statuses": self.agent_statuses,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }
