"""
orchestrator/research_orchestrator.py
Central coordinator for the ResearchFlow AI multi-agent pipeline.

The Orchestrator is responsible for:
1. Initialising the shared ResearchState.
2. Invoking each agent in sequence.
3. Passing outputs between agents through structured state.
4. Managing the Writer↔Critic revision loop (bounded by max_revisions).
5. Persisting results to SQLite after completion.
6. Returning the final ResearchResult to the caller (Streamlit).

Pipeline:
    User Query
        ↓
    SearchAgent.run()     → state.search_output
        ↓
    ReaderAgent.run()     → state.reader_output
        ↓
    WriterAgent.generate_initial_report()  → state.writer_output
        ↓
    CriticAgent.evaluate()  → state.critic_output
        ↓ (loop if not approved AND revisions_remaining > 0)
    WriterAgent.revise_report()  → state.revised_reports[-1]
    CriticAgent.evaluate()  → state.critic_feedbacks[-1]
        ↓ (loop ends)
    Persist → SQLite
    Return ResearchResult
"""

from __future__ import annotations

import traceback
from datetime import datetime
from typing import Callable, Optional

from agents.critic_agent import CriticAgent
from agents.reader_agent import ReaderAgent
from agents.search_agent import SearchAgent
from agents.writer_agent import WriterAgent
from memory.database import save_research_session
from memory.state import ResearchState
from models.schemas import ResearchConfiguration, ResearchResult
from utils.helpers import generate_research_id
from utils.logging_config import setup_logging

logger = setup_logging("researchflow.orchestrator")


class ResearchOrchestrator:
    """
    Coordinates the four-agent research pipeline with a bounded revision loop.

    Usage:
        orchestrator = ResearchOrchestrator()
        result = orchestrator.run(config, progress_callback=my_callback)
    """

    def __init__(self):
        # Instantiate agents once — they are stateless and reusable
        self.search_agent = SearchAgent()
        self.reader_agent = ReaderAgent()
        self.writer_agent = WriterAgent()
        self.critic_agent = CriticAgent()

    def run(
        self,
        research_config: ResearchConfiguration,
        progress_callback: Optional[Callable[[str, str, str], None]] = None,
    ) -> ResearchResult:
        """
        Execute the full research pipeline.

        Args:
            research_config:    User-specified research parameters.
            progress_callback:  Optional function(agent_name, status, message)
                                called after each agent completes. Used by
                                Streamlit to update the UI in real-time.

        Returns:
            ResearchResult containing all agent outputs and the final report.
        """
        # --- Initialise shared state ---
        research_id = generate_research_id(research_config.research_topic)
        state = ResearchState(research_id=research_id, config=research_config)
        state.status = "running"

        if progress_callback:
            state.set_progress_callback(progress_callback)

        logger.info(
            "Orchestrator starting research: ID=%s, topic='%s'",
            research_id, research_config.research_topic
        )
        state.log(f"Research session started: {research_id}")

        try:
            # ================================================================
            # PHASE 1: SEARCH AGENT
            # ================================================================
            state.set_agent_status("search_agent", "running", "Generating queries and searching the web...")
            state.log("Invoking SearchAgent")

            search_output = self.search_agent.run(research_config)
            state.search_output = search_output

            if search_output.status == "failed":
                raise RuntimeError(f"SearchAgent failed: {search_output.error}")

            if not search_output.sources:
                raise RuntimeError("SearchAgent returned no sources. Cannot continue.")

            state.set_agent_status(
                "search_agent", "completed",
                f"Found {len(search_output.sources)} sources using {len(search_output.search_queries)} queries"
            )
            state.log(f"SearchAgent: {len(search_output.sources)} sources discovered")

            # ================================================================
            # PHASE 2: READER AGENT
            # ================================================================
            state.set_agent_status("reader_agent", "running", "Extracting content from sources...")
            state.log("Invoking ReaderAgent")

            reader_output = self.reader_agent.run(
                search_output=search_output,
                research_topic=research_config.research_topic,
            )
            state.reader_output = reader_output

            state.set_agent_status(
                "reader_agent", "completed",
                f"Extracted {reader_output.successful_extractions}/{reader_output.total_sources} sources"
                + (f" ({len(reader_output.failed_sources)} failed)" if reader_output.failed_sources else "")
            )
            state.log(
                f"ReaderAgent: {reader_output.successful_extractions} successful, "
                f"{len(reader_output.failed_sources)} failed"
            )

            # ================================================================
            # PHASE 3: WRITER AGENT (Initial Report)
            # ================================================================
            state.set_agent_status("writer_agent", "running", "Writing initial research report...")
            state.log("Invoking WriterAgent (initial)")

            writer_output = self.writer_agent.generate_initial_report(
                research_topic=research_config.research_topic,
                search_output=search_output,
                reader_output=reader_output,
            )
            state.writer_output = writer_output

            if writer_output.status == "failed":
                raise RuntimeError(f"WriterAgent failed: {writer_output.error}")

            state.set_agent_status(
                "writer_agent", "completed",
                f"Initial report: {writer_output.word_count} words, {len(writer_output.sources_used)} sources cited"
            )
            state.log(f"WriterAgent: initial report generated ({writer_output.word_count} words)")

            # ================================================================
            # PHASE 4: CRITIC AGENT + REVISION LOOP
            # ================================================================
            current_report = writer_output
            max_revisions = research_config.max_revisions
            approval_threshold = research_config.approval_threshold

            for revision_round in range(max_revisions + 1):
                # Evaluate current report
                state.set_agent_status(
                    "critic_agent", "running",
                    f"Evaluating report (round {revision_round + 1}/{max_revisions + 1})..."
                )
                state.log(f"CriticAgent: evaluating revision {revision_round}")

                critic_output = self.critic_agent.evaluate(
                    writer_output=current_report,
                    sources=search_output.sources,
                    documents=reader_output.extracted_documents,
                    approval_threshold=approval_threshold,
                    revision_number=revision_round,
                )
                state.critic_output = critic_output
                state.critic_feedbacks.append(critic_output)

                state.set_agent_status(
                    "critic_agent", "completed",
                    f"Score: {critic_output.overall_score}/100 — "
                    + ("✓ Approved" if critic_output.approved else "✗ Needs revision")
                )
                state.log(
                    f"CriticAgent: score={critic_output.overall_score}, "
                    f"approved={critic_output.approved}"
                )

                # Exit loop if approved or we've reached max revisions
                if critic_output.approved:
                    state.log("CriticAgent approved the report — stopping revision loop")
                    break

                revisions_remaining = max_revisions - revision_round
                if revisions_remaining <= 0:
                    state.log(
                        f"Maximum revisions ({max_revisions}) reached — "
                        "finalising with current report"
                    )
                    break

                # Trigger Writer revision
                state.revision_count += 1
                next_revision = revision_round + 1
                state.set_agent_status(
                    "writer_agent", "running",
                    f"Revising report (revision {next_revision}/{max_revisions})..."
                )
                state.log(f"WriterAgent: starting revision {next_revision}")

                revised = self.writer_agent.revise_report(
                    research_topic=research_config.research_topic,
                    current_report=current_report.report_markdown,
                    critic_feedback=critic_output,
                    search_output=search_output,
                    reader_output=reader_output,
                    revision_number=next_revision,
                )
                state.revised_reports.append(revised)
                current_report = revised

                state.set_agent_status(
                    "writer_agent", "completed",
                    f"Revision {next_revision} complete: {revised.word_count} words"
                )
                state.log(f"WriterAgent: revision {next_revision} complete")

            # ================================================================
            # FINALISE
            # ================================================================
            final_report_output = current_report
            state.final_report = final_report_output.report_markdown
            state.final_score = state.critic_output.overall_score
            state.status = "completed"
            state.completed_at = datetime.now()
            state.log("Research pipeline completed successfully")

            logger.info(
                "Orchestrator complete: ID=%s, score=%d, revisions=%d",
                research_id, state.final_score, state.revision_count
            )

        except Exception as exc:
            error_msg = str(exc)
            tb = traceback.format_exc()
            logger.error("Orchestrator failed: %s\n%s", error_msg, tb)
            state.status = "failed"
            state.error_message = error_msg
            state.completed_at = datetime.now()
            state.log(f"ERROR: {error_msg}", level="ERROR")

            # Mark any pending agents as skipped
            for agent_name, status in state.agent_statuses.items():
                if status == "pending":
                    state.agent_statuses[agent_name] = "skipped"
                elif status == "running":
                    state.agent_statuses[agent_name] = "failed"

        finally:
            # Always persist — even failed sessions are saved for debugging
            try:
                save_research_session(state.to_dict())
                state.log("Session persisted to database")
            except Exception as db_exc:
                logger.error("Failed to persist session: %s", db_exc)
                state.log(f"WARNING: Database persistence failed: {db_exc}")

        return self._build_result(state)

    @staticmethod
    def _build_result(state: ResearchState) -> ResearchResult:
        """Convert the final ResearchState into a ResearchResult for the frontend."""
        return ResearchResult(
            research_id=state.research_id,
            research_topic=state.research_topic,
            configuration=state.configuration,
            search_output=state.search_output,
            reader_output=state.reader_output,
            writer_output=state.writer_output,
            critic_output=state.critic_output,
            final_report=state.final_report,
            final_score=state.final_score,
            revision_count=state.revision_count,
            status=state.status,
            error_message=state.error_message,
            started_at=state.started_at,
            completed_at=state.completed_at,
            execution_logs=state.execution_logs,
        )
