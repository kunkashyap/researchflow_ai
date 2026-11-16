"""
memory/database.py
SQLite persistence layer for ResearchFlow AI.

Stores completed and in-progress research sessions so that users can
review previous results through the History tab. The schema uses a single
research_sessions table with JSON columns for the complex nested objects.
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from utils.config import config
from utils.logging_config import setup_logging

logger = setup_logging("researchflow.database")

DB_PATH: Path = config.DATABASE_DIR / "research_history.db"


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS research_sessions (
    research_id     TEXT PRIMARY KEY,
    research_topic  TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'pending',
    final_score     INTEGER DEFAULT 0,
    revision_count  INTEGER DEFAULT 0,
    configuration   TEXT,   -- JSON
    search_output   TEXT,   -- JSON
    reader_output   TEXT,   -- JSON
    writer_output   TEXT,   -- JSON
    critic_output   TEXT,   -- JSON
    final_report    TEXT,
    execution_logs  TEXT,   -- JSON array
    error_message   TEXT,
    started_at      TEXT,
    completed_at    TEXT
);
"""


def _get_connection() -> sqlite3.Connection:
    """Return a SQLite connection with row_factory set for dict-like access."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def initialise_db() -> None:
    """Create the database file and schema if they don't already exist."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _get_connection() as conn:
        conn.execute(CREATE_TABLE_SQL)
        conn.commit()
    logger.info("Database initialised at %s", DB_PATH)


# ---------------------------------------------------------------------------
# Write Operations
# ---------------------------------------------------------------------------

def save_research_session(state_dict: dict) -> None:
    """
    Insert or replace a research session record.

    Args:
        state_dict: The serialised ResearchState dictionary (from state.to_dict()).
    """
    try:
        with _get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO research_sessions (
                    research_id, research_topic, status, final_score,
                    revision_count, configuration, search_output, reader_output,
                    writer_output, critic_output, final_report,
                    execution_logs, error_message, started_at, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    state_dict.get("research_id"),
                    state_dict.get("research_topic"),
                    state_dict.get("status", "pending"),
                    state_dict.get("final_score", 0),
                    state_dict.get("revision_count", 0),
                    json.dumps(state_dict.get("configuration")),
                    json.dumps(state_dict.get("search_output")),
                    json.dumps(state_dict.get("reader_output")),
                    json.dumps(state_dict.get("writer_output")),
                    json.dumps(state_dict.get("critic_output")),
                    state_dict.get("final_report", ""),
                    json.dumps(state_dict.get("execution_logs", [])),
                    state_dict.get("error_message"),
                    state_dict.get("started_at"),
                    state_dict.get("completed_at"),
                ),
            )
            conn.commit()
        logger.info("Saved research session: %s", state_dict.get("research_id"))
    except sqlite3.Error as exc:
        logger.error("Failed to save research session: %s", exc)
        raise


# ---------------------------------------------------------------------------
# Read Operations
# ---------------------------------------------------------------------------

def get_all_sessions(limit: int = 50) -> list[dict]:
    """
    Retrieve a summary list of all research sessions, newest first.

    Returns lightweight dicts suitable for the History tab list view.
    """
    try:
        with _get_connection() as conn:
            rows = conn.execute(
                """
                SELECT research_id, research_topic, status, final_score,
                       revision_count, started_at, completed_at
                FROM research_sessions
                ORDER BY started_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]
    except sqlite3.Error as exc:
        logger.error("Failed to fetch sessions: %s", exc)
        return []


def get_session_by_id(research_id: str) -> Optional[dict]:
    """
    Retrieve the full record for a single research session.

    JSON columns are deserialised back to Python objects.
    """
    try:
        with _get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM research_sessions WHERE research_id = ?",
                (research_id,),
            ).fetchone()
        if not row:
            return None
        record = dict(row)
        # Deserialise JSON columns
        for col in ("configuration", "search_output", "reader_output",
                     "writer_output", "critic_output", "execution_logs"):
            if record.get(col):
                try:
                    record[col] = json.loads(record[col])
                except json.JSONDecodeError:
                    record[col] = None
        return record
    except sqlite3.Error as exc:
        logger.error("Failed to fetch session %s: %s", research_id, exc)
        return None


def delete_session(research_id: str) -> bool:
    """Delete a research session by ID. Returns True if a row was deleted."""
    try:
        with _get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM research_sessions WHERE research_id = ?",
                (research_id,),
            )
            conn.commit()
        return cursor.rowcount > 0
    except sqlite3.Error as exc:
        logger.error("Failed to delete session %s: %s", research_id, exc)
        return False


# Initialise on import — creates the DB file and schema if missing
initialise_db()
