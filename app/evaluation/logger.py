# app/evaluation/logger.py
import sqlite3
import logging
import json
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

DB_PATH = Path("data/query_log.db")


def get_connection() -> sqlite3.Connection:
    """Returns a SQLite connection, creates DB file if it doesn't exist."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row  # rows behave like dicts
    return conn


def init_db():
    """
    Creates the tables if they don't exist yet.
    Called once on server startup.
    """
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS query_log (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp           TEXT    NOT NULL,
                question            TEXT    NOT NULL,
                answer              TEXT    NOT NULL,
                sources             TEXT,       -- JSON list
                chunks_used         INTEGER,
                faithfulness_score  REAL,
                confidence_level    TEXT,
                nli_verdict         TEXT,
                latency_ms          REAL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ingestion_log (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp     TEXT    NOT NULL,
                source        TEXT    NOT NULL,
                chunks        INTEGER,
                status        TEXT,
                doc_id        TEXT
            )
        """)
        conn.commit()
    logger.info("Database initialised")


def log_query(
    question:           str,
    answer:             str,
    sources:            list[str],
    chunks_used:        int,
    faithfulness_score: float,
    confidence_level:   str,
    nli_verdict:        str,
    latency_ms:         float,
):
    """Saves a completed query + its scores to the database."""
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO query_log
            (timestamp, question, answer, sources, chunks_used,
             faithfulness_score, confidence_level, nli_verdict, latency_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now(timezone.utc).isoformat(),
            question,
            answer,
            json.dumps(sources),
            chunks_used,
            faithfulness_score,
            confidence_level,
            nli_verdict,
            latency_ms,
        ))
        conn.commit()


def log_ingestion(source: str, chunks: int, status: str, doc_id: str):
    """Saves a completed ingestion event to the database."""
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO ingestion_log (timestamp, source, chunks, status, doc_id)
            VALUES (?, ?, ?, ?, ?)
        """, (
            datetime.now(timezone.utc).isoformat(),
            source, chunks, status, doc_id,
        ))
        conn.commit()


def get_all_queries() -> list[dict]:
    """Returns all logged queries, newest first."""
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT * FROM query_log ORDER BY timestamp DESC
        """).fetchall()
    return [dict(row) for row in rows]


def get_all_ingestions() -> list[dict]:
    """Returns all ingestion events, newest first."""
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT * FROM ingestion_log ORDER BY timestamp DESC
        """).fetchall()
    return [dict(row) for row in rows]


def get_summary_stats() -> dict:
    """Returns aggregate metrics for the Overview page."""
    with get_connection() as conn:
        total = conn.execute(
            "SELECT COUNT(*) FROM query_log"
        ).fetchone()[0]

        avg_faith = conn.execute(
            "SELECT AVG(faithfulness_score) FROM query_log WHERE faithfulness_score >= 0"
        ).fetchone()[0]

        hal_rate = conn.execute(
            "SELECT COUNT(*) FROM query_log WHERE confidence_level = 'low'"
        ).fetchone()[0]

        avg_latency = conn.execute(
            "SELECT AVG(latency_ms) FROM query_log"
        ).fetchone()[0]

        confidence_counts = conn.execute("""
            SELECT confidence_level, COUNT(*) as count
            FROM query_log GROUP BY confidence_level
        """).fetchall()

    return {
        "total_queries":      total,
        "avg_faithfulness":   round(avg_faith, 3) if avg_faith else 0.0,
        "hallucination_count": hal_rate,
        "hallucination_rate": round(hal_rate / total * 100, 1) if total else 0.0,
        "avg_latency_ms":     round(avg_latency, 1) if avg_latency else 0.0,
        "confidence_counts":  {row[0]: row[1] for row in confidence_counts},
    }