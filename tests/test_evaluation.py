import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

# ── Logger tests ──────────────────────────────────────────────────────────────

def test_init_db_creates_tables():
    """init_db should create query_log and ingestion_log tables."""
    with tempfile.TemporaryDirectory() as tmp:
        with patch("app.evaluation.logger.DB_PATH", Path(tmp) / "test.db"):
            from app.evaluation.logger import init_db, get_connection
            init_db()
            with get_connection() as conn:
                tables = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
                table_names = [t[0] for t in tables]
            assert "query_log"     in table_names
            assert "ingestion_log" in table_names


def test_log_and_retrieve_query():
    """log_query should save a record retrievable by get_all_queries."""
    with tempfile.TemporaryDirectory() as tmp:
        with patch("app.evaluation.logger.DB_PATH", Path(tmp) / "test.db"):
            from app.evaluation.logger import (
                init_db, log_query, get_all_queries
            )
            init_db()
            log_query(
                question="test question",
                answer="test answer",
                sources=["doc.pdf (page 1)"],
                chunks_used=5,
                faithfulness_score=0.9,
                confidence_level="high",
                nli_verdict="clean",
                latency_ms=1234.5,
            )
            rows = get_all_queries()
            assert len(rows) == 1
            assert rows[0]["question"] == "test question"
            assert rows[0]["faithfulness_score"] == 0.9
            assert rows[0]["confidence_level"] == "high"


def test_log_and_retrieve_ingestion():
    """log_ingestion should save a record retrievable by get_all_ingestions."""
    with tempfile.TemporaryDirectory() as tmp:
        with patch("app.evaluation.logger.DB_PATH", Path(tmp) / "test.db"):
            from app.evaluation.logger import (
                init_db, log_ingestion, get_all_ingestions
            )
            init_db()
            log_ingestion(
                source="data/documents/test.pdf",
                chunks=42,
                status="success",
                doc_id="abc123",
            )
            rows = get_all_ingestions()
            assert len(rows) == 1
            assert rows[0]["chunks"] == 42
            assert rows[0]["status"] == "success"


def test_get_summary_stats_empty():
    """Summary stats on empty DB should return zeros without crashing."""
    with tempfile.TemporaryDirectory() as tmp:
        with patch("app.evaluation.logger.DB_PATH", Path(tmp) / "test.db"):
            from app.evaluation.logger import init_db, get_summary_stats
            init_db()
            stats = get_summary_stats()
            assert stats["total_queries"]    == 0
            assert stats["avg_faithfulness"] == 0.0
            assert stats["hallucination_rate"] == 0.0


def test_get_summary_stats_with_data():
    """Summary stats should correctly aggregate logged queries."""
    with tempfile.TemporaryDirectory() as tmp:
        with patch("app.evaluation.logger.DB_PATH", Path(tmp) / "test.db"):
            from app.evaluation.logger import (
                init_db, log_query, get_summary_stats
            )
            init_db()
            for i, (score, conf) in enumerate([
                (1.0, "high"),
                (0.6, "medium"),
                (0.2, "low"),
            ]):
                log_query(
                    question=f"question {i}",
                    answer=f"answer {i}",
                    sources=[],
                    chunks_used=5,
                    faithfulness_score=score,
                    confidence_level=conf,
                    nli_verdict="clean",
                    latency_ms=1000.0,
                )
            stats = get_summary_stats()
            assert stats["total_queries"]      == 3
            assert stats["hallucination_count"] == 1  # one "low"
            assert stats["hallucination_rate"]  == pytest.approx(33.3, rel=0.1)


# ── /metrics endpoint ─────────────────────────────────────────────────────────

def test_metrics_endpoint():
    """/metrics should return system health as JSON."""
    mock_stats = {
        "total_queries":       10,
        "avg_faithfulness":    0.85,
        "hallucination_count": 1,
        "hallucination_rate":  10.0,
        "avg_latency_ms":      4500.0,
        "confidence_counts":   {"high": 8, "medium": 1, "low": 1},
    }
    with patch("app.evaluation.logger.get_connection") as mock_conn:
        # Mock the connection to avoid hitting the real DB
        with patch("app.main.get_summary_stats", return_value=mock_stats, create=True):
            pass  # just verify the endpoint exists

    # Test with real DB (it exists and is initialised)
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "total_queries"      in data
    assert "avg_faithfulness"   in data
    assert "hallucination_rate" in data
    assert "avg_latency_ms"     in data