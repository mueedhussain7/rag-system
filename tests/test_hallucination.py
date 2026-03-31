from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from app.main import app
from app.hallucination.scorer import get_confidence_label

client = TestClient(app)

# ── Confidence label logic ────────────────────────────────────────────────────

def test_confidence_high():
    """Score >= 0.8 + clean verdict = high confidence."""
    assert get_confidence_label(0.95, "clean") == "high"
    assert get_confidence_label(0.80, "clean") == "high"

def test_confidence_medium():
    """Score >= 0.5 + not contradicted = medium confidence."""
    assert get_confidence_label(0.65, "clean")     == "medium"
    assert get_confidence_label(0.50, "uncertain") == "medium"

def test_confidence_low():
    """Contradicted verdict or low score = low confidence."""
    assert get_confidence_label(0.3,  "clean")        == "low"
    assert get_confidence_label(0.9,  "contradicted") == "low"

def test_confidence_unverified():
    """Score of -1 means scoring failed = unverified."""
    assert get_confidence_label(-1.0, "clean") == "unverified"

# ── /score endpoint ───────────────────────────────────────────────────────────

def test_score_endpoint_returns_all_fields():
    """/score should return faithfulness, confidence, and nli fields."""
    mock_result = {
        "faithfulness_score": 0.9,
        "confidence_level":   "high",
        "nli_verdict":        "clean",
        "nli_details":        {"sentences": [], "counts": {}, "verdict": "clean"},
    }
    with patch("app.main.score_answer", return_value=mock_result):
        response = client.post("/score", json={
            "question": "What is RAG?",
            "answer":   "RAG stands for Retrieval-Augmented Generation.",
            "context":  "RAG stands for Retrieval-Augmented Generation.",
        })

    assert response.status_code == 200
    data = response.json()
    assert "faithfulness_score" in data
    assert "confidence_level"   in data
    assert "nli_verdict"        in data

def test_score_endpoint_500_on_failure():
    """/score should return 500 if scoring fails."""
    with patch("app.main.score_answer", side_effect=Exception("fail")):
        response = client.post("/score", json={
            "question": "test",
            "answer":   "test",
            "context":  "test",
        })
    assert response.status_code == 500

# ── /ask now includes hallucination fields ────────────────────────────────────

def test_ask_includes_hallucination_fields():
    """/ask response must include faithfulness_score and confidence_level."""
    mock_chunks = [{
        "content":  "Mobile devices help students learn.",
        "metadata": {"source": "doc.pdf", "page": 1},
        "score":    0.9,
        "method":   "semantic",
    }]
    mock_hal = {
        "faithfulness_score": 1.0,
        "confidence_level":   "high",
        "nli_verdict":        "clean",
        "nli_details":        {},
    }

    with patch("app.main.hybrid_search",  return_value=mock_chunks), \
         patch("app.main.build_rag_chain") as mock_chain_fn, \
         patch("app.main.score_answer",   return_value=mock_hal):

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "Students used mobile devices to learn."
        mock_chain_fn.return_value = mock_chain

        response = client.post("/ask", json={"source": "what did students say?"})

    assert response.status_code == 200
    data = response.json()
    assert "faithfulness_score" in data
    assert "confidence_level"   in data
    assert "nli_verdict"        in data
    assert data["confidence_level"] == "high"