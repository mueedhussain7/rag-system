from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from app.main import app
from app.retrieval.context import assemble_context
from app.retrieval.hybrid import reciprocal_rank_fusion

client = TestClient(app)

# ── Context assembler ─────────────────────────────────────────────────────────

def test_assemble_context_format():
    """Assembled context should be numbered and include source info."""
    chunks = [
        {"content": "First chunk text.", "metadata": {"source": "doc.pdf", "page": 1}},
        {"content": "Second chunk text.", "metadata": {"source": "doc.pdf", "page": 2}},
    ]
    context = assemble_context(chunks)
    assert "[1]" in context
    assert "[2]" in context
    assert "doc.pdf" in context
    assert "First chunk text." in context

def test_assemble_context_empty():
    """Empty chunk list should return empty string."""
    assert assemble_context([]) == ""

# ── Reciprocal Rank Fusion ────────────────────────────────────────────────────

def test_rrf_combines_results():
    """RRF should merge semantic and keyword results without duplicates."""
    semantic = [
        {"content": "shared chunk", "metadata": {}, "score": 0.9, "method": "semantic"},
        {"content": "only semantic", "metadata": {}, "score": 0.8, "method": "semantic"},
    ]
    keyword = [
        {"content": "shared chunk", "metadata": {}, "score": 5.0, "method": "keyword"},
        {"content": "only keyword",  "metadata": {}, "score": 4.0, "method": "keyword"},
    ]
    fused = reciprocal_rank_fusion(semantic, keyword)
    contents = [r["content"] for r in fused]

    # shared chunk should rank highest (appeared in both lists)
    assert contents[0] == "shared chunk"
    # all three unique chunks should be present
    assert len(fused) == 3

# ── /retrieve endpoint ────────────────────────────────────────────────────────

def test_retrieve_endpoint_returns_chunks():
    """/retrieve should return chunks, context, and total."""
    mock_chunks = [
        {"content": "Relevant text.", "metadata": {"source": "doc.pdf", "page": 1},
         "score": 0.9, "method": "semantic"},
    ]
    with patch("app.main.hybrid_search", return_value=mock_chunks):
        response = client.get("/retrieve", params={"q": "test question"})

    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "test question"
    assert len(data["chunks"]) == 1
    assert "context" in data
    assert data["total"] == 1

def test_retrieve_endpoint_empty_query():
    """/retrieve with empty results should still return a valid response."""
    with patch("app.main.hybrid_search", return_value=[]):
        response = client.get("/retrieve", params={"q": "unknown topic"})

    assert response.status_code == 200
    assert response.json()["total"] == 0