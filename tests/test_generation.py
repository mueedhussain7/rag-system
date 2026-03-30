from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient
from app.main import app
from app.generation.prompt import RAG_PROMPT

client = TestClient(app)

# ── Prompt template ───────────────────────────────────────────────────────────

def test_prompt_contains_context_and_question():
    """Prompt template should include both context and question placeholders."""
    prompt_str = str(RAG_PROMPT)
    assert "context"  in prompt_str
    assert "question" in prompt_str

def test_prompt_enforces_grounding():
    """Prompt must instruct the model to answer only from context."""
    system_msg = RAG_PROMPT.messages[0].prompt.template
    assert "ONLY" in system_msg
    assert "context" in system_msg.lower()

# ── /ask endpoint ─────────────────────────────────────────────────────────────

def test_ask_returns_answer_and_sources():
    """/ask should return question, answer, sources, and chunks_used."""
    mock_result = {
        "question":    "what is this about?",
        "answer":      "This paper is about mobile learning.",
        "sources":     ["doc.pdf (page 1)"],
        "chunks_used": 5,
    }
    with patch("app.main.ask", return_value=mock_result):
        response = client.post("/ask", json={"source": "what is this about?"})

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "This paper is about mobile learning."
    assert "sources" in data
    assert data["chunks_used"] == 5

def test_ask_returns_500_on_failure():
    """/ask should return 500 if generation fails."""
    with patch("app.main.ask", side_effect=Exception("LLM error")):
        response = client.post("/ask", json={"source": "test question"})
    assert response.status_code == 500

# ── /ask/stream endpoint ──────────────────────────────────────────────────────

def test_ask_stream_returns_text():
    """/ask/stream should return streaming plain text."""
    mock_chunks = [
        {"content": "Some content.", "metadata": {"source": "doc.pdf", "page": 1},
         "score": 0.9, "method": "semantic"},
    ]

    async def mock_astream(*args, **kwargs):
        for token in ["Mobile ", "learning ", "is ", "the ", "topic."]:
            yield token

    mock_chain = MagicMock()
    mock_chain.astream = mock_astream

    with patch("app.main.hybrid_search", return_value=mock_chunks), \
         patch("app.main.build_rag_chain", return_value=mock_chain):
        response = client.post("/ask/stream", json={"source": "what is this about?"})

    assert response.status_code == 200
    assert "mobile" in response.text.lower() or len(response.text) > 0

# ── Auto-refresh scheduler ────────────────────────────────────────────────────

def test_refresh_documents_skips_empty_folder(tmp_path):
    """refresh_documents should handle missing docs folder gracefully."""
    from app.generation.scheduler import refresh_documents
    with patch("app.generation.scheduler.Path") as mock_path:
        mock_path.return_value.exists.return_value = False
        # Should not raise, just log a warning and return
        refresh_documents()