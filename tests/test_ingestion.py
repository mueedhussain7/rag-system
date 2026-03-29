from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_ingest_file_not_found():
    """Ingesting a non-existent file should return 404."""
    response = client.post("/ingest", json={"source": "data/documents/nonexistent.pdf"})
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()

def test_ingest_unsupported_file_type():
    """Ingesting an unsupported file type should return 400."""
    with patch("app.ingestion.loaders.Path.exists", return_value=True):
        response = client.post("/ingest", json={"source": "data/documents/file.csv"})
    assert response.status_code == 400
    assert "unsupported" in response.json()["detail"].lower()

def test_ingest_success():
    """Successful ingestion returns status=success with chunk count."""
    with patch("app.main.load_document") as mock_load, \
         patch("app.main.chunk_documents") as mock_chunk, \
         patch("app.main.ingest_chunks") as mock_ingest:

        mock_load.return_value = [MagicMock()]
        mock_chunk.return_value = [MagicMock()] * 10
        mock_ingest.return_value = {
            "status": "success",
            "source": "data/documents/test.pdf",
            "chunks_ingested": 10,
            "doc_id": "abc123"
        }

        response = client.post("/ingest", json={"source": "data/documents/test.pdf"})
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["chunks_ingested"] == 10
        assert "doc_id" in data

def test_ingest_duplicate():
    """Ingesting the same document twice returns status=skipped."""
    with patch("app.main.load_document") as mock_load, \
         patch("app.main.chunk_documents") as mock_chunk, \
         patch("app.main.ingest_chunks") as mock_ingest:

        mock_load.return_value = [MagicMock()]
        mock_chunk.return_value = [MagicMock()] * 10
        mock_ingest.return_value = {
            "status": "skipped",
            "reason": "already ingested",
            "source": "data/documents/test.pdf"
        }

        response = client.post("/ingest", json={"source": "data/documents/test.pdf"})
        assert response.status_code == 200
        assert response.json()["status"] == "skipped"