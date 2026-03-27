# app/main.py
import os
os.environ.setdefault("USER_AGENT", "rag-system/0.1.0")
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.config import settings
from app.ingestion.loaders import load_document
from app.ingestion.chunker import chunk_documents
from app.ingestion.embedder import ingest_chunks

os.environ.setdefault("USER_AGENT", "rag-system/0.1.0")

logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"RAG System starting | env={settings.app_env} | v={settings.app_version}")
    yield
    logger.info("RAG System shutting down")

app = FastAPI(
    title="RAG System",
    description="Auto-Updating, Hallucination-Aware RAG with Evaluation Dashboard",
    version=settings.app_version,
    lifespan=lifespan,
)

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "version": settings.app_version,
        "environment": settings.app_env,
    }

# ── Ingestion ─────────────────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    source: str  # file path or URL

@app.post("/ingest")
async def ingest(request: IngestRequest):
    """
    Load a document, chunk it, embed it, and store it in ChromaDB.
    Accepts a file path (PDF or .txt) or a web URL.
    """
    try:
        documents = load_document(request.source)
        chunks    = chunk_documents(documents)
        result    = ingest_chunks(chunks, request.source)
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail="Ingestion failed — check server logs")