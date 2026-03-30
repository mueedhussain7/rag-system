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
from app.retrieval.hybrid import hybrid_search
from app.retrieval.context import assemble_context
from fastapi.responses import StreamingResponse
from app.generation.chain import ask, build_rag_chain
from app.generation.scheduler import start_scheduler, refresh_documents

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
    

@app.get("/retrieve")
async def retrieve(q: str, top_k: int = 5):
    """
    Given a question, return the most relevant chunks from ChromaDB.
    Uses hybrid search (semantic + BM25) with RRF fusion.
    """
    try:
        chunks  = hybrid_search(q, top_k=top_k)
        context = assemble_context(chunks)
        return {
            "query":   q,
            "chunks":  chunks,
            "context": context,
            "total":   len(chunks),
        }
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        raise HTTPException(status_code=500, detail="Retrieval failed — check server logs")
    

@app.post("/ask")
async def ask_question(request: IngestRequest):
    """
    Non-streaming version — waits for the full answer then returns it.
    Good for testing and API consumers that don't support streaming.
    """
    try:
        result = ask(request.source)  # reusing IngestRequest for simplicity (has 'source' field)
        return result
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail="Generation failed — check server logs")


@app.post("/ask/stream")
async def ask_stream(request: IngestRequest):
    """
    Streaming version — words appear as GPT-4 generates them.
    The client sees output immediately, making the app feel fast.
    """
    async def token_generator():
        try:
            chunks  = hybrid_search(request.source, top_k=5)
            context = assemble_context(chunks)
            chain   = build_rag_chain(streaming=True)

            # .astream() is the async streaming version of .invoke()
            # It yields one token at a time as GPT-4 generates them
            async for token in chain.astream({
                "context":  context,
                "question": request.source,
            }):
                yield token
        except Exception as e:
            logger.error(f"Streaming failed: {e}")
            yield f"\n[Error: {e}]"

    return StreamingResponse(token_generator(), media_type="text/plain")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"RAG System starting | env={settings.app_env} | v={settings.app_version}")
    # Start the background scheduler on startup
    scheduler = start_scheduler()
    # Run one immediate refresh to pick up any new documents
    refresh_documents()
    yield
    # Cleanly shut down the scheduler when the server stops
    scheduler.shutdown()
    logger.info("RAG System shutting down")