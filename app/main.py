import os
os.environ.setdefault("USER_AGENT", "rag-system/0.1.0")

import time
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from app.config import settings
from app.ingestion.loaders import load_document
from app.ingestion.chunker import chunk_documents
from app.ingestion.embedder import ingest_chunks
from app.retrieval.hybrid import hybrid_search
from app.retrieval.context import assemble_context
from app.generation.chain import ask, build_rag_chain
from app.generation.scheduler import start_scheduler, refresh_documents
from app.hallucination.scorer import score_answer
from app.evaluation.logger import init_db, log_query, log_ingestion, get_summary_stats



logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"RAG System starting | env={settings.app_env} | v={settings.app_version}")
    init_db()
    scheduler = start_scheduler()
    refresh_documents()
    yield
    scheduler.shutdown()
    logger.info("RAG System shutting down")


app = FastAPI(
    title="RAG System",
    description="Auto-Updating, Hallucination-Aware RAG with Evaluation Dashboard",
    version=settings.app_version,
    lifespan=lifespan,
)


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    return {
        "status":      "ok",
        "version":     settings.app_version,
        "environment": settings.app_env,
    }


# ── Models ────────────────────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    source: str

class ScoreRequest(BaseModel):
    question: str
    answer:   str
    context:  str


# ── Ingestion ─────────────────────────────────────────────────────────────────

@app.post("/ingest")
async def ingest(request: IngestRequest):
    try:
        documents = load_document(request.source)
        chunks    = chunk_documents(documents)
        result    = ingest_chunks(chunks, request.source)
        log_ingestion(
            source=request.source,
            chunks=result.get("chunks_ingested", 0),
            status=result.get("status", "unknown"),
            doc_id=result.get("doc_id", ""),
        )
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail="Ingestion failed — check server logs")


# ── Retrieval ─────────────────────────────────────────────────────────────────

@app.get("/retrieve")
async def retrieve(q: str, top_k: int = 5):
    try:
        chunks  = hybrid_search(q, top_k=top_k)
        context = assemble_context(chunks)
        return {"query": q, "chunks": chunks, "context": context, "total": len(chunks)}
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        raise HTTPException(status_code=500, detail="Retrieval failed — check server logs")


# ── Generation ────────────────────────────────────────────────────────────────

@app.post("/ask")
async def ask_question(request: IngestRequest):
    try:
        start   = time.time()
        chunks  = hybrid_search(request.source, top_k=5)
        context = assemble_context(chunks)
        chain   = build_rag_chain(streaming=False)
        answer  = chain.invoke({"context": context, "question": request.source})

        hal_score  = score_answer(request.source, answer, chunks)
        sources    = list({
            f"{c['metadata'].get('source', 'unknown')} (page {c['metadata'].get('page', '?')})"
            for c in chunks
        })
        latency_ms = round((time.time() - start) * 1000, 1)

        log_query(
            question=request.source,
            answer=answer,
            sources=sources,
            chunks_used=len(chunks),
            faithfulness_score=hal_score["faithfulness_score"],
            confidence_level=hal_score["confidence_level"],
            nli_verdict=hal_score["nli_verdict"],
            latency_ms=latency_ms,
        )

        return {
            "question":           request.source,
            "answer":             answer,
            "sources":            sources,
            "chunks_used":        len(chunks),
            "faithfulness_score": hal_score["faithfulness_score"],
            "confidence_level":   hal_score["confidence_level"],
            "nli_verdict":        hal_score["nli_verdict"],
            "latency_ms":         latency_ms,
        }
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail="Generation failed — check server logs")


@app.post("/ask/stream")
async def ask_stream(request: IngestRequest):
    async def token_generator():
        try:
            chunks  = hybrid_search(request.source, top_k=5)
            context = assemble_context(chunks)
            chain   = build_rag_chain(streaming=True)
            async for token in chain.astream({"context": context, "question": request.source}):
                yield token
        except Exception as e:
            logger.error(f"Streaming failed: {e}")
            yield f"\n[Error: {e}]"
    return StreamingResponse(token_generator(), media_type="text/plain")


# ── Hallucination scoring ─────────────────────────────────────────────────────

@app.post("/score")
async def score(request: ScoreRequest):
    try:
        chunks = [{"content": request.context, "metadata": {}}]
        result = score_answer(request.question, request.answer, chunks)
        return result
    except Exception as e:
        logger.error(f"Scoring failed: {e}")
        raise HTTPException(status_code=500, detail="Scoring failed — check server logs")
    

@app.get("/metrics")
async def metrics():
    """Returns current system health metrics as JSON."""
    try:
        return get_summary_stats()
    except Exception as e:
        logger.error(f"Metrics failed: {e}")
        raise HTTPException(status_code=500, detail="Metrics failed")