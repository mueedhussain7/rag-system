import os
os.environ.setdefault("USER_AGENT", "rag-system/0.1.0")

import time
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from app.config import settings
from app.ingestion.loaders import load_document
from app.ingestion.chunker import chunk_documents
from app.ingestion.embedder import ingest_chunks
from app.retrieval.hybrid import hybrid_search
from app.retrieval.context import assemble_context
from app.generation.chain import ask, build_rag_chain, _validate_and_truncate_context
from app.generation.scheduler import start_scheduler, refresh_documents
from app.hallucination.scorer import score_answer
from app.evaluation.logger import init_db, log_query, log_ingestion, get_summary_stats, log_source_refresh



logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)


# ── API Key Authentication ────────────────────────────────────

async def verify_api_key(x_api_key: str = Header(None)) -> str:
    """Verify client provides valid API key in X-API-Key header"""
    if not x_api_key:
        raise HTTPException(
            status_code=401,
            detail="Missing X-API-Key header"
        )
    if x_api_key != settings.api_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    return x_api_key


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

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

# Trusted host middleware (protects against Host header attacks)
if settings.app_env == "production":
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"],  # Change to your domain in production
    )

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return StreamingResponse(
        iter([f"Rate limit exceeded: {exc.detail}".encode()]),
        status_code=429,
        media_type="text/plain"
    )

@app.middleware("http")
async def enforce_https(request: Request, call_next):
    """In production, enforce HTTPS by rejecting HTTP requests."""
    if settings.app_env == "production" and request.url.scheme == "http":
        return StreamingResponse(
            iter(["HTTPS required in production".encode()]),
            status_code=400,
            media_type="text/plain"
        )
    return await call_next(request)


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

    class Config:
        max_anystr_length = 2000

class AskRequest(BaseModel):
    question: str

    class Config:
        max_anystr_length = 5000

    def __init__(self, **data):
        super().__init__(**data)
        if not self.question or len(self.question.strip()) == 0:
            raise ValueError("question cannot be empty")
        if len(self.question) > 5000:
            raise ValueError("question exceeds maximum length of 5000 characters")

class ScoreRequest(BaseModel):
    question: str
    answer:   str
    context:  str

    class Config:
        max_anystr_length = 10000


# ── Ingestion ─────────────────────────────────────────────────────────────────

@app.post("/ingest")
@limiter.limit("10/minute")
async def ingest(request: IngestRequest, api_key: str = Depends(verify_api_key), _=None):
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
        source_type = "url" if request.source.startswith(("http://", "https://")) else "file"
        log_source_refresh(request.source, source_type)
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
@limiter.limit("10/minute")
async def retrieve(q: str, top_k: int = 5, api_key: str = Depends(verify_api_key), _=None):
    if not q or len(q.strip()) == 0:
        raise HTTPException(status_code=400, detail="query cannot be empty")
    if len(q) > 5000:
        raise HTTPException(status_code=400, detail="query exceeds maximum length of 5000 characters")
    if top_k < 1 or top_k > 20:
        raise HTTPException(status_code=400, detail="top_k must be between 1 and 20")

    try:
        chunks  = hybrid_search(q, top_k=top_k)
        context = assemble_context(chunks)
        return {"query": q, "chunks": chunks, "context": context, "total": len(chunks)}
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        raise HTTPException(status_code=500, detail="Retrieval failed — check server logs")


# ── Generation ────────────────────────────────────────────────────────────────

@app.post("/ask")
@limiter.limit("10/minute")
async def ask_question(request: AskRequest, api_key: str = Depends(verify_api_key), _=None):
    try:
        start   = time.time()
        chunks  = hybrid_search(request.question, top_k=5)
        context = assemble_context(chunks)
        context = _validate_and_truncate_context(context)
        chain   = build_rag_chain(streaming=False)
        answer  = chain.invoke({"context": context, "question": request.question})

        hal_score  = score_answer(request.question, answer, chunks)
        sources    = list({
            f"{c['metadata'].get('source', 'unknown')} (page {c['metadata'].get('page', '?')})"
            for c in chunks
        })
        latency_ms = round((time.time() - start) * 1000, 1)

        log_query(
            question=request.question,
            answer=answer,
            sources=sources,
            chunks_used=len(chunks),
            faithfulness_score=hal_score["faithfulness_score"],
            confidence_level=hal_score["confidence_level"],
            nli_verdict=hal_score["nli_verdict"],
            latency_ms=latency_ms,
        )

        return {
            "question":           request.question,
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
@limiter.limit("10/minute")
async def ask_stream(request: AskRequest, api_key: str = Depends(verify_api_key), _=None):
    async def token_generator():
        try:
            start   = time.time()
            chunks  = hybrid_search(request.question, top_k=5)
            context = assemble_context(chunks)
            context = _validate_and_truncate_context(context)
            chain   = build_rag_chain(streaming=True)
            answer_tokens = []

            async for token in chain.astream({"context": context, "question": request.question}):
                answer_tokens.append(token)
                yield token

            answer = "".join(answer_tokens)
            hal_score = score_answer(request.question, answer, chunks)
            sources = list({
                f"{c['metadata'].get('source', 'unknown')} (page {c['metadata'].get('page', '?')})"
                for c in chunks
            })
            latency_ms = round((time.time() - start) * 1000, 1)

            log_query(
                question=request.question,
                answer=answer,
                sources=sources,
                chunks_used=len(chunks),
                faithfulness_score=hal_score["faithfulness_score"],
                confidence_level=hal_score["confidence_level"],
                nli_verdict=hal_score["nli_verdict"],
                latency_ms=latency_ms,
            )
        except Exception as e:
            logger.error(f"Streaming failed: {e}", exc_info=True)
            yield f"\n[Error: {type(e).__name__}: {str(e)}]"
    return StreamingResponse(token_generator(), media_type="text/plain")


# ── Hallucination scoring ─────────────────────────────────────────────────────

@app.post("/score")
@limiter.limit("10/minute")
async def score(request: ScoreRequest, api_key: str = Depends(verify_api_key), _=None):
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