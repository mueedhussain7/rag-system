import os
os.environ.setdefault("USER_AGENT", "rag-system/0.1.0")

import time
import logging
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict
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


def validate_file_path(source: str) -> str:
    """Validate file path to prevent directory traversal attacks."""
    from pathlib import Path

    # Check if it's a URL
    if source.startswith(("http://", "https://")):
        return source

    # For file paths, validate against directory traversal
    try:
        path = Path(source).resolve()
        allowed_dir = Path("data/documents").resolve()

        # Ensure path is within allowed directory
        if not str(path).startswith(str(allowed_dir)):
            raise ValueError(f"Access denied: path must be within data/documents directory")

        return source
    except (ValueError, RuntimeError) as e:
        raise ValueError(f"Invalid file path: {str(e)}")


def redact_sensitive_data(text: str) -> str:
    """Redact PII and sensitive patterns from text before logging."""
    import re
    if not text:
        return text

    # Redact credit card numbers (16 digits)
    text = re.sub(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b', '[REDACTED_CC]', text)
    # Redact SSN (XXX-XX-XXXX)
    text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[REDACTED_SSN]', text)
    # Redact email addresses
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[REDACTED_EMAIL]', text)
    # Redact phone numbers (XXX-XXX-XXXX or (XXX) XXX-XXXX)
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[REDACTED_PHONE]', text)
    # Redact API keys (sk-, api_, etc.)
    text = re.sub(r'(sk-|api-|key[_-])[A-Za-z0-9_\-]{20,}', '[REDACTED_API_KEY]', text)

    return text


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
    docs_url="/docs" if settings.app_env == "development" else None,
    redoc_url="/redoc" if settings.app_env == "development" else None,
    openapi_url="/openapi.json" if settings.app_env == "development" else None,
    max_request_size=1048576,  # 1MB max request body size
)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

# CORS middleware (restrict cross-origin requests)
allowed_origins = (
    ["http://localhost:3000", "http://localhost:8501"] 
    if settings.app_env == "development"
    else ["https://yourdomain.com"]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Trusted host middleware (protects against Host header attacks)
if settings.app_env == "production":
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["yourdomain.com"],  # Change to your domain in production
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
    response = await call_next(request)

    # Add security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

    if settings.app_env == "production":
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'"

    return response


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
    model_config = ConfigDict(str_max_length=2000)
    source: str

class AskRequest(BaseModel):
    model_config = ConfigDict(str_max_length=5000)
    question: str

    def __init__(self, **data):
        super().__init__(**data)
        if not self.question or len(self.question.strip()) == 0:
            raise ValueError("question cannot be empty")
        if len(self.question) > 5000:
            raise ValueError("question exceeds maximum length of 5000 characters")

class ScoreRequest(BaseModel):
    model_config = ConfigDict(str_max_length=10000)
    question: str
    answer:   str
    context:  str


# ── Ingestion ─────────────────────────────────────────────────────────────────

@app.post("/ingest")
@limiter.limit("10/minute")
async def ingest(request: Request, ingest_req: IngestRequest, api_key: str = Depends(verify_api_key)):
    try:
        # Validate file path to prevent directory traversal
        validate_file_path(ingest_req.source)

        documents = load_document(ingest_req.source)
        chunks    = chunk_documents(documents)
        result    = ingest_chunks(chunks, ingest_req.source)
        log_ingestion(
            source=ingest_req.source,
            chunks=result.get("chunks_ingested", 0),
            status=result.get("status", "unknown"),
            doc_id=result.get("doc_id", ""),
        )
        source_type = "url" if ingest_req.source.startswith(("http://", "https://")) else "file"
        log_source_refresh(ingest_req.source, source_type)
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
async def retrieve(request: Request, q: str, top_k: int = 5, api_key: str = Depends(verify_api_key)):
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
async def ask_question(request: Request, ask_req: AskRequest, api_key: str = Depends(verify_api_key)):
    try:
        start   = time.time()
        chunks  = hybrid_search(ask_req.question, top_k=5)
        context = assemble_context(chunks)
        context = _validate_and_truncate_context(context)
        chain   = build_rag_chain(streaming=False)

        try:
            answer = await asyncio.wait_for(
                asyncio.to_thread(chain.invoke, {"context": context, "question": ask_req.question}),
                timeout=settings.llm_timeout
            )
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=504,
                detail=f"LLM request timeout after {settings.llm_timeout} seconds"
            )

        hal_score  = score_answer(ask_req.question, answer, chunks)
        sources    = list({
            f"{c['metadata'].get('source', 'unknown')} (page {c['metadata'].get('page', '?')})"
            for c in chunks
        })
        latency_ms = round((time.time() - start) * 1000, 1)

        log_query(
            question=redact_sensitive_data(ask_req.question),
            answer=redact_sensitive_data(answer),
            sources=sources,
            chunks_used=len(chunks),
            faithfulness_score=hal_score["faithfulness_score"],
            confidence_level=hal_score["confidence_level"],
            nli_verdict=hal_score["nli_verdict"],
            latency_ms=latency_ms,
        )

        return {
            "question":           ask_req.question,
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
async def ask_stream(request: Request, ask_req: AskRequest, api_key: str = Depends(verify_api_key)):
    async def token_generator():
        try:
            start   = time.time()
            chunks  = hybrid_search(ask_req.question, top_k=5)
            context = assemble_context(chunks)
            context = _validate_and_truncate_context(context)
            chain   = build_rag_chain(streaming=True)
            answer_tokens = []

            try:
                stream = chain.astream({"context": context, "question": ask_req.question})
                async for token in stream:
                    answer_tokens.append(token)
                    yield token
            except asyncio.TimeoutError:
                yield f"\n[Error: LLM request timeout after {settings.llm_timeout} seconds]"
                return

            answer = "".join(answer_tokens)
            hal_score = score_answer(ask_req.question, answer, chunks)
            sources = list({
                f"{c['metadata'].get('source', 'unknown')} (page {c['metadata'].get('page', '?')})"
                for c in chunks
            })
            latency_ms = round((time.time() - start) * 1000, 1)

            log_query(
                question=redact_sensitive_data(ask_req.question),
                answer=redact_sensitive_data(answer),
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
async def score(request: Request, score_req: ScoreRequest, api_key: str = Depends(verify_api_key)):
    try:
        chunks = [{"content": score_req.context, "metadata": {}}]
        result = score_answer(score_req.question, score_req.answer, chunks)
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