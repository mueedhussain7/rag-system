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
from app.hallucination.scorer import score_answer

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
    Retrieve → generate → score for hallucination.
    Every answer now includes a faithfulness score and confidence level.
    """
    try:
        # Step 1 — retrieve chunks
        chunks  = hybrid_search(request.source, top_k=5)
        context = assemble_context(chunks)

        # Step 2 — generate answer
        chain  = build_rag_chain(streaming=False)
        answer = chain.invoke({
            "context":  context,
            "question": request.source,
        })

        # Step 3 — score for hallucination
        hal_score = score_answer(request.source, answer, chunks)

        # Step 4 — extract sources
        sources = list({
            f"{c['metadata'].get('source', 'unknown')} (page {c['metadata'].get('page', '?')})"
            for c in chunks
        })

        return {
            "question":          request.source,
            "answer":            answer,
            "sources":           sources,
            "chunks_used":       len(chunks),
            "faithfulness_score": hal_score["faithfulness_score"],
            "confidence_level":   hal_score["confidence_level"],
            "nli_verdict":        hal_score["nli_verdict"],
        }
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

class ScoreRequest(BaseModel):
    question: str
    answer:   str
    context:  str  # plain text context

@app.post("/score")
async def score(request: ScoreRequest):
    """
    Score any question/answer pair for hallucination risk.
    Useful for evaluating answers from outside the system.
    """
    try:
        # Wrap context as a single chunk for the scorer
        chunks = [{"content": request.context, "metadata": {}}]
        result = score_answer(request.question, request.answer, chunks)
        return result
    except Exception as e:
        logger.error(f"Scoring failed: {e}")
        raise HTTPException(status_code=500, detail="Scoring failed — check server logs")