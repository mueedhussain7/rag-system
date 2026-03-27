# app/main.py
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.config import settings

logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Everything before yield runs on startup
    logger.info(f"RAG System starting up | env={settings.app_env} | version={settings.app_version}")
    yield
    # Everything after yield runs on shutdown
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