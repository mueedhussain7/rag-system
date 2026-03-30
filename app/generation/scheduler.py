import logging
from pathlib import Path
from apscheduler.schedulers.background import BackgroundScheduler
from app.ingestion.loaders import load_document
from app.ingestion.chunker import chunk_documents
from app.ingestion.embedder import ingest_chunks
from app.config import settings

logger = logging.getLogger(__name__)

def refresh_documents():
    """
    Scans the documents folder and re-ingests any new files.
    Already-ingested files are skipped automatically by the
    duplicate detection in ingest_chunks().

    This runs in the background on a schedule — the web server
    keeps serving requests while this runs in a separate thread.
    """
    docs_path = Path("data/documents")
    if not docs_path.exists():
        logger.warning("data/documents folder not found — skipping refresh")
        return

    pdf_files = list(docs_path.glob("*.pdf"))
    txt_files = list(docs_path.glob("*.txt"))
    all_files = pdf_files + txt_files

    if not all_files:
        logger.info("No documents found to refresh")
        return

    logger.info(f"Auto-refresh started — scanning {len(all_files)} file(s)")
    for file_path in all_files:
        try:
            docs   = load_document(str(file_path))
            chunks = chunk_documents(docs)
            result = ingest_chunks(chunks, str(file_path))
            logger.info(f"Refresh: {file_path.name} → {result['status']}")
        except Exception as e:
            logger.error(f"Refresh failed for {file_path.name}: {e}")

    logger.info("Auto-refresh complete")


def start_scheduler() -> BackgroundScheduler:
    """
    Starts the background scheduler.
    Runs refresh_documents() every 24 hours automatically.
    Also runs once immediately on startup so new files are picked up right away.
    """
    scheduler = BackgroundScheduler()
    scheduler.add_job(
        refresh_documents,
        trigger="interval",
        hours=24,
        id="document_refresh",
        name="Auto-refresh documents",
        replace_existing=True,
    )
    scheduler.start()
    logger.info("Document auto-refresh scheduler started — runs every 24 hours")
    return scheduler