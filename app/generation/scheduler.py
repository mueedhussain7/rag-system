import logging
from pathlib import Path
from apscheduler.schedulers.background import BackgroundScheduler
from app.ingestion.loaders import load_document
from app.ingestion.chunker import chunk_documents
from app.ingestion.embedder import ingest_chunks
from app.config import settings
from app.evaluation.logger import get_sources_needing_refresh, log_source_refresh

logger = logging.getLogger(__name__)

def refresh_documents():
    """
    Scans the documents folder and re-ingests any new files.
    Also refreshes any previously ingested URLs that are due for refresh.
    Already-ingested files are skipped automatically by the
    duplicate detection in ingest_chunks().

    This runs in the background on a schedule — the web server
    keeps serving requests while this runs in a separate thread.
    """
    items_refreshed = 0

    # Refresh local files
    docs_path = Path("data/documents")
    if docs_path.exists():
        pdf_files = list(docs_path.glob("*.pdf"))
        txt_files = list(docs_path.glob("*.txt"))
        all_files = pdf_files + txt_files

        if all_files:
            logger.info(f"Auto-refresh started — scanning {len(all_files)} local file(s)")
            for file_path in all_files:
                try:
                    docs   = load_document(str(file_path))
                    chunks = chunk_documents(docs)
                    result = ingest_chunks(chunks, str(file_path))
                    logger.info(f"Refresh: {file_path.name} → {result['status']}")
                    if result['status'] == 'success':
                        log_source_refresh(str(file_path), 'file')
                        items_refreshed += 1
                except Exception as e:
                    logger.error(f"Refresh failed for {file_path.name}: {e}")

    # Refresh URLs that need updating
    urls_to_refresh = get_sources_needing_refresh(hours_since_refresh=24)
    if urls_to_refresh:
        logger.info(f"Refreshing {len(urls_to_refresh)} URL(s)")
        for url in urls_to_refresh:
            try:
                docs   = load_document(url)
                chunks = chunk_documents(docs)
                result = ingest_chunks(chunks, url)
                logger.info(f"Refresh: {url[:60]}... → {result['status']}")
                if result['status'] == 'success':
                    log_source_refresh(url, 'url')
                    items_refreshed += 1
            except Exception as e:
                logger.error(f"Refresh failed for {url}: {e}")

    logger.info(f"Auto-refresh complete — {items_refreshed} item(s) updated")


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