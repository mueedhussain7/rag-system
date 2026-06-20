# app/ingestion/embedder.py
import hashlib
import logging
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from app.config import settings

logger = logging.getLogger(__name__)

_embeddings = None
_store = None

def _get_embeddings() -> OpenAIEmbeddings:
    """Returns cached embeddings instance."""
    global _embeddings
    if _embeddings is None:
        _embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            openai_api_key=settings.openai_api_key,
        )
    return _embeddings

def get_vector_store() -> Chroma:
    """
    Returns a cached ChromaDB vector store instance.
    Creates the collection if it doesn't exist yet.
    """
    global _store
    if _store is None:
        _store = Chroma(
            collection_name=settings.chroma_collection_name,
            embedding_function=_get_embeddings(),
            persist_directory=settings.chroma_db_path,
        )
    return _store

def document_hash(source: str, content_hash: str = "") -> str:
    """
    Create unique fingerprint combining source path and content.
    If content_hash provided, uses it; otherwise just uses source.
    This allows detecting when document content changes.
    """
    combined = f"{source}:{content_hash}"
    return hashlib.md5(combined.encode()).hexdigest()


def compute_content_hash(chunks: list) -> str:
    """Compute hash of all chunk content combined."""
    content = "".join([c.page_content for c in chunks])
    return hashlib.md5(content.encode()).hexdigest()

def ingest_chunks(chunks: list[Document], source: str) -> dict:
    """
    Embed chunks and store them in ChromaDB.
    Skips ingestion only if the exact same document (source + content) was already ingested.
    If content changed, re-ingests and replaces old version.
    """
    store = get_vector_store()
    content_hash = compute_content_hash(chunks)
    doc_id = document_hash(source, content_hash)

    # Check if this exact version already exists
    existing = store.get(where={"doc_id": doc_id})
    if existing and existing["ids"]:
        logger.info(f"Skipping duplicate: {source} (content unchanged) already in ChromaDB")
        return {"status": "skipped", "reason": "already ingested", "source": source}

    # If source exists but with different content, delete old version
    old_docs = store.get(where={"source": source})
    if old_docs and old_docs["ids"]:
        logger.info(f"Removing outdated version of {source}")
        store.delete(ids=old_docs["ids"])

    # Attach metadata to every chunk
    for i, chunk in enumerate(chunks):
        chunk.metadata.update({
            "doc_id": doc_id,
            "source": source,
            "chunk_index": i,
            "total_chunks": len(chunks),
        })

    store.add_documents(chunks)
    logger.info(f"Ingested {len(chunks)} chunks from: {source}")

    return {
        "status": "success",
        "source": source,
        "chunks_ingested": len(chunks),
        "doc_id": doc_id,
    }