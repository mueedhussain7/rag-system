# app/ingestion/embedder.py
import hashlib
import logging
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from app.config import settings

logger = logging.getLogger(__name__)

def get_vector_store() -> Chroma:
    """
    Returns a ChromaDB vector store instance.
    Creates the collection if it doesn't exist yet.
    """
    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        openai_api_key=settings.openai_api_key,
    )
    return Chroma(
        collection_name=settings.chroma_collection_name,
        embedding_function=embeddings,
        persist_directory=settings.chroma_db_path,
    )

def document_hash(source: str) -> str:
    """Unique fingerprint for a source — used for duplicate detection."""
    return hashlib.md5(source.encode()).hexdigest()

def ingest_chunks(chunks: list[Document], source: str) -> dict:
    """
    Embed chunks and store them in ChromaDB.
    Skips ingestion if the source was already ingested.
    """
    store = get_vector_store()
    doc_id = document_hash(source)

    # Duplicate detection
    existing = store.get(where={"doc_id": doc_id})
    if existing and existing["ids"]:
        logger.info(f"Skipping duplicate: {source} already in ChromaDB")
        return {"status": "skipped", "reason": "already ingested", "source": source}

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