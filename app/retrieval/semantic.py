import logging
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
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

def _get_store() -> Chroma:
    """Returns cached vector store instance."""
    global _store
    if _store is None:
        _store = Chroma(
            collection_name=settings.chroma_collection_name,
            embedding_function=_get_embeddings(),
            persist_directory=settings.chroma_db_path,
        )
    return _store

def semantic_search(query: str, k: int = 20) -> list[dict]:
    """
    Convert the query to a vector and find the k most similar
    chunks in ChromaDB using cosine similarity.

    Returns a list of dicts with 'content', 'metadata', and 'score'.
    """
    store = _get_store()

    # similarity_search_with_relevance_scores returns (Document, score) pairs
    results = store.similarity_search_with_relevance_scores(query, k=k)

    output = []
    for doc, score in results:
        output.append({
            "content":  doc.page_content,
            "metadata": doc.metadata,
            "score":    round(score, 4),
            "method":   "semantic",
        })

    logger.info(f"Semantic search returned {len(output)} results for: '{query[:60]}'")
    return output