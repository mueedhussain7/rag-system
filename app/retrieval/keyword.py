import logging
from rank_bm25 import BM25Okapi
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from app.config import settings

logger = logging.getLogger(__name__)

def keyword_search(query: str, k: int = 20) -> list[dict]:
    """
    BM25 keyword search over all chunks in ChromaDB.

    Why load from ChromaDB?
    We want both search methods to work on the same set of documents.
    BM25 doesn't use vectors — it scores based on word frequency.

    Why BM25 over simple word matching?
    BM25 accounts for document length and term frequency, making it
    much more accurate than just counting word matches.
    """
    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        openai_api_key=settings.openai_api_key,
    )
    store = Chroma(
        collection_name=settings.chroma_collection_name,
        embedding_function=embeddings,
        persist_directory=settings.chroma_db_path,
    )

    # Load all stored chunks (text + metadata)
    all_docs = store.get()
    if not all_docs["documents"]:
        logger.warning("ChromaDB collection is empty — no docs to keyword search")
        return []

    documents = all_docs["documents"]
    metadatas = all_docs["metadatas"]

    # Tokenise each chunk into words for BM25
    tokenised = [doc.lower().split() for doc in documents]
    bm25 = BM25Okapi(tokenised)

    # Score every chunk against the query
    query_tokens = query.lower().split()
    scores = bm25.get_scores(query_tokens)

    # Pair each chunk with its score and sort — highest first
    scored = sorted(
        zip(documents, metadatas, scores),
        key=lambda x: x[2],
        reverse=True,
    )

    output = []
    for content, metadata, score in scored[:k]:
        output.append({
            "content":  content,
            "metadata": metadata,
            "score":    round(float(score), 4),
            "method":   "keyword",
        })

    logger.info(f"BM25 search returned {len(output)} results for: '{query[:60]}'")
    return output