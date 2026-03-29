import logging
from app.retrieval.semantic import semantic_search
from app.retrieval.keyword import keyword_search

logger = logging.getLogger(__name__)

def reciprocal_rank_fusion(
    semantic_results: list[dict],
    keyword_results:  list[dict],
    k: int = 60,
) -> list[dict]:
    """
    Reciprocal Rank Fusion (RRF) — merges two ranked lists into one.

    How it works:
    Each chunk gets a score of 1/(rank + k) from each list.
    A chunk ranked #1 in semantic gets 1/61, ranked #1 in keyword also
    gets 1/61. If the same chunk appears in both lists, its scores add up
    — rewarding chunks that multiple methods agree on.

    k=60 is the standard value from the original RRF paper.
    """
    scores: dict[str, dict] = {}

    for rank, result in enumerate(semantic_results):
        key = result["content"][:100]  # use first 100 chars as unique key
        if key not in scores:
            scores[key] = {"data": result, "rrf_score": 0.0}
        scores[key]["rrf_score"] += 1.0 / (rank + k)

    for rank, result in enumerate(keyword_results):
        key = result["content"][:100]
        if key not in scores:
            scores[key] = {"data": result, "rrf_score": 0.0}
        scores[key]["rrf_score"] += 1.0 / (rank + k)

    # Sort by combined RRF score
    sorted_results = sorted(
        scores.values(),
        key=lambda x: x["rrf_score"],
        reverse=True,
    )

    return [item["data"] for item in sorted_results]


def hybrid_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Full hybrid retrieval pipeline:
    1. Semantic search  → top 20 results
    2. BM25 keyword search  → top 20 results
    3. RRF fusion  → merge and deduplicate
    4. Return top_k results

    Why top 20 for each then narrow to 5?
    Cast a wide net first to maximise recall, then be selective.
    The final 5 chunks are what gets sent to GPT-4 as context.
    """
    semantic_results = semantic_search(query, k=20)
    keyword_results  = keyword_search(query,  k=20)

    fused = reciprocal_rank_fusion(semantic_results, keyword_results)
    top   = fused[:top_k]

    logger.info(f"Hybrid search: {len(semantic_results)} semantic + "
                f"{len(keyword_results)} keyword → top {len(top)} after fusion")
    return top