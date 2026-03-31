import logging
from app.hallucination.ragas_scorer import score_faithfulness
from app.hallucination.nli_checker import nli_check

logger = logging.getLogger(__name__)


def get_confidence_label(faithfulness_score: float, nli_verdict: str) -> str:
    """
    Converts numeric scores into a human-readable confidence level.

    high      → safe to show the user as-is
    medium    → probably fine but worth reviewing
    low       → likely contains unsupported claims
    unverified → scoring failed for some reason
    """
    if faithfulness_score < 0:
        return "unverified"
    if faithfulness_score >= 0.8 and nli_verdict == "clean":
        return "high"
    if faithfulness_score >= 0.5 and nli_verdict != "contradicted":
        return "medium"
    return "low"


def score_answer(
    question: str,
    answer: str,
    chunks: list[dict],
) -> dict:
    """
    Full hallucination scoring pipeline.

    Takes the question, generated answer, and retrieved chunks.
    Returns faithfulness score, NLI results, and confidence label.
    """
    # Extract plain text from chunks for scoring
    contexts = [c["content"] for c in chunks]
    full_context = "\n\n".join(contexts)

    # Run both checks
    faithfulness_score = score_faithfulness(question, answer, contexts)
    nli_result         = nli_check(answer, full_context)
    confidence         = get_confidence_label(faithfulness_score, nli_result["verdict"])

    logger.info(
        f"Hallucination score: faithfulness={faithfulness_score} "
        f"confidence={confidence} verdict={nli_result['verdict']}"
    )

    return {
        "faithfulness_score": faithfulness_score,
        "confidence_level":   confidence,
        "nli_verdict":        nli_result["verdict"],
        "nli_details":        nli_result,
    }