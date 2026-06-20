import logging
from openai import AsyncOpenAI
from ragas.metrics.collections import Faithfulness
from ragas.llms import llm_factory
from app.config import settings

logger = logging.getLogger(__name__)

_openai_client = None

def _get_async_client() -> AsyncOpenAI:
    """Returns cached AsyncOpenAI client."""
    global _openai_client
    if _openai_client is None:
        _openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
    return _openai_client


async def _run_ragas_async(question: str, answer: str, contexts: list[str]) -> float:
    """Async RAGAS faithfulness scoring."""
    try:
        client = _get_async_client()
        llm = llm_factory("gpt-4o-mini", client=client)
        metric = Faithfulness(llm=llm)

        score = await metric.ascore(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
        )
        return round(float(score), 3)

    except Exception as e:
        logger.error(f"RAGAS scoring failed: {e}", exc_info=True)
        return -1.0


def score_faithfulness(
    question: str,
    answer: str,
    contexts: list[str],
) -> float:
    """
    Scores how faithful the answer is to the retrieved context.
    Returns 0.0–1.0. Returns -1.0 if scoring fails.

    Uses asyncio.run() to execute async code from sync context.
    """
    try:
        import asyncio
        result = asyncio.run(_run_ragas_async(question, answer, contexts))
        logger.info(f"Faithfulness score: {result} for: '{answer[:60]}'")
        return result
    except Exception as e:
        logger.error(f"score_faithfulness failed: {e}", exc_info=True)
        return -1.0