import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from openai import AsyncOpenAI
from ragas.metrics.collections import Faithfulness
from ragas.llms import llm_factory
from app.config import settings

logger = logging.getLogger(__name__)


def _run_ragas(question: str, answer: str, contexts: list[str]) -> float:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
        llm    = llm_factory("gpt-4o-mini", client=openai_client)
        metric = Faithfulness(llm=llm)

        score = loop.run_until_complete(
            metric.ascore(
                user_input=question,
                response=answer,
                retrieved_contexts=contexts,
            )
        )
        return round(float(score), 3)

    except Exception as e:
        import traceback
        logger.error(f"RAGAS scoring failed: {e}")
        logger.error(traceback.format_exc())
        return -1.0

    finally:
        loop.close()


def score_faithfulness(
    question: str,
    answer: str,
    contexts: list[str],
) -> float:
    """
    Scores how faithful the answer is to the retrieved context.
    Returns 0.0–1.0. Returns -1.0 if scoring fails.
    """
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_run_ragas, question, answer, contexts)
            result = future.result(timeout=60)
        logger.info(f"Faithfulness score: {result} for: '{answer[:60]}'")
        return result
    except Exception as e:
        logger.error(f"score_faithfulness outer failed: {e}")
        return -1.0