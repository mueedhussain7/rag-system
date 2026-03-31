import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from ragas import SingleTurnSample
from ragas.metrics.collections import Faithfulness
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from app.config import settings

logger = logging.getLogger(__name__)


def _run_ragas(question: str, answer: str, contexts: list[str]) -> float:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        llm = LangchainLLMWrapper(ChatOpenAI(
            model="gpt-4o-mini",
            openai_api_key=settings.openai_api_key,
        ))

        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
        )

        metric = Faithfulness(llm=llm)  # ← fixed
        score  = loop.run_until_complete(metric.single_turn_ascore(sample))
        return round(float(score), 3)
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
        # Run in a separate thread with its own clean event loop
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_run_ragas, question, answer, contexts)
            result = future.result(timeout=60)

        logger.info(f"Faithfulness score: {result} for: '{answer[:60]}'")
        return result

    except Exception as e:
        logger.error(f"RAGAS scoring failed: {e}")
        return -1.0