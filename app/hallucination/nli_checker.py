import logging
import re
from langchain_openai import ChatOpenAI
from app.config import settings

logger = logging.getLogger(__name__)

NLI_PROMPT = """You are a precise fact-checker.

Given a CONTEXT and a SENTENCE, classify the sentence as one of:
- "entailed"     → the context clearly supports this sentence
- "contradicted" → the context clearly contradicts this sentence
- "neutral"      → the context neither supports nor contradicts this sentence

Reply with ONLY one word: entailed, contradicted, or neutral.

CONTEXT:
{context}

SENTENCE:
{sentence}"""


def _split_sentences(text: str) -> list[str]:
    """
    Split text into sentences, handling common cases like abbreviations.
    Works with periods, question marks, and exclamation marks.
    """
    text = text.strip()
    if not text:
        return []

    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def check_sentence(sentence: str, context: str, llm: ChatOpenAI) -> str:
    """Classify a single sentence as entailed, contradicted, or neutral."""
    prompt = NLI_PROMPT.format(context=context, sentence=sentence)
    response = llm.invoke(prompt)
    label = response.content.strip().lower()

    # Normalise — only accept valid labels
    if label not in {"entailed", "contradicted", "neutral"}:
        return "neutral"
    return label


def nli_check(answer: str, context: str) -> dict:
    """
    Runs NLI check on every sentence in the answer.

    Returns a summary:
    - per-sentence labels
    - counts of entailed / contradicted / neutral
    - overall verdict
    """
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=settings.openai_api_key,
    )

    # Split answer into sentences
    sentences = _split_sentences(answer)

    results = []
    for sentence in sentences:
        if len(sentence) < 10:  # skip very short fragments
            continue
        label = check_sentence(sentence, context, llm)
        results.append({"sentence": sentence, "label": label})

    # Count labels
    counts = {"entailed": 0, "contradicted": 0, "neutral": 0}
    for r in results:
        counts[r["label"]] += 1

    total = len(results)
    verdict = "clean"
    if counts["contradicted"] > 0:
        verdict = "contradicted"
    elif total > 0 and counts["entailed"] / total < 0.5:
        verdict = "uncertain"

    logger.info(f"NLI check: {counts} → verdict: {verdict}")
    return {
        "sentences": results,
        "counts":    counts,
        "verdict":   verdict,
    }