import logging
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import tiktoken
from app.config import settings
from app.generation.prompt import RAG_PROMPT
from app.retrieval.hybrid import hybrid_search
from app.retrieval.context import assemble_context

logger = logging.getLogger(__name__)

MAX_CONTEXT_TOKENS = 4000
ENCODING = tiktoken.encoding_for_model("gpt-4o-mini")

def _validate_and_truncate_context(context: str) -> str:
    """
    Ensure context doesn't exceed token limits.
    Truncates from the end if needed, preserving earlier chunks.
    """
    tokens = ENCODING.encode(context)
    if len(tokens) <= MAX_CONTEXT_TOKENS:
        return context

    logger.warning(f"Context exceeds {MAX_CONTEXT_TOKENS} tokens ({len(tokens)} actual), truncating")
    truncated_tokens = tokens[:MAX_CONTEXT_TOKENS]
    return ENCODING.decode(truncated_tokens)


def get_llm(streaming: bool = False) -> ChatOpenAI:
    """
    Returns a GPT-4o model instance.
    streaming=True means tokens are sent to the client as they generate,
    rather than waiting for the full response.
    """
    return ChatOpenAI(
        model="gpt-4o-mini",       
        temperature=0,             # 0 = deterministic, no randomness, less hallucination
        streaming=streaming,
        openai_api_key=settings.openai_api_key,
    )

def build_rag_chain(streaming: bool = False):
    """
    Builds a LangChain pipeline:
    prompt → LLM → string parser

    The | operator chains these together, output of each step
    flows into the next automatically.
    """
    llm    = get_llm(streaming=streaming)
    parser = StrOutputParser()
    return RAG_PROMPT | llm | parser

def ask(question: str) -> dict:
    """
    Full RAG pipeline, non-streaming version.
    1. Retrieve relevant chunks
    2. Assemble context
    3. Validate context length
    4. Generate grounded answer
    5. Return answer + sources
    """
    # Step 1 — retrieve
    chunks  = hybrid_search(question, top_k=5)
    context = assemble_context(chunks)

    # Step 2 — validate context
    context = _validate_and_truncate_context(context)

    # Step 3 — generate
    chain  = build_rag_chain(streaming=False)
    answer = chain.invoke({
        "context":  context,
        "question": question,
    })

    # Step 3 — extract unique sources for citations
    sources = list({
        f"{c['metadata'].get('source', 'unknown')} (page {c['metadata'].get('page', '?')})"
        for c in chunks
    })

    logger.info(f"Generated answer for: '{question[:60]}'")
    return {
        "question": question,
        "answer":   answer,
        "sources":  sources,
        "chunks_used": len(chunks),
    }
