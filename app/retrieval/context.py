def assemble_context(chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a single context string for the prompt.

    Each chunk is numbered and labelled with its source so GPT-4
    can reference them when generating citations.
    """
    parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk["metadata"].get("source", "unknown")
        page   = chunk["metadata"].get("page",   "?")
        parts.append(
            f"[{i}] Source: {source} (page {page})\n{chunk['content']}"
        )
    return "\n\n---\n\n".join(parts)