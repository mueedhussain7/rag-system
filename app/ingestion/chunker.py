# app/ingestion/chunker.py
import logging
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config import settings

logger = logging.getLogger(__name__)

def chunk_documents(documents: list[Document]) -> list[Document]:
    """
    Split documents into smaller chunks.

    Why RecursiveCharacterTextSplitter?
    It tries to split on paragraph breaks first, then sentences,
    then words — so chunks always end at natural boundaries, not
    mid-sentence.

    Why overlap?
    If a sentence spans a chunk boundary, both chunks contain it.
    This prevents losing context at the edges.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)
    logger.info(f"Split {len(documents)} document(s) into {len(chunks)} chunks")
    return chunks