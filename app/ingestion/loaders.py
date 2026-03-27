# app/ingestion/loaders.py
import logging
from pathlib import Path
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    WebBaseLoader,
)
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

def load_document(source: str) -> list[Document]:
    """
    Load a document from a file path or URL.
    Returns a list of LangChain Document objects.
    Each Document has .page_content (text) and .metadata (source info).
    """
    # Web URL
    if source.startswith("http://") or source.startswith("https://"):
        logger.info(f"Loading web page: {source}")
        loader = WebBaseLoader(source)
        return loader.load()

    path = Path(source)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {source}")

    suffix = path.suffix.lower()

    if suffix == ".pdf":
        logger.info(f"Loading PDF: {source}")
        loader = PyPDFLoader(str(path))
        return loader.load()

    if suffix == ".txt":
        logger.info(f"Loading text file: {source}")
        loader = TextLoader(str(path), encoding="utf-8")
        return loader.load()

    raise ValueError(f"Unsupported file type: {suffix}. Supported: .pdf, .txt, URLs")