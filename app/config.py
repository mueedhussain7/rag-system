from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # ── OpenAI ──────────────────────────────────────────────
    openai_api_key: str
    embedding_model: str = "text-embedding-3-small"

    # ── ChromaDB ────────────────────────────────────────────
    chroma_db_path: str = "./chroma_db"
    chroma_collection_name: str = "rag_documents"

    # ── Chunking ────────────────────────────────────────────
    chunk_size: int = 500
    chunk_overlap: int = 50

    # ── App ─────────────────────────────────────────────────
    app_env: str = "development"
    app_version: str = "0.1.0"
    log_level: str = "INFO"
    user_agent: str = "rag-system/0.1.0"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

settings = Settings()