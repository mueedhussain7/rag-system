# app/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    Central configuration for the entire RAG system.
    Pydantic reads values from the .env file automatically.
    """
    openai_api_key: str
    app_env: str = "development"
    app_version: str = "0.1.0"
    log_level: str = "INFO"

    # Modern Pydantic V2 style — replaces the old inner class Config
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

# Single shared instance — every module imports this
settings = Settings()