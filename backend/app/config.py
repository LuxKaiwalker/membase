from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    database_url: str
    ollama_host: str = "http://127.0.0.1:11434"
    membase_generation_model: str = "qwen2.5:7b-instruct"
    membase_embed_model: str = "nomic-embed-text"
    membase_embed_dim: int = 768
    uploads_dir: Path = Path(".data/uploads")
    membase_max_upload_mb: int = 25
    llm_timeout_seconds: float = 60.0


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    return settings
