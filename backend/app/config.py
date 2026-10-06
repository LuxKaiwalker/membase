from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    database_url: str
    ollama_host: str = "http://127.0.0.1:11434"
    # qwen2.5:14b-instruct-q4_K_M, not the 7b variant: segmentation and
    # enrichment need real judgment (see docs/DECISIONS.md) and 7b visibly
    # under-performs on them. Deliberately the *only* generation model used
    # anywhere in the app -- mixing it with another generation model would
    # exceed this GPU's 11GB VRAM alongside nomic-embed-text and cause a
    # ~7-14s reload stall on every alternating call (see CLAUDE.md).
    membase_generation_model: str = "qwen2.5:14b-instruct-q4_K_M"
    membase_embed_model: str = "nomic-embed-text"
    membase_embed_dim: int = 768
    uploads_dir: Path = Path(".data/uploads")
    membase_max_upload_mb: int = 25
    llm_timeout_seconds: float = 90.0


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    return settings
