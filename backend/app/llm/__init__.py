from functools import lru_cache

from app.config import get_settings
from app.llm.base import (
    LLMError,
    LLMMalformedOutputError,
    LLMProvider,
    LLMTimeoutError,
    LLMUnavailableError,
)
from app.llm.ollama_provider import OllamaProvider

__all__ = [
    "LLMError",
    "LLMMalformedOutputError",
    "LLMProvider",
    "LLMTimeoutError",
    "LLMUnavailableError",
    "get_llm_provider",
]


@lru_cache
def get_llm_provider() -> LLMProvider:
    settings = get_settings()
    return OllamaProvider(
        host=settings.ollama_host,
        generation_model=settings.membase_generation_model,
        embed_model=settings.membase_embed_model,
        embed_dim=settings.membase_embed_dim,
        timeout_seconds=settings.llm_timeout_seconds,
    )
