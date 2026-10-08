from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    database_url: str
    ollama_host: str = "http://127.0.0.1:11434"
    # gemma4:12b -- natively multimodal (text/image/audio, no separate
    # vision encoder to wire up), used for *every* generation/structured-
    # output call, including describing an uploaded image (see
    # app/ingestion/pipeline.py's _describe_image). Migrated from
    # qwen2.5:14b-instruct-q4_K_M; see docs/DECISIONS.md for why, and for
    # the measured VRAM cost. Deliberately the *only* generation model
    # used anywhere in the app -- this GPU has 11GB VRAM and gemma4:12b
    # alone already measures ~8.7GB alongside nomic-embed-text's ~0.25GB,
    # leaving no realistic headroom for a second generation model without
    # risking the eviction/reload thrashing CLAUDE.md documents.
    membase_generation_model: str = "gemma4:12b"
    membase_embed_model: str = "nomic-embed-text"
    membase_embed_dim: int = 768
    uploads_dir: Path = Path(".data/uploads")
    membase_max_upload_mb: int = 50
    llm_timeout_seconds: float = 90.0
    # Bounds on how much of one attachment gets expensive per-chunk LLM/
    # embedding treatment -- see docs/DECISIONS.md "Large attachments".
    # The *original* file and its full extracted text are never truncated
    # by this; only how many AttachmentChunk rows get embedded for deep
    # chunk-level search into a single huge document.
    membase_max_chunks_per_attachment: int = 600
    # One captioning call per image already bounds per-image cost to O(1)
    # regardless of resolution (the vision encoder's own fixed tiling
    # handles that) -- this instead bounds the number of images in a
    # *single upload batch* that get captioned, so attaching dozens of
    # images to one capture at once doesn't turn one request into dozens
    # of sequential vision calls with no visible bound. Enforced per
    # call to _caption_new_images(), i.e. per create_capture()'s batch
    # of uploads or per individual add_attachment() call -- NOT a
    # cumulative budget across a capture's lifetime, so repeatedly
    # calling add_attachment() one file at a time is not bounded by
    # this across calls (each call gets its own fresh budget). The
    # risk this exists for -- attaching many images in one request --
    # is fully covered either way.
    membase_max_images_captioned_per_capture: int = 8


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    return settings
