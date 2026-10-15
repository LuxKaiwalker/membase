from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    database_url: str
    # The one origin allowed to make cross-origin requests to this API
    # (see app/main.py's CORSMiddleware) -- the published GitHub Pages
    # frontend's origin in production. No wildcard: authentication is
    # no longer this app's job (see docs/DECISIONS.md "Retire app-level
    # auth for nginx Basic Auth"), so CORS is the one remaining layer
    # deciding which *websites* may call this API from a browser at
    # all, as opposed to who may call it directly (e.g. curl, which
    # CORS never restricts -- it's a browser-enforced policy, not a
    # server-side access control).
    frontend_origin: str = "http://127.0.0.1:8000"
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
    # Found during a deployment-readiness audit (see docs/DEPLOYMENT.md):
    # unlike file uploads, the raw `text` field had no server-side size
    # limit at all -- an unauthenticated caller could post an arbitrarily
    # large body, which the chunking logic (see docs/DECISIONS.md "Chunk
    # long captures before analysis") would then amplify into many real
    # LLM calls. 50,000 chars is generously above any real single memory
    # (the streamlining phase's own ~2,000-char, 20-fact stress test
    # chunked into just a handful of calls) while still bounding the
    # worst case.
    membase_max_text_chars: int = 50_000
    llm_timeout_seconds: float = 90.0
    # Bounds on how much of one attachment gets expensive per-chunk LLM/
    # embedding treatment -- see docs/DECISIONS.md "Large attachments".
    # The *original* file and its full extracted text are never truncated
    # by this; only how many AttachmentChunk rows get embedded for deep
    # chunk-level search into a single huge document.
    membase_max_chunks_per_attachment: int = 600
    # Streamlining phase additions (see docs/DECISIONS.md):
    # A background unit-processing lease older than this is treated as
    # abandoned (e.g. the server restarted mid-task) and safe for
    # POST /retry to reclaim -- generous above llm_timeout_seconds (90s)
    # since this now covers at most one generation + one embed call, not
    # several.
    membase_processing_lease_stale_seconds: float = 300.0
    # Below this many characters, an attachment's extracted text is just
    # folded into the parent unit's full-text search (already free) --
    # chunk-level embedding is skipped entirely, since a short document
    # has nothing a single excerpt doesn't already cover. Chosen as
    # roughly "a couple of chunks' worth" (chunking.py's MAX_CHUNK_CHARS
    # is 1200) -- an engineering default, not swept against an eval set.
    membase_min_chars_for_chunking: int = 2000
    # How many existing ready units to check a new unit's relations
    # against for a potential factual conflict (see docs/DECISIONS.md
    # "Contradiction detection") -- reuses the same hybrid_search() call
    # already being made for retrieval, just capped small since this is
    # meant to stay cheap, not an exhaustive scan of the whole store.
    membase_conflict_check_candidates: int = 5


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    return settings
