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

    # --- associative memory graph (see docs/ARCHITECTURE.md "Associative
    # memory graph" and app/memory/graph.py).
    #
    # All three exist to answer the same question -- "when is a connection
    # worth storing?" -- from three directions, because the single biggest
    # failure mode of an automatically-built memory graph is not missing edges
    # but degenerating into one where everything connects to everything.
    #
    # An entity linked by more than this share of all ready memories is
    # excluded from *generating association edges* (its link rows are still
    # kept, as real provenance). Your own home city or employer legitimately
    # appears in half your memories and says nothing about which two are
    # related. 0.5 is an engineering default, not swept -- it only starts to
    # bind once the corpus is large enough for an entity to reach that share,
    # and the specificity weighting (see entity_specificity) already discounts
    # common entities smoothly below it.
    membase_graph_entity_df_ceiling: float = 0.5
    # The ceiling above is a fraction, so it is meaningless on a small corpus:
    # with 3 stored memories an entity in 2 of them is already at 0.67. Both
    # of these must also hold before the ceiling may exclude an entity --
    # without them, the most informative entity in a small store (a person
    # named in two of three memories) got filtered out and the association
    # feature silently did nothing until the corpus grew. Found by running the
    # real model over a four-memory corpus, not by reasoning about it.
    membase_graph_entity_df_min_units: int = 10
    membase_graph_entity_df_min_mentions: int = 3
    # Associations weaker than this are not written at all. Set relative to
    # the weight a single shared mid-specificity topic produces (~0.1), so one
    # shared generic theme is not by itself a connection, while a shared
    # person or place comfortably clears it.
    membase_graph_min_association_weight: float = 0.15
    # Hard fan-out cap per memory, keeping the strongest. Bounds both storage
    # and the worst-case cost of a 2-hop traversal (which is this value
    # squared in the limit) without needing a recursive query planner.
    membase_graph_max_associations_per_unit: int = 24


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    return settings
