# Architecture

## Components

```
Browser (Jinja2-rendered HTML + vanilla JS)
        |  fetch() -- JSON over HTTP, same origin
        v
FastAPI app (backend/app)
  ├── api/memories.py   -- POST /api/memories, GET .../{id}, .../clarify, .../attachments
  ├── api/recall.py     -- POST /api/recall
  ├── web/routes.py     -- GET /, GET /recall (serves the HTML shell)
  ├── ingestion/        -- file processing, metadata extraction, embedding
  ├── retrieval/        -- hybrid search, fusion, grounded answer generation
  ├── llm/              -- LLMProvider abstraction + Ollama implementation
  └── models.py / db.py -- SQLAlchemy models, session management
        |
        v
PostgreSQL 17 + pgvector  (one process; see docs/DECISIONS.md)
Ollama (system service, already running; see CLAUDE.md)
```

The browser never talks to Postgres or Ollama directly, and never contains
business logic beyond form handling and rendering — it only calls the
JSON API. That's the "separate frontend from memory service" requirement,
satisfied without a second deployable process.

## Data flow: ingestion

```
caption + files
      |
      v
create_memory()                         [app/ingestion/pipeline.py]
      |
      ├─ for each file: process_upload() [app/ingestion/files.py]
      |     - sniff real mime type (not trusting filename/header)
      |     - reject unsupported types outright
      |     - save original bytes under .data/uploads/<memory_id>/
      |     - best-effort text extraction (pdf/txt/md; images: none, see
      |       docs/TODO.md)
      |
      ├─ extract_metadata()              [app/ingestion/metadata.py]
      |     - LLM structured-output call: people, location, topics,
      |       event_date, summary, (optional) clarification_question
      |     - never raises: LLM failure -> memory still persists with
      |       caption only (see "data integrity" below)
      |
      ├─ build searchable_text = caption + attachment text + derived
      |   people/topics/location/summary  (feeds the GIN full-text index)
      |
      └─ embed(caption + summary + topics) -> pgvector column
            - best-effort; failure leaves embedding NULL, memory still
              full-text searchable
      |
      v
Postgres: INSERT memories (+ attachments), single transaction
```

If `extract_metadata` surfaced a `clarification_question`, the API returns
it in `pending_clarification`; answering it
(`POST /memories/{id}/clarify`) merges the answer into `derived`,
re-extends `searchable_text`, and re-embeds. The memory is NOT held back
pending an answer — see docs/DECISIONS.md ("clarification doesn't gate
persistence").

## Data flow: retrieval

```
query
  |
  ├─ embed(query) -> vector search (pgvector, cosine, HNSW index)
  |     top 20 by similarity, kept if similarity >= 0.55
  |
  ├─ Postgres full-text search (websearch_to_tsquery, GIN index)
  |     top 20 by ts_rank, kept if matched at all
  |
  ├─ fuse: 0.65 * normalized-vector-score + 0.35 * normalized-fts-score
  |     (this fused score *is* the MVP's "reranking" step -- no separate
  |     cross-encoder model; see docs/DECISIONS.md)
  |
  ├─ take top 5 -> format as evidence blocks (caption, derived summary/
  |     people/location/event_date, attachment excerpts)
  |
  └─ LLM generate_structured({answer, grounded}, evidence)
        - system prompt: answer ONLY from evidence, partial answers are
          fine, grounded=false only if evidence is about something else
        - grounded=false (or no candidates at all) -> the API
          deterministically substitutes the fixed string
          "I don't have a stored memory about this." and empty sources,
          regardless of what text the model wrote. This is enforced in
          code (app/retrieval/pipeline.py:answer()), not just prompted.
```

## Memory schema

```
memories
  id                      uuid, pk
  caption                 text        -- AUTHORITATIVE, never overwritten
  created_at, updated_at  timestamptz
  derived                 jsonb       -- people/location/topics/event_date/
                                          summary/enrichment_status/
                                          clarification_answers (all derived)
  pending_clarification   jsonb null  -- {question, field} or null
  searchable_text         text        -- caption + attachment text + derived
                                          text, feeds `fts` below
  fts                     tsvector    -- GENERATED from searchable_text, GIN
  embedding               vector(768) null
  embedding_model         text null   -- which model produced `embedding`
  embedding_dim           int null

attachments
  id, memory_id (fk, cascade delete)
  original_filename, stored_path, content_type, size_bytes
  extracted_text          text null  -- best-effort
  extraction_status       text       -- ok | unsupported | failed | skipped
  created_at
```

`caption` and the files under `.data/uploads/` are the only data a user
typed or uploaded; everything else is machine-derived and in principle
recomputable from those two things (see "data integrity" below). Original
files are never deleted or rewritten by any derived-data recomputation.

## LLM integration

`app/llm/base.py` defines `LLMProvider` (`generate`, `generate_structured`,
`embed`) and three exception types (`LLMUnavailableError`,
`LLMTimeoutError`, `LLMMalformedOutputError`) that ingestion/retrieval
catch explicitly rather than letting a generic exception propagate.
`app/llm/ollama_provider.py` is the only implementation, calling Ollama's
`/api/generate` (with a JSON-schema `format` for structured output) and
`/api/embed`. Swapping to llama.cpp's server or another OpenAI-compatible
backend means writing one new class against this interface — no changes
to ingestion/retrieval.

## File handling

Real file type is sniffed from bytes (`python-magic`), not trusted from
the filename or `Content-Type` header, before anything is written to
disk. Supported: PDF (text extraction via `pypdf`), `.txt`/`.md` (direct
decode), PNG/JPEG/GIF/WEBP (validated as real images via Pillow, but not
captioned/OCR'd — no local vision model is wired in for the MVP). Anything
else is rejected with a 400 at upload time, not silently dropped.

## Persistence

Everything writable lives under `.data/` (gitignored): Postgres's own data
directory, its unix socket directory, and `uploads/<memory_id>/<file>`
for original attachment bytes. Nothing is written into the Nix store or
outside the repo. See `CLAUDE.md` for why this matters on NixOS
specifically.

## Data integrity

`caption`, uploaded files, and timestamps are authoritative and are never
overwritten by any derived-data process. `derived`, `searchable_text`,
`fts`, and `embedding` are all recomputable from the authoritative data
plus whatever the current metadata-extraction/embedding logic produces --
there is deliberately no code path where a derived summary becomes the
only surviving record of a fact. (There is, however, no backfill/reindex
*tool* yet if you change the embedding model or extraction prompt after
the fact -- see docs/TODO.md.)
