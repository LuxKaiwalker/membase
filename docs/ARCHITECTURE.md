# Architecture

## Components

```
Browser (Jinja2-rendered HTML + vanilla JS)
        |  fetch() -- JSON over HTTP, same origin
        v
FastAPI app (backend/app)
  ├── api/captures.py   -- POST /api/captures, GET .../{id}, .../attachments
  ├── api/units.py      -- GET/clarify/retry on /api/memory-units/{id}
  ├── api/recall.py     -- POST /api/recall
  ├── web/routes.py     -- GET /, GET /recall (serves the HTML shell)
  ├── ingestion/        -- segmentation, enrichment, pipeline orchestration
  ├── retrieval/        -- hybrid search, sibling expansion, answer generation, agent
  ├── llm/              -- LLMProvider abstraction + Ollama implementation
  └── models.py / db.py -- SQLAlchemy models, session management
        |
        v
PostgreSQL 17 + pgvector  (one process; see docs/DECISIONS.md)
Ollama (system service, already running; see CLAUDE.md)
```

## Core concepts

**Capture.** What the user actually submitted in one storage interaction
(`captures` table: `id`, `raw_text`, `created_at`). `raw_text` is
authoritative and never rewritten by anything downstream.

**MemoryUnit.** One independently-retrievable piece of memory
(`memory_units` table). A capture decomposes into one or more units —
`capture_id` is the "origin group" shared by all units from the same
submission. The system decides the split, not the user (spec sections
2-5): a tightly-connected episode ("I met Anna at ETH after the lecture
and she recommended a paper") can stay one unit; unrelated facts in one
submission ("My AHV number is X. Anna studies robotics. Renew passport
in March.") become several. See "Segmentation" below.

**Authority hierarchy**, highest to lowest:

```
Capture.raw_text / Attachment originals   (user-typed/uploaded)
        >
MemoryUnit.authoritative_content           (user's own words, possibly
                                             a verbatim excerpt chosen by
                                             segmentation)
        >
MemoryUnit.clarifications                  (user-answered, append-only
                                             list of verbatim Q&A)
        >
MemoryUnit.semantic                        (LLM-derived: meaning, aliases,
                                             entities, relations, ...)
        >
retrieval_text / searchable_text / fts / embedding
                                            (fully mechanical/derived
                                             indexes, rebuilt from
                                             everything above)
```

Nothing below a given line in this hierarchy is ever treated as more
trustworthy than something above it. In particular: an LLM's guess about
what a term means never overwrites or outranks what the user actually
said, including in a clarification answer.

## Data flow: ingestion

```
capture text + files
      |
      ├─ process each file (app/ingestion/files.py): sniff real mime type,
      |   reject unsupported, save original, best-effort text extraction
      |
      ├─ COMMIT (phase 0): capture + attachments durably saved
      |
      ├─ segment_capture() (app/ingestion/segmentation.py): LLM decides
      |   how many units, each unit's content must be a verbatim excerpt
      |   of raw_text or the whole thing is kept as one unit (safety
      |   fallback -- see "Segmentation" below)
      |
      ├─ COMMIT (phase 1): every unit's authoritative_content durably saved,
      |   status="draft"
      |
      └─ for each unit: _enrich_and_index() (app/ingestion/pipeline.py)
            ├─ enrich() (app/ingestion/enrichment.py): LLM structured-output
            |   call -> meaning, aliases, entities, topics, definitions,
            |   relations, event_date, optional clarification_question
            |   (never raises: LLM failure -> unit still persists, just
            |   with semantic={"enrichment_status": "unavailable"})
            ├─ build_retrieval_text(): deterministic (non-LLM) text
            |   combining original + clarifications + meaning + aliases +
            |   definitions + relations (spec section 11)
            ├─ searchable_text = authoritative_content + attachment text
            |   + retrieval_text  (feeds the GIN full-text index)
            ├─ embed(retrieval_text) -> pgvector column (best-effort;
            |   failure leaves embedding NULL, unit still full-text
            |   searchable)
            └─ status = "ready" (or "failed" on a genuinely unexpected
                exception -- see "Failure model" below), COMMIT
```

This two-phase-commit structure is what guarantees "authoritative content
persists even if derived processing fails" (spec section 25): the
authoritative text is committed *before* any LLM call in the derived
stage runs, in a separate transaction.

If `enrich()` surfaced a `clarification_question`, it's attached to the
unit as `pending_clarification` and returned to the caller. Answering it
(`POST /memory-units/{id}/clarify`) appends the verbatim Q&A to
`clarifications`, then re-runs the *entire* derived stage (enrichment,
retrieval_text, searchable_text, embedding) with the new clarification as
additional context -- this is what makes a clarified term's aliases
actually searchable afterward, not just present in a one-off answer. The
unit does **not** wait in an unsearchable state for this to happen (see
docs/DECISIONS.md: clarification enriches, it doesn't gate).

Clarification is bounded to `MAX_CLARIFICATION_ROUNDS = 1` per unit
(`app/ingestion/pipeline.py`) -- after that, further clarification
questions the model proposes are discarded rather than surfaced.

## Segmentation

`app/ingestion/segmentation.py` asks the LLM to split `raw_text` into
units, each tagged `independence: "independent" | "contextual"` and a
`reason`. Two hard safety rules, enforced in code (not just prompted):

1. **Verbatim requirement.** Every proposed unit's `content` must be a
   normalized substring of `raw_text`. If any proposed unit fails this
   check, the *entire* proposal is discarded and the capture becomes a
   single whole-text unit instead. This is the mechanism that keeps the
   LLM from inventing "authoritative" content that the user never wrote
   -- in practice it also tends to protect genuinely coherent episodes
   from being shredded, since the model's sentence-boundary punctuation
   most often drifts from the original exactly when it's splitting
   something that shouldn't have been split (see docs/TODO.md for the
   failure mode this doesn't fully cover).
2. **Cap.** At most `MAX_SEGMENT_UNITS = 8` units per capture, to bound a
   pathological over-segmentation.

## Data flow: retrieval (the agent)

```
query
  |
  v
app/retrieval/agent.py: run_agent()  -- bounded deterministic state machine
(spec section 29; NOT a general tool-calling framework)

  for up to MAX_AGENT_ITERATIONS=3 rounds:
      SEARCH: hybrid_search() for each query in play this round
        - vector search (pgvector, cosine, HNSW) over each unit's
          retrieval_text embedding -- catches paraphrase/alias cues
        - full-text search (GIN) over searchable_text -- catches literal
          keyword overlap (including literal alias/definition text)
        - fuse: 0.65*vector + 0.35*fts, keep candidates clearing a basic
          relevance bar
      (on the last round, skip straight to answering with what's found)
      EVALUATE: ask the LLM {sufficient: bool, reformulated_queries: [..]}
        given the question and best snippets so far -- it may propose up
        to 2 alternative phrasings per round, using its general knowledge
        of likely synonyms/aliases *only* to decide what to search next,
        never as answer content (spec section 17)
      if sufficient or no new queries proposed: stop

  if nothing found at all: ABSTAIN immediately, no LLM call needed
  else:
      expand with sibling_context(): up to 2 extra units per primary hit
      from the same capture_id, bounded to 6 total -- context to help
      interpret a primary hit (e.g. resolve a pronoun), never a
      substitute for it, never merged into one giant blob (spec section 20)
      ANSWER: generate_answer() -- LLM answers using primary + context
      evidence, must mark grounded=true/false
      if not grounded: ABSTAIN (fixed message, empty sources) -- this
      override lives in exactly one place, run_agent(), regardless of
      what text the model wrote
```

## Failure model

Three independent layers of tolerance, from most to least routine:

1. **LLM down/timeout/malformed output** during enrichment, embedding, or
   answer generation: caught by `LLMError` handling at the point of the
   call, degrades gracefully (empty semantics / null embedding / fall
   back to full-text-only search / abstain), never raises further.
2. **An actually unexpected exception** during the derived-processing
   stage (a real bug, not an LLM hiccup): caught by a broad `except` in
   `_enrich_and_index()`, sets `status="failed"`. The unit's
   `authoritative_content` (committed in an earlier, separate
   transaction) is untouched. `POST /memory-units/{id}/retry` re-runs
   just the derived stage.
3. **Invalid input** (empty text, unsupported file type): raised as
   `IngestionError` before anything is persisted, mapped to HTTP 400.

A unit is only ever searchable (`status="ready"`) after its derived
stage actually ran to completion -- a unit stuck in `"draft"` (crash
before phase 1 commit's units finished enriching) or `"failed"` never
appears in `hybrid_search()` results (both are filtered out by the
`status = 'ready'` clause in the SQL).

## Memory unit schema

```
captures
  id, raw_text, created_at

memory_units
  id, capture_id (fk, cascade delete)
  authoritative_content          text        -- user-originated
  clarifications                 jsonb       -- [{question, answer, round, created_at}]
  status                         text        -- draft | ready | failed
  pending_clarification          jsonb null  -- {question, field, round}
  clarification_round            int
  semantic                       jsonb       -- meaning, aliases, people,
                                                 organizations, locations,
                                                 topics, definitions,
                                                 relations, event_date,
                                                 enrichment_status
  retrieval_text                 text        -- original + semantic, for
                                                 embedding + fts
  searchable_text                text        -- + attachment text, feeds `fts`
  fts                             tsvector    -- GENERATED, GIN index
  embedding                       vector(768) null, HNSW/cosine index
  embedding_model, embedding_dim  text, int null
  created_at, updated_at

attachments
  id, capture_id (fk, cascade delete)
  original_filename, stored_path, content_type, size_bytes
  extracted_text null, extraction_status, created_at
```

## LLM integration

`app/llm/base.py` defines `LLMProvider` (`generate`, `generate_structured`,
`embed`) and three exception types (`LLMUnavailableError`,
`LLMTimeoutError`, `LLMMalformedOutputError`). `app/llm/ollama_provider.py`
is the only implementation. Every structured-output caller in the app
(segmentation, enrichment, agent evaluation, answer generation) goes
through the same `generate_structured()` method with a JSON-schema
`format` -- there's no separate prompt-chaining framework, just four
distinct, independently testable call sites.

## File handling, persistence, security

Unchanged from the original MVP: real file type sniffed from bytes before
anything is written to disk; all writable state (Postgres data, uploaded
files) lives under `.data/`, gitignored, never in the Nix store; the web
server binds `127.0.0.1` only.
