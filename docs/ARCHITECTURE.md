# Architecture

## Components

```
Browser (Jinja2-rendered HTML + vanilla JS)
        |  fetch() -- JSON over HTTP, same origin
        v
FastAPI app (backend/app)
  ├── api/captures.py   -- POST/GET/DELETE /api/captures/{id}, .../attachments
  ├── api/units.py      -- GET/clarify/correct/discard/retry on /api/memory-units/{id}
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

If `enrich()` surfaced a `clarification_question`, the unit's status
becomes `"awaiting_clarification"` instead of `"ready"` -- **gated**: it
is durably saved (survives a restart) but invisible to recall until
answered, because `hybrid_search()` only ever considers
`status='ready'` rows. Answering it (`POST /memory-units/{id}/clarify`)
appends the verbatim Q&A to `clarifications`, then re-runs the *entire*
derived stage (enrichment, retrieval_text, searchable_text, embedding)
with the new clarification as additional context -- this is what makes a
clarified term's aliases actually searchable afterward, not just present
in a one-off answer -- and flips status to `"ready"` once no further
question is open. The user may also explicitly cancel an awaiting unit
(`POST .../discard`, status becomes `"discarded"`) rather than answer it.

Clarification is bounded to `MAX_CLARIFICATION_ROUNDS = 1` per unit
(`app/ingestion/pipeline.py`) -- after that, further clarification
questions the model proposes are discarded rather than surfaced. A
user-initiated correction (`POST /memory-units/{id}/correct`, spec
section 30 -- free-text, re-runs the same enrichment mechanism) does
*not* consume this budget: the bound exists to stop the system from
interrogating the user, not to limit how often the user can proactively
fix something.

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
      SEARCH: for each query in play this round, BOTH:
        - hybrid_search() over memory_units (status='ready' only):
          vector (pgvector, cosine, HNSW) over retrieval_text embeddings
          -- catches paraphrase/alias cues -- fused with full-text (GIN)
          over searchable_text -- catches literal keyword overlap
        - hybrid_search_chunks() over attachment_chunks: same fusion,
          surfaces a specific excerpt from a large attached document
          rather than the whole file
        fuse weights: 0.65*vector + 0.35*fts, keep candidates clearing a
        basic relevance bar (VECTOR_SIM_THRESHOLD, see docs/DECISIONS.md
        for the calibration experiment behind the current value)
      (on the last round, skip straight to answering with what's found)
      EVALUATE: ask the LLM {sufficient: bool, reformulated_queries: [..]}
        given the question and best snippets so far -- it may propose up
        to 2 alternative phrasings per round, using its general knowledge
        of likely synonyms/aliases *only* to decide what to search next,
        never as answer content (spec section 17)
      if sufficient or no new queries proposed: stop

  if nothing found at all (neither units nor chunks): ABSTAIN immediately,
  no LLM call needed
  else:
      expand with sibling_context(): up to 2 extra units per primary hit
      from the same capture_id, bounded to 6 total -- context to help
      interpret a primary hit (e.g. resolve a pronoun), never a
      substitute for it, never merged into one giant blob (spec section 20)
      ANSWER: generate_answer() -- LLM answers using primary units +
      sibling context + attachment-chunk evidence, must mark
      grounded=true/false
      if not grounded: ABSTAIN (fixed message, empty sources) -- this
      override lives in exactly one place, run_agent(), regardless of
      what text the model wrote
```

## Failure model

Four independent layers of tolerance, from most to least routine:

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
3. **An open clarification question**: `status="awaiting_clarification"`
   -- not a failure at all, but deliberately excluded from recall until
   resolved (see "clarification" above).
4. **Invalid input** (empty text, unsupported file type): raised as
   `IngestionError` before anything is persisted, mapped to HTTP 400.

A unit is only ever searchable (`status="ready"`) after its derived
stage ran to completion with no open question -- `"draft"` (crash before
phase 1 finished), `"awaiting_clarification"`, `"failed"`, and
`"discarded"` (user-cancelled) all never appear in `hybrid_search()`
results, filtered out by its `status = 'ready'` SQL clause.

## Memory unit schema

```
captures
  id, raw_text, created_at

memory_units
  id, capture_id (fk, cascade delete)
  authoritative_content          text        -- user-originated
  clarifications                 jsonb       -- [{question, answer, round,
                                                 created_at, source?}];
                                                 source="user_correction"
                                                 for a correct_unit() entry
  status                         text        -- draft | awaiting_clarification
                                                 | ready | failed | discarded
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

attachment_chunks
  id, attachment_id (fk, cascade delete), capture_id (fk, denormalized)
  chunk_index, page_number null              -- real page for PDFs, else null
  text                            text        -- one deterministic chunk
                                                 (app/ingestion/chunking.py:
                                                 per-page for PDFs, fixed
                                                 character window + overlap
                                                 otherwise)
  fts                              tsvector    -- GENERATED, GIN index
  embedding                        vector(768) null, HNSW/cosine index
  embedding_model, embedding_dim   text, int null
  created_at
```

Deleting a capture (`DELETE /api/captures/{id}`) cascades through
`memory_units`, `attachments`, and `attachment_chunks` at the DB level,
and also removes the capture's upload directory from disk -- the only
way a memory ever goes away; there is no automatic/relevance-based
deletion anywhere in this codebase.

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
