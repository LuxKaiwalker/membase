# Architecture

## Components

```
Browser (Jinja2-rendered HTML + vanilla JS)
        |  fetch() -- JSON over HTTP, same origin
        v
FastAPI app (backend/app)
  ├── api/captures.py   -- POST /api/captures, GET /api/captures (list),
  |                        GET/DELETE /api/captures/{id}, .../attachments
  ├── api/units.py      -- GET/clarify/correct/discard/retry on /api/memory-units/{id}
  ├── api/recall.py     -- POST /api/recall (legacy one-shot);
  |                        POST/GET /api/recall/sessions[/{id}],
  |                        POST .../messages, POST .../cancel
  ├── web/routes.py     -- GET / (Remember), GET /recall (Recall),
  |                        GET /captures (Browse) -- serves the HTML shell
  ├── ingestion/        -- segmentation, enrichment, pipeline orchestration
  ├── retrieval/        -- hybrid search, sibling expansion, answer
  |                        generation, one-shot agent, RecallSession
  |                        controller, cue extraction, ambiguity heuristics
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

## Design principles

A handful of ideas recur across both ingestion and retrieval; worth
naming once instead of re-deriving from each section below:

- **Never let derived data outrank what the user actually said.** The
  authority hierarchy above isn't just a schema note -- it's the reason
  clarification *answers* get folded back into the searchable
  representation (so a clarified alias is actually findable) while
  clarification *questions* and retrieval-time *hypotheses* never are
  (they're reasoning aids, not facts).
- **Decompose before you index.** A single capture may describe several
  unrelated things; cramming them into one embedding blurs every
  retrieval target it touches. Segmentation (below) exists specifically
  so "my AHV number is X, Anna studies robotics, and I need to renew my
  passport" doesn't become one noisy vector.
- **Resolve ambiguity before committing to a representation, not after.**
  Both the storage side (clarification gates full enrichment) and the
  retrieval side (RecallSession can ask a discriminating question before
  answering) follow the same shape: when the system isn't confident what
  something means or which candidate is right, it asks rather than
  guesses -- and in storage's case, it doesn't even do the expensive
  semantic-extraction work until the ambiguity is settled, because that
  work would just be redone once it is.
- **Abstain rather than fabricate.** Every answer-generation path has
  exactly one grounding-override point that can force an abstention
  regardless of what text the model produced. This override is
  deliberately centralized (not re-implemented per call site) so it
  can't be silently weakened by a future change to one path and not
  the other.
- **Bound every LLM-driven loop, explicitly and for a stated reason.**
  Segmentation caps unit count (`MAX_SEGMENT_UNITS`); storage-time
  clarification caps rounds (`MAX_CLARIFICATION_ROUNDS` -- there's no
  questions-per-round bound any more since clarification is strictly
  sequential, one question at a time, not a batch); the one-shot
  agent caps search iterations (`MAX_AGENT_ITERATIONS`); RecallSession
  caps rounds (`MAX_SESSION_ROUNDS`). None of these are the same kind of
  bound -- see each section for what specifically each one prevents --
  but the pattern itself (never let an LLM-driven loop run unbounded) is
  deliberate throughout, not incidental to any one of them.

## Data flow: ingestion

**Streamlining-phase rewrite** (see docs/DECISIONS.md "Streamlining
phase"): `app/ingestion/clarify.py`, `enrichment.py`, and
`segmentation.py` are deleted. Everything they did -- deciding
segmentation, deciding whether a question is needed, and producing the
full semantic representation -- is now one call,
`analyze_capture()` (`app/ingestion/analysis.py`), per capture. A
clarification round costs exactly one further call,
`reanalyze_unit()`, which both re-derives semantics and decides whether
a *further* question is still warranted. There is no pre-generated
clarification queue any more -- clarification is strictly sequential,
one question at a time.

```
capture text + files
      |
      ├─ process each file (app/ingestion/files.py): sniff real mime type,
      |   reject unsupported, save original, best-effort text extraction
      |   (images are stored but not captioned here -- no LLM call before
      |   phase 0 commits, same rule as everything else derived)
      |
      ├─ COMMIT (phase 0): capture + attachments durably saved
      |
      ├─ _find_related_context() (app/ingestion/pipeline.py): cheap,
      |   non-LLM lexical lookup (regex-extracted proper-noun-shaped
      |   terms + SQL ILIKE against ready units' retrieval_text) --
      |   up to 3 short snippets, injected as context-only (never
      |   authority) into the analysis call below.
      |
      ├─ analyze_capture() (app/ingestion/analysis.py): ONE structured-
      |   output call. Decides segmentation (each unit's content must be
      |   a verbatim excerpt of raw_text, or the whole text is kept as
      |   one unit -- safety fallback, see "Segmentation" below) AND,
      |   for each resulting unit, its full semantic representation
      |   (meaning/people/organizations/locations/aliases/topics/
      |   definitions/relations/event_date) AND whether it needs a
      |   clarifying question -- all in one pass. Takes attachment_text
      |   (non-image attachments only -- see the image-captioning note
      |   below) and the prior-context snippets above as prompt context.
      |
      ├─ explicit metadata applied (app/ingestion/pipeline.py
      |   `_apply_explicit_metadata`): a user-supplied location/date/
      |   people/organizations/topics field always overrides the same
      |   inferred field; a genuine conflict becomes a synthetic
      |   clarification question (see docs/DECISIONS.md "Explicit
      |   metadata authority").
      |
      ├─ COMMIT (phase 1): every unit's authoritative_content + full
      |   analysis durably saved, status="draft" (or
      |   "awaiting_clarification" if a question -- model-proposed or a
      |   metadata conflict -- is pending)
      |
      └─ for each unit with no pending question: _claim_processing()
          (app/ingestion/pipeline.py) sets status="processing" + a fresh
          processing_token, COMMIT -- the HTTP response returns here.
          Finalization (below) runs as a scheduled FastAPI BackgroundTask
          (finish_unit_processing()), after the request/response cycle.

api/captures.py also schedules, per attachment: finish_image_captioning()
for an uncaptioned image, or finish_attachment_chunking() (below) if the
attachment's extracted text clears the chunking size threshold --
both fully independent of each other and of the synchronous path above.
```

```
finish_image_captioning() (background task, own DB session)
      |
      └─ the generation model's native multimodal understanding
          describes the image (best-effort; a failure leaves the image
          attached but uncaptioned -- never blocks or fails the
          capture, see docs/DECISIONS.md "Decouple image captioning").
          Explicit trade-off: analyze_capture() above already ran
          without this caption as context, since captioning is no
          longer in the synchronous path. If the caption clears the
          chunking threshold it's chunked (below); otherwise it's
          folded as a short excerpt into every already-"ready" sibling
          unit's retrieval_text and re-embedded (the same light path
          add_attachment() uses for a posthoc file), so it still
          becomes searchable either way.
```

```
finish_attachment_chunking() (background task, own DB session)
      |
      └─ should_chunk(): skipped entirely if the attachment's extracted
          text is under membase_min_chars_for_chunking (2000 chars) --
          the parent unit's own full-text search already covers a
          document that short (see docs/DECISIONS.md "Lazy attachment
          chunking"). Otherwise, _index_attachment_chunks() chunks the
          text (app/ingestion/chunking.py, capped at
          membase_max_chunks_per_attachment) and best-effort embeds each
          chunk into attachment_chunks. Idempotent (clears existing
          chunks first), which is what makes retry_attachment_chunking()
          safe to call again after a dropped background task -- no
          separate pending/processing status needed for chunks.
```

```
finish_unit_processing() (background task, own DB session + processing lease)
      |
      ├─ build_retrieval_text(): deterministic (non-LLM) text combining
      |   original + clarifications + meaning + aliases + definitions +
      |   relations -- pure Python, using semantic fields the analysis
      |   call already produced. No LLM call here any more.
      ├─ searchable_text = authoritative_content + attachment text
      |   + retrieval_text  (feeds the GIN full-text index)
      ├─ embed(retrieval_text) -> pgvector column (best-effort; failure
      |   leaves embedding NULL, unit still full-text searchable) --
      |   this is the ONLY LLM-adjacent call left in this background step.
      ├─ _commit_if_token_matches(): status="ready" (or "failed" on a
      |   genuinely unexpected exception), committed only if this call's
      |   processing_token still matches the row's -- a concurrent retry
      |   that claimed the unit first already landed its own result, so
      |   this one is simply dropped rather than stomping it (see
      |   "Processing lease" below).
      └─ _detect_conflicts(): reuses hybrid_search() to find other ready
          units whose relations share a (subject, predicate) but
          disagree on the object; marks both units' potential_conflicts
          (advisory only -- see docs/DECISIONS.md "Contradiction
          detection").
```

**Processing lease, not a held row lock.** The previous phase held
`SELECT ... FOR UPDATE` for the entire duration of
`finish_unit_processing()`, including the LLM calls. This phase
replaces that with a lease: `_claim_processing()` sets
`status="processing"` plus a fresh random `processing_token` and
`processing_started_at` in a short transaction and commits immediately
-- no lock is held while the embedding call runs. The finalization step
commits its result only if `processing_token` still matches
(`_commit_if_token_matches()`, a single `UPDATE ... WHERE id=? AND
processing_token=?` checked via `rowcount`); a mismatch means a
concurrent attempt already finished first, so this one's result is
dropped rather than applied on top of it. `POST /memory-units/{id}/retry`
only reclaims a unit stuck in `"processing"` if `processing_started_at`
is older than `membase_processing_lease_stale_seconds` (300s) -- a fresh
lease is left alone, since a live background task is presumably still
going to finish it.

This two-phase-commit structure is what guarantees "authoritative content
persists even if derived processing fails" (spec section 25): the
authoritative text is committed *before* any LLM call in the derived
stage runs, in a separate transaction.

If `analyze_capture()` (or a metadata conflict) surfaces a question, the
unit's status becomes `"awaiting_clarification"` instead of `"ready"` --
**gated**: it is durably saved (survives a restart) but invisible to
recall until answered, because `hybrid_search()` only ever considers
`status='ready'` rows. Answering it (`POST /memory-units/{id}/clarify`)
classifies the answer deterministically as answered/unknown/declined (no
LLM call -- `_classify_outcome`), appends the verbatim Q&A to
`clarifications`, then -- if the round budget
(`MAX_CLARIFICATION_ROUNDS`) allows -- calls `reanalyze_unit()` exactly
once to re-derive semantics AND decide whether a further question is
still warranted; a field already marked declined/unknown is hard-coded
to never be re-asked regardless of what the model proposes. Once no
further question is pending (or the round budget is exhausted), the unit
moves to `status="processing"` and background finalization (above) runs
-- this is what makes a clarified term's aliases actually searchable
afterward, not just present in a one-off answer.

Answering is optional, enforced only as a client-side preference (the
settings menu's "obligatory" toggle, `app/static/app.js`), never by the
API: `ClarifyRequest.answer` is `str | None`, and a blank/omitted answer
skips the question -- nothing is appended to `clarifications`, the unit
just proceeds through `_advance_unit()` exactly as if the question had
never been asked. The user may also explicitly cancel an awaiting unit
(`POST .../discard`, status becomes `"discarded"`) rather than answer
or skip it.

Clarification is bounded on two axes (`app/ingestion/pipeline.py`):
`MAX_QUESTIONS_PER_ROUND = 3` candidate questions considered per check
(asked one at a time, never as a batch), and `MAX_CLARIFICATION_ROUNDS =
3` rounds of checking -- a "round" only counts if it actually surfaced a
question, so an ordinary unit that never needed clarification stays at
round 0 forever. After the round budget is exhausted, further questions
the model proposes are discarded and enrichment proceeds regardless. A
user-initiated correction (`POST /memory-units/{id}/correct`, spec
section 30 -- free-text, re-runs enrichment directly, bypassing the
clarification check entirely) does *not* consume this budget: the bound
exists to stop the system from interrogating the user, not to limit how
often the user can proactively fix something.

## Segmentation

Segmentation is the first half of `analyze_capture()`'s single call
(`app/ingestion/analysis.py`), asking the LLM to split `raw_text` into
units in the same pass as deciding each unit's semantics and
clarification need. Two hard safety rules, enforced in code (not just
prompted), unchanged by the streamlining-phase call collapse:

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
2. **Cap.** At most `MAX_SEGMENT_UNITS = 24` units per capture, to bound a
   pathological over-segmentation (raised from 8 once a long/dense
   capture became a correctly-handled case via chunking rather than
   something to defend against -- see docs/DECISIONS.md "Chunk long
   captures before analysis").

## The membase structurer

A second, independent background pass that deepens an already-`"ready"`
(searchable) unit's representation, triggered automatically the moment
a unit first reaches `"ready"` -- not a feature the user invokes, and
not something that can make a unit *less* available. See
docs/DECISIONS.md "The 'membase structurer'" for the full design
rationale; this section is the current-state summary.

```
unit reaches status="ready" (fast pipeline finishes)
      |
      ├─ api layer schedules finish_structuring(unit.id) as a
      |   BackgroundTask, registered right after finish_unit_processing
      |   for the same unit -- Starlette runs BackgroundTasks
      |   sequentially in registration order, so this is guaranteed to
      |   run after the unit is already "ready", with no polling
      |
      finish_structuring() (own DB session)
      |
      ├─ guard: only proceeds if status == "ready" AND
      |   structure_status == "unstructured" -- anything else (not yet
      |   ready, already structured, already in flight, previously
      |   failed) is a no-op here; a "failed" unit is only retried via
      |   the explicit POST /memory-units/{id}/restructure endpoint
      |
      ├─ _claim_structuring(): structure_status="structuring" + a fresh
      |   structure_token + structure_started_at, committed immediately
      |   -- the SAME lease pattern as processing_token/
      |   processing_started_at, just a second, independent lease so
      |   the two can never collide
      |
      ├─ find candidates: hybrid_search() (the existing retrieval
      |   primitive, no new one) using an OR-joined query built from the
      |   unit's own topics/people/organizations/locations/aliases --
      |   OR, not AND, so a candidate only needs to share SOME of what
      |   this unit is about, not all of it (see docs/DECISIONS.md for
      |   the bug this fixes)
      |
      ├─ structure_unit() (app/ingestion/structurer.py): one deliberately
      |   unhurried LLM call -- re-derives the full Semantics shape
      |   (meaning/people/organizations/locations/aliases/topics/
      |   definitions/relations/event_date) more thoroughly than the
      |   fast pipeline's single combined call, and decides which of
      |   the candidates are genuinely related
      |
      └─ _commit_structuring_if_token_matches(): atomically swaps
          semantic/retrieval_text/searchable_text/embedding/
          related_unit_ids in place, and sets
          structure_status="structured" + structured_at -- gated on the
          lease token, so a stale/duplicate attempt can never stomp a
          result that already landed. A failure anywhere above commits
          only structure_status="failed", leaving every other field
          (the fast pipeline's original output) completely untouched --
          structuring only ever upgrades, never degrades.
```

**The one invariant this whole feature is built around:** `status`
(the only searchability gate `hybrid_search()` checks) is never read or
written by any of the above. `structure_status` is a second, entirely
separate piece of state -- a unit sitting at `structure_status=
"unstructured"` indefinitely (structuring never ran, or is still
queued behind other work) is exactly as searchable as one at
`"structured"`. `hybrid_search()` itself required zero changes for this
feature to exist.

**Recovery:** `POST /memory-units/{id}/restructure` reclaims a unit
stuck in `structure_status="structuring"` past
`membase_processing_lease_stale_seconds`, or retries one that
previously `"failed"`. Runs synchronously (unlike the fast pipeline's
`/retry`, which reschedules a background task), since this is a
deliberately infrequent manual action, not part of the normal flow.

**Visualized only on the Browse page** (`structure-badge-*` in
`app/static/app.css`, read at page-load time, no polling) -- the Store
page shows "Saved" as soon as clarification is resolved and never waits
on or displays structuring status at all.

**Deferred:** structurer-driven re-segmentation (splitting an existing
ready unit into more units if the deeper pass judges the original
boundary wrong). See docs/DECISIONS.md and docs/TODO.md for why --
short version: it runs into the never-rewrite-`authoritative_content`
invariant and would need a "supersede with new siblings, retire the
old unit" lifecycle that doesn't exist yet.

## Data flow: retrieval (the agent)

```
query
  |
  v
app/retrieval/agent.py: run_agent()  -- bounded deterministic state machine
(spec section 29; NOT a general tool-calling framework)

  for up to MAX_AGENT_ITERATIONS=2 rounds (reduced from 3 -- see
  docs/DECISIONS.md "Reduce automatic retrieval iterations"):
      SEARCH: for each query in play this round, embed it ONCE
        (embed_query(), see docs/DECISIONS.md "Dedupe query embeddings")
        and reuse that vector for BOTH:
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
      SHORTCUT: if the top candidate is overwhelmingly dominant (fused
        score + margin over the runner-up both clear a bar) AND its
        semantic analysis isn't a degraded fallback, skip EVALUATE
        entirely and stop (see docs/DECISIONS.md "Deterministic
        sufficiency short-circuit") -- the common, unambiguous case
        never pays for the reasoning call below.
      EVALUATE: ask the LLM {sufficient: bool, reformulated_queries: [..]}
        given the question and best snippets so far -- it may propose up
        to 1 alternative phrasing per round (reduced from 2), using its
        general knowledge of likely synonyms/aliases *only* to decide
        what to search next, never as answer content (spec section 17)
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
      if not grounded: ABSTAIN (fixed message, empty sources) -- the
      underlying grounded true/false decision is computed in exactly
      one place, generate_answer(); what each caller *does* with
      grounded=false differs (run_agent() abstains immediately,
      RecallSession's _run_round() may ask a discriminating question
      first -- see below), but neither caller ever shows an answer the
      model itself marked ungrounded
```

## Iterative recall (RecallSession) -- now canonical end to end

**Status, updated this phase (see docs/DECISIONS.md "RecallSession
canonical"):** `RecallSession` is no longer a parallel, optional path --
`POST /api/recall` (the "one-shot" endpoint) now internally calls
`start_session()` and runs exactly the same first automatic round a
multi-turn session would (same ambiguity narrowing, same grounding
decision), reporting the outcome in the flat one-shot response shape.
There are no longer two independent retrieval algorithms. What's
*still* one-shot-only is the frontend: `app/static/app.js`'s
`initRecallPage()` only calls `POST /api/recall` and has no UI for a
session id, cues, or a follow-up question -- a prior benchmark found
the fully interactive UI underperforming a one-shot answer on a
deliberately ambiguous synthetic corpus (see docs/DECISIONS.md), and
that UI decision hasn't been revisited. `run_agent()` (below, now an
internal primitive) is kept only for callers that explicitly want a
one-shot search-and-answer with no persisted session row and no
ambiguity narrowing -- the eval harness and latency benchmarks.

`app/retrieval/session.py`'s `RecallSession` builds the same bounded
search primitives into a multi-round process, because a user's first
recall attempt is often just a weak or partial cue, not a wording close
enough for one `hybrid_search` call to land on. This is the
architectural response to that: retrieval accumulates evidence and
user-provided cues across rounds instead of treating every query as
isolated -- and, as of this phase, every recall request goes through
at least this machinery's first round, one-shot or not.

```
POST /api/recall/sessions {query}
  |
  v
start_session() -- persists a RecallSession row, then runs one round
  |
  v
_run_round():
  automatic_search(original_query, extra_queries=<all positive cues so far>)
    -- the SAME bounded 2-iteration search/evaluate/reformulate loop
       run_agent() uses (extracted as automatic_search(), shared by
       both). The original query is *always* re-searched every round
       (never replaced by a reformulation or a cue) -- the one thing
       standing between this and the query-drift failure mode (spec
       section 23).
  |
  v
apply_negative_cues()        -- exclude a candidate the user explicitly
                                 ruled out ("no, it wasn't Anna") by a
                                 real metadata match, not a loose
                                 substring guess
  |
  v
apply_positive_cue_boost()   -- reward a candidate whose own semantic
                                 metadata (people/orgs/locations/
                                 aliases/topics) matches a cue the user
                                 just confirmed. Necessary in practice:
                                 re-searching with a bare one-or-two-word
                                 cue as its own query often scores
                                 *lower* on the fused metric than the
                                 original full-sentence query already
                                 did for every candidate (short queries
                                 embed/rank differently), so merging via
                                 max-across-queries alone never resolved
                                 anything in testing -- see
                                 docs/DECISIONS.md.
  |
  v
is_ambiguous()?  -- two-or-more distinct origin-capture candidates
                    (not sibling units of the same episode) whose top
                    scores are within AMBIGUITY_SCORE_MARGIN (0.15)
  |
  +-- yes --> generate_discriminating_question()
  |             picks the semantic attribute (people/locations/orgs/
  |             topics/event_date) that best bisects the remaining
  |             candidates, phrased as one concrete question ("Did
  |             this happen at ETH?"), skipping any question already
  |             asked this session
  |             |
  |             +-- found one  --> status=NEEDS_USER_CUE, ask it, STOP
  |             +-- none left  --> fall through to answer anyway
  |
  +-- no  ---> narrow_to_dominant_group() -- drop the losing group
               entirely from evidence (spec section 12 governs the
               *ambiguous* case; once there's a clear winner there's no
               decision left for answer-generation to make between
               groups)
  |
  v
generate_answer() (same grounding-enforcement function run_agent() uses)
  |
  +-- grounded   --> status=SUCCESS, snapshot sources, STOP
  +-- ungrounded --> try one more discriminating question (same as
                      above); if none, status=EXHAUSTED, STOP
```

`continue_session()` is what a user reply (an answer to the question,
an unprompted extra recollection, or a correction) turns into: `app/
retrieval/cues.py:extract_cues()` is a narrow structured-output call
that pulls out `{text, confidence: certain|likely|uncertain, polarity:
positive|negative}` cues from the free-text message (explicitly
instructed to extract *every* concrete detail even if the message also
answers yes/no -- a reply like "Yes, it was at ETH" must not lose the
new "ETH" detail just because it also confirmed the question), appends
them to the session, and runs another round. A **positive** cue, at any
confidence level, is added to the search queries for every subsequent
round -- never a hard filter, since an uncertain guess must not
silently eliminate a candidate (spec section 39). A **negative** cue is
never searched for; it only feeds `apply_negative_cues`.

A session ends in exactly one of: `success` (grounded answer),
`exhausted` (nothing found, or ambiguity never resolved and no further
discriminating angle exists -- `MAX_SESSION_ROUNDS=12` is a runaway-loop
safeguard, not a UX-facing turn limit; see spec section 10), `cancelled`
(explicit `POST .../cancel`), or `error` (an LLM/infra failure mid-round).
`needs_user_cue` and `active` are the only non-terminal statuses;
`continue_session()`/`cancel_session()` raise `RecallSessionError` on
any terminal session.

**Memory-contamination safeguard** (spec section 5/29, the reason this
needed its own data model rather than reusing MemoryUnit): `cues`,
`hypotheses` (the reformulated queries `evaluate_sufficiency` proposes,
kept on the session purely for transparency/testability -- spec section
18), and `messages` all live *only* on the `RecallSession` row. Nothing
in `app/retrieval/session.py` or `app/retrieval/cues.py` ever calls
`create_capture` or writes to `MemoryUnit` -- the only way something
discussed during recall becomes a durable memory is the user separately
using the ordinary capture-creation flow. `final_sources`/
`final_chunk_sources` are a snapshot (id + content + scores as of answer
time), not a live reference, so a later correction or deletion of the
underlying memory can't retroactively corrupt a past session's record of
what it answered from.

**Persistence**: one row per session in Postgres (the app's only state
store already), not in-memory server state -- an in-progress multi-turn
recall survives an application restart the same way everything else
does (verified: start a session, restart the dev server, `GET
/api/recall/sessions/{id}` returns identical state, `POST .../messages`
continues it to completion).

## Failure model

Four independent layers of tolerance, from most to least routine:

1. **LLM down/timeout/malformed output** during clarification-question
   generation, enrichment, embedding, or answer generation: caught by
   `LLMError` handling at the point of the call, degrades gracefully
   (no question raised / empty semantics / null embedding / fall back to
   full-text-only search / abstain), never raises further.
2. **An actually unexpected exception** during the derived-processing
   stage (a real bug, not an LLM hiccup): caught by a broad `except` in
   `_advance_unit()` (the clarification-check step) or
   `_run_full_enrichment()` (the enrichment step) in
   `app/ingestion/pipeline.py`, sets `status="failed"`. The unit's
   `authoritative_content` (committed in an earlier, separate
   transaction) is untouched. `POST /memory-units/{id}/retry` calls
   `_advance_unit()` again, resuming from wherever it got stuck -- it
   doesn't blindly restart from enrichment, since the unit might have
   failed mid-clarification-check instead. `retry` is also the recovery
   path for a unit stuck in `"processing"` because a server restart
   dropped its `BackgroundTask`; it re-acquires the same row lock
   `finish_unit_processing()` holds and re-checks status afterward, so
   racing a genuinely-still-running enrichment pass can't stomp its
   result (see "Data flow: ingestion" above).
3. **An open clarification question**: `status="awaiting_clarification"`
   -- not a failure at all, but deliberately excluded from recall until
   resolved (see "clarification" above).
4. **Invalid input** (empty text, unsupported file type): raised as
   `IngestionError` before anything is persisted, mapped to HTTP 400.

A unit is only ever searchable (`status="ready"`) after its derived
stage ran to completion with no open question -- `"draft"` (crash before
phase 1 finished), `"awaiting_clarification"`, `"processing"` (background
enrichment scheduled but not yet finished), `"failed"`, and
`"discarded"` (user-cancelled) all never appear in `hybrid_search()`
results, filtered out by its `status = 'ready'` SQL clause. The same
gate applies to `attachment_chunks` via `hybrid_search_chunks()`, even
though chunks have no status column of their own -- they're indexed as
soon as an attachment is uploaded, before segmentation or clarification
even run, so an EXISTS check requiring at least one `status='ready'`
sibling `MemoryUnit` in the same capture closes that gap at the
granularity chunks actually have.

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
                                                 | processing | ready | failed
                                                 | discarded
  pending_clarification          jsonb null  -- {question, field, round}
  clarification_round            int         -- no clarification_queue any
                                                 more (streamlining phase):
                                                 clarification is strictly
                                                 sequential, one question
                                                 at a time, never a
                                                 pre-generated batch
  processing_token                text null  -- processing-lease token,
                                                 see "Processing lease"
                                                 above
  processing_started_at           timestamptz null
  explicit_metadata               jsonb       -- user-supplied location/
                                                 date/people/organizations/
                                                 topics; always wins over
                                                 the same field below
  potential_conflicts             jsonb       -- [{unit_id, note}, ...]
                                                 advisory contradiction
                                                 markers, see
                                                 docs/DECISIONS.md
  structure_status                 text       -- unstructured | structuring
                                                 | structured | failed --
                                                 the membase structurer's
                                                 own progress; NEVER read
                                                 by hybrid_search(), never
                                                 gates searchability (see
                                                 "The membase structurer"
                                                 above)
  structure_token                  text null  -- structuring-lease token,
                                                 same pattern as
                                                 processing_token
  structure_started_at             timestamptz null
  structured_at                    timestamptz null
  related_unit_ids                 jsonb       -- [unit_id, ...] advisory
                                                 cross-references the
                                                 structurer found, NOT a
                                                 knowledge graph
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

recall_sessions
  id, original_query
  status                          text       -- active | needs_user_cue |
                                                 success | exhausted |
                                                 cancelled | error
  current_question                text null
  messages                        jsonb      -- [{role, text, created_at}]
  cues                             jsonb      -- [{text, confidence, polarity,
                                                 round, created_at}]
  hypotheses                      jsonb      -- [{text, round, created_at}]
  tried_queries                   jsonb      -- [str, ...]
  asked_questions                 jsonb      -- [str, ...]
  round_count, last_candidate_count int
  final_answer                    text null
  final_sources, final_chunk_sources jsonb  -- evidence snapshots (id,
                                                 content, scores as of
                                                 answer time), not live refs
  created_at, updated_at
```

No foreign key from `recall_sessions` to `memory_units`/`attachment_chunks`:
`final_sources`/`final_chunk_sources` are self-contained snapshots, so a
session's record of what it once answered stays intact even if the
underlying memory is later corrected or deleted (see "Iterative recall"
above).

Deleting a capture (`DELETE /api/captures/{id}`) cascades through
`memory_units`, `attachments`, and `attachment_chunks` at the DB level,
and also removes the capture's upload directory from disk -- the only
way a memory ever goes away; there is no automatic/relevance-based
deletion anywhere in this codebase.

## LLM integration

`app/llm/base.py` defines `LLMProvider` (`generate`, `generate_structured`,
`embed`) and three exception types (`LLMUnavailableError`,
`LLMTimeoutError`, `LLMMalformedOutputError`). `app/llm/ollama_provider.py`
is the only implementation, with a configurable request timeout
(`Settings.llm_timeout_seconds`, default 90s -- see `app/config.py`).
Every structured-output caller in the app goes through the same
`generate_structured()` method with a JSON-schema `format` -- there's no
separate prompt-chaining framework, just independently testable call
sites, one per narrow decision:

| Call site | Module | Decides |
|---|---|---|
| `analyze_capture()` | `ingestion/analysis.py` | how many memory units this capture contains, each unit's full semantic representation, AND whether it needs a clarifying question -- all in one call (see docs/DECISIONS.md "Collapse ingestion LLM calls") |
| `reanalyze_unit()` | `ingestion/analysis.py` | the same semantic representation plus further-clarification-needed, re-derived once per clarification round |
| `evaluate_sufficiency()` | `retrieval/pipeline.py` | whether retrieved evidence is enough to answer, and what to search next -- skipped entirely when a deterministic high-confidence check already says yes (see docs/DECISIONS.md "Deterministic sufficiency short-circuit") |
| `generate_answer()` | `retrieval/pipeline.py` | the final answer text and whether it's grounded |
| `extract_cues()` | `retrieval/cues.py` | structured cues from a RecallSession user reply (RecallSession only) |

Streamlining-phase history: `enrich()` (`ingestion/enrichment.py`) used
to make three independent sub-calls (narrative/entities/retrieval-aids),
and a separate `generate_questions()` (`ingestion/clarify.py`) decided
clarification need before it ever ran -- five calls total per capture
in the worst case. `analyze_capture()` now does all of that in one call;
see docs/DECISIONS.md for why the consolidation was judged not to cost
retrieval quality the eval set could detect (two cases out of 30 did
regress -- documented honestly there, not hidden). No call site calls
back into another call site.

## File handling, persistence, security

Unchanged from the original MVP: real file type sniffed from bytes before
anything is written to disk; all writable state (Postgres data, uploaded
files) lives under `.data/`, gitignored, never in the Nix store; the web
server binds `127.0.0.1` only.

## Authentication

This app has no authentication code of its own -- see docs/DECISIONS.md
"Retire app-level auth for nginx Basic Auth". Access control is HTTP
Basic Auth enforced entirely by the nginx reverse proxy in front of the
backend (`deploy/nginx-membase.conf`); this app trusts whatever request
already reached it. The frontend's login overlay (`app/static/app.js`'s
`initAuth()`) collects a backend URL + username/password once, stores
them in `localStorage`, and attaches `Authorization: Basic
<base64(user:pass)>` to every API request itself -- there's no
`/api/auth/*` route, session, or token on the backend to manage. The
app's own contribution to this boundary is CORS only (`app/main.py`'s
`CORSMiddleware`, allow-listing `Settings.frontend_origin`), needed
because the frontend is published separately (GitHub Pages) from
wherever the backend is reached. See docs/DEPLOYMENT.md "nginx +
Cloudflare Tunnel" for the full request path from browser to backend.
