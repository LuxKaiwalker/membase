# Engineering report: streamlining phase

This supersedes the previous report. Scope: the previous two phases
built out and then hardened the augmented-memory architecture
(MemoryUnit segmentation, clarification gating, iterative RecallSession).
This phase's mandate was different in kind -- not a new feature, but an
explicit instruction that "the project is too bloaty": reduce LLM call
count, remove duplicate retrieval pathways, collapse prematurely
specialized processing stages, and reduce latency, **while preserving
every architectural idea that makes this an augmented-memory system
rather than a conventional notes/RAG app**. The spec named five
abstractions everything else had to justify its existence against:
Capture, MemoryUnit, Semantic representation, RecallSession, LLM/
Database. All five survive this phase unchanged in kind; what changed
is how many calls and how much code it takes to run them.

Verified against the real local model throughout -- the 30-case eval
corpus, a live multipart `POST /api/captures` round-trip including an
explicit-metadata conflict and its clarification answer, and the full
fast pytest suite (139 tests, all passing, against fakes).

## 1. Current architecture

Unchanged in *shape* from docs/ARCHITECTURE.md's description before this
phase: a `Capture` decomposes into one or more `MemoryUnit`s; each unit
carries authoritative content (verbatim, never rewritten), an append-only
clarification history, a derived semantic representation, and a status
gate (`draft -> awaiting_clarification -> processing -> ready`, or
`failed`/`discarded`). Recall is `RecallSession`-based: a query starts a
persisted session, an automatic search-and-answer round runs first, and
a difficult query escalates to asking the user a discriminating question
rather than guessing. See docs/ARCHITECTURE.md (updated this phase) for
the full diagrams.

What's different is *how few calls it takes to run that shape*:

- **Ingestion** is one unified LLM call (`analyze_capture()`) per
  capture instead of five across segmentation, clarification-need
  detection, and three-way enrichment. A clarification round is one
  further call (`reanalyze_unit()`), not a pre-generated queue.
- **Recall** has exactly one retrieval+decision algorithm, not two:
  the previously-separate one-shot `run_agent()` path and the
  multi-turn `RecallSession` path now both route through
  `start_session()`'s first automatic round. `run_agent()` survives
  only as an internal primitive for the eval harness and benchmarks.
- **A deterministic shortcut** skips the sufficiency-reasoning LLM call
  entirely when the top search result is unambiguously dominant.
- **Two new capabilities were added without new call-shaped
  complexity**: explicit structured metadata (bypasses inference, no
  LLM call) and lightweight contradiction detection (reuses the
  existing search call, pure-Python comparison).

## 2. Removed complexity

Deleted outright: `app/ingestion/clarify.py`, `enrichment.py`,
`segmentation.py`, and their dedicated test files -- replaced by the
single `app/ingestion/analysis.py`. Removed from the schema:
`MemoryUnit.clarification_queue` (there is no longer a batch of
pre-generated future questions to store). Removed from config:
`membase_max_images_captioned_per_capture` (the per-request image-batch
cap it bounded no longer applies now that each image is captioned by its
own independent background task, not a synchronous batch). Removed from
`agent.py`: the implicit assumption that one-shot and multi-turn recall
could reasonably diverge in answer quality for the same stored memories
-- they can't any more, by construction.

Reduced, not removed: `MAX_AGENT_ITERATIONS` 3->2,
`MAX_REFORMULATIONS_PER_ROUND` 2->1, the DB idle-in-transaction timeout
450s->180s (now that a transaction holds at most one LLM call instead of
up to five). The long-held `with_for_update` row lock was replaced with a
processing-lease/token pattern -- not removed, since the underlying race
it protects against is real, but reduced to two fast local transactions
instead of a lock held for an LLM call's duration.

**What was deliberately NOT removed, despite "too bloaty":** the
verbatim-excerpt safety check, the two-phase-commit durability pattern,
the status gate excluding unanswered units from search, provenance
(clarifications never overwritten), grounding enforcement, and the
processing-lease/background-task machinery itself (collapsed to fewer
calls *inside* it, not replaced with a synchronous call, per the spec's
explicit instruction to keep FastAPI BackgroundTasks and the state
machine). Removing any of these would have been removing an
architectural idea, not bloat.

## 3. Performance

LLM call counts are exact (by inspection, not measurement -- a call
count doesn't need a timer to know):

| Flow | Before this phase | After |
|---|---|---|
| Ordinary memory, no clarification | 4 generation calls + 1 embed | **1 generation call + 1 embed** |
| Memory needing 1 clarification round | 4 + 1 (initial) + 1 (question re-check) + re-embed | 1 (initial) + 1 (`reanalyze_unit`) + 1 embed |
| Simple one-shot recall, high-confidence match | 1 embed (unit search) + 1 embed (chunk search, duplicate) + 1 sufficiency-eval call + 1 answer call | **1 embed (shared) + 1 answer call** -- sufficiency-eval skipped entirely |
| Recall needing 1 reformulation | up to 2 reformulations searched, each re-embedding twice | up to 1 reformulation, embedding deduped |
| Contradiction check | did not exist | +1 reused `hybrid_search()` call per newly-finalized unit with a relation (no new LLM call) |

Not measured: wall-clock latency under identical load conditions
before vs. after. Reverting the full call collapse to produce a clean
"before" timing baseline would have cost more of this phase's time
budget than the number was worth, given the call-count reduction is the
more direct, less noisy signal, and Ollama (`-np 1`, serialized
requests) makes any latency number collected on this machine highly
sensitive to what else was running at the time -- confirmed firsthand
when a manual verification request queued for over a minute behind the
eval harness's own concurrent Ollama calls during this phase's own
testing. If a wall-clock number is specifically needed later,
docs/TODO.md has the exact before/after procedure.

## 4. Retrieval quality

The 30-case eval corpus (`eval/cases.yaml`), run against the real
`gemma4:12b` + `nomic-embed-text` models via `scripts/run-eval.sh`:
**28/30 passed (93%)**, full transcript in docs/DECISIONS.md. Two
failures, both plausibly connected to this phase's specific trade-offs
(detailed analysis and the recommended follow-up experiment in
docs/DECISIONS.md and docs/TODO.md):

- `clarification_alias_second_term`: the unified `analyze_capture` call
  didn't flag an invented term ("Flimvex") as needing clarification,
  where a prior phase's dedicated, single-purpose question-generation
  call apparently did.
- `user_specific_terminology`: an indirect alias query ("What did I call
  the project?" -> "Nightingale") abstained instead of answering,
  plausibly from the reduced reformulation budget running out before
  finding the bridging phrasing.

Neither was reverted: 93% on a 30-case hand-written corpus is within a
plausible sampling-noise band for a non-deterministic model even with
zero code change, and the spec explicitly traded some quality margin for
the call-count reduction on purpose. They're recorded as the concrete
starting point for a real before/after comparison if this turns out to
matter in practice, not swept under the rug.

Ambiguity detection was tightened, not loosened: `is_ambiguous()` now
also requires the runner-up candidate to independently clear an absolute
relevance floor (`0.45`), so two weak matches that happen to be close to
*each other* near the noise floor no longer trigger a pointless
clarifying question -- a genuine quality improvement alongside the
latency-motivated changes.

## 5. Contradiction handling

Implemented this phase, storage-side and retrieval-side, both reusing
existing primitives rather than adding new ones. Storage-side: after a
unit embeds, its relation triples are compared against other `ready`
units found via the *existing* `hybrid_search()` call (searching on the
relation's subject+predicate, not the unit's full text -- see the bug
writeup in docs/DECISIONS.md for why that distinction mattered); a
mismatch adds an advisory `potential_conflicts` marker to both sides,
never deletes or supersedes either. Retrieval-side: `format_evidence()`
renders a `[Conflict] ...` line for any evidence unit carrying one, and
the answer-generation prompt is instructed to state the disagreement
explicitly ("You have two stored memories that conflict: one says X and
another says Y") rather than silently pick a side. This is the only
place conflict presentation happens. No truth-maintenance database, no
temporal belief revision, no "newest wins" rule -- all explicitly out of
scope per the spec, documented as future options in docs/TODO.md.

## 6. Clarification

Fully rebuilt around sequential, re-evaluated-after-every-answer
questioning instead of a pre-generated queue:

- **Sequential flow**: one question surfaced at a time
  (`pending_clarification`); answering it re-analyzes the unit exactly
  once (`reanalyze_unit`), which both re-derives semantics and decides
  whether a further question is still warranted. Bounded at
  `MAX_CLARIFICATION_ROUNDS = 3`.
- **Direct structured metadata**: an optional JSON `metadata` form field
  on `POST /api/captures` (location/date/people/organizations/topics),
  always authoritative over the same inferred field, with a genuine
  conflict surfaced as its own clarification question. A small UI
  (collapsed `<details>` block on the Remember page) was added for this,
  not just the API.
- **Unknown/declined handling**: a deterministic, no-LLM-call classifier
  recognizes "I don't know" vs. "I'd rather not say" vs. an actual
  answer from the free-text reply. A field marked declined/unknown is
  hard-coded (in code, not just the prompt) to never be re-asked, even
  if the model proposes it again -- verified by a dedicated test that
  configures the fake model to keep asking about the same field and
  confirms it's suppressed.
- **Prior-memory context**: a cheap SQL lexical lookup (no LLM/embedding
  call) injects up to 3 snippets from existing ready memories as
  context into the analysis prompt, explicitly marked as
  context-not-authority -- prevents re-asking about an already-explained
  term without paying for a semantic search.

All four were verified against the real model via a live multipart
request to the dev server: submitting explicit metadata that
conflicted with the model's own inference produced exactly the designed
clarification question, with both values retained pending the answer.

## 7. Local LLM setup

Unchanged from the previous phase's decision and re-confirmed, not
re-litigated: `gemma4:12b` for every generation/structured-output call,
`nomic-embed-text` (768-dim) for embeddings, both via the system Ollama
service, no second generation model resident (VRAM/eviction-thrashing
reasons documented in CLAUDE.md and docs/DECISIONS.md predate this
phase and still hold). This phase's own real-model testing surfaced one
new, concrete finding about this model specifically: **gemma4:12b
sometimes emits the literal string `"null"` (or `"none"`) for an
empty/unset field instead of `""`** -- a structured-output quirk, not an
intentional value. Found live (an `event_date` field came back as the
string `"null"`), fixed in `app/ingestion/analysis.py`'s string-parsing
helpers, and covered by a regression test. `scripts/model-doctor.sh`
(unchanged) continues to be the environment-level check that both
configured models are present and respond correctly.

## 8. Deferred functionality

Explicitly deferred this phase, each with a documented reason in
docs/TODO.md:

- **"Save as memory" has no frontend affordance.** The backend
  (`POST /api/recall/sessions/{id}/save-as-memory`, restricted to text
  provably said by the user in that session) is implemented and tested;
  the UI isn't, because the recall page is one-shot-only today and shows
  no session/cue history to attach a button to -- building that would be
  a materially larger piece of work than this feature itself.
- **`retry_attachment_chunking()` has no API route.** The function is
  implemented, idempotent, and tested; nothing calls it from `app/api/`
  yet.
- **Recall-session history is not searchable.** Confirmed out of scope
  per this phase's spec explicitly, not just unaddressed.
- **Causal/automatic forgetting.** Not implemented; explicit user
  deletion remains the only destructive path, as before.
- **A real threshold sweep** for the new deterministic shortcut's and
  ambiguity detector's constants. Sanity-checked against one real-model
  eval run, not swept across a range -- the 30-case corpus is small
  enough that a swept threshold risks overfitting it.

## 9. Remaining technical risks

**Engineering risks** (fixable with more implementation time, not more
research):

- The two eval regressions (section 4) aren't yet isolated to a specific
  cause among three candidate changes (ingestion collapse, iteration/
  reformulation reduction, the new shortcut). A multi-run baseline
  comparison would resolve this.
- `retry_attachment_chunking()` and "save as memory"'s UI are both
  functionally complete but unreachable from the product surface.
- No wall-clock latency baseline exists for this phase's changes, only
  call counts.

**Research/product risks** (need a product decision or more evidence,
not just more engineering):

- Whether the deterministic shortcut's thresholds (`0.80`/`0.20`) and
  the ambiguity floor (`0.45`) generalize beyond a 30-case hand-written
  corpus is genuinely unknown without a larger, more representative eval
  set.
- Whether recall-session history should ever become searchable, whether
  causal forgetting is wanted at all, and what "save as memory" should
  look like in the UI once/if the interactive recall surface itself gets
  built, are all still open product questions (docs/TODO.md), not
  engineering ones.

## 10. Recommended next experiment

**Isolate the two eval regressions.** Re-run `scripts/run-eval.sh`
3-5 times unchanged to establish how much result variance is pure model
sampling noise on this 30-case corpus. If `clarification_alias_second_term`
and `user_specific_terminology` fail consistently (not just once), bisect
by re-enabling one changed subsystem at a time against the new code
(first: restore `MAX_AGENT_ITERATIONS=3`/`MAX_REFORMULATIONS_PER_ROUND=2`
alone and re-run just `user_specific_terminology`'s category; separately:
compare `analyze_capture()`'s clarification-decision behavior against
the old dedicated `generate_questions()` prompt on the Flimvex case
specifically) to determine which specific streamlining decision, if any,
is the actual cause -- rather than guessing from one combined run. This
is the most direct way to find out whether this phase traded real
accuracy for latency, or just got unlucky on two cases out of thirty.
