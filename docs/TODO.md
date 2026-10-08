# TODO / open questions

## Resolved this phase (iterative recall)

- **Recall is a persisted multi-round session**, not a single
  stateless request. See docs/ARCHITECTURE.md "Iterative recall" and
  docs/DECISIONS.md.
- **Storage-time clarification and retrieval-time clarification are
  separate mechanisms** with separate data models (`MemoryUnit.
  pending_clarification` vs. `RecallSession.current_question`) --
  answering a recall-session question never mutates a MemoryUnit.
- **Whether recall sessions persist**: yes, in Postgres, one row per
  session (see docs/DECISIONS.md) -- verified to survive an app restart.
- **Whether a recall-session fact can be manually promoted to a
  memory**: not implemented. Left genuinely open (see "Open product
  questions" below), not defaulted silently.
- **Ambiguity/discriminating-question heuristic**: a flat score-margin
  check over origin-capture groups, plus attribute-bisection for the
  question itself -- not a probabilistic/Bayesian model (see
  docs/DECISIONS.md; spec explicitly permits this).

## Resolved previous phase (spec section 44 decisions)

- **Clarification gates recall availability.** `status="awaiting_clarification"`
  is excluded from search until answered or discarded. See docs/DECISIONS.md.
- **Derived metadata is not directly editable**; corrections go through
  `POST /memory-units/{id}/correct` (natural-language, re-runs enrichment),
  not a metadata editor.
- **Attachments are semantically searchable through derived chunks**
  (`attachment_chunks`) linked to the parent capture; chunks are not
  independent top-level memories.
- **No automatic deletion/expiry.** Explicit user deletion
  (`DELETE /api/captures/{id}`) is supported and removes files too.
- **Resolved: image understanding**, via the gemma4:12b migration (see
  docs/DECISIONS.md) -- an uploaded image is captioned by the
  generation model's native multimodal understanding and becomes
  full-text/chunk-searchable through the same path PDF/text extraction
  already used. No separate OCR/vision model.
- **Knowledge graph: deferred** -- `semantic.relations` stays a flat list
  of subject/predicate/object triples, not a queryable graph.
- **Multi-agent architecture: deferred** -- one bounded retrieval agent.

## Open product questions

- **Should a user be able to promote something said during recall into
  a permanent memory** ("save this as a memory")? Deliberately
  unresolved (spec section 46) -- it's a real product decision (what
  exactly gets saved, in whose words, with what provenance) rather than
  an obvious engineering call. Current behavior: nothing said during
  recall is ever saved; the user must separately use the ordinary
  capture-creation flow in their own words.
- **Should recall-session history itself become searchable memory**
  (e.g. "what did I ask about last week")? Not implemented. Sessions
  are currently fetchable by id (`GET /api/recall/sessions/{id}`) but
  there's no list/search endpoint over past sessions.
- **Should `AMBIGUITY_SCORE_MARGIN`/`POSITIVE_CUE_BOOST` be calibrated
  via a sweep** the way `VECTOR_SIM_THRESHOLD` was? Not done this phase
  -- there wasn't yet a large enough multi-round eval set (6 cases vs.
  the one-shot set's 30) to make a sweep meaningful. See "Known
  technical gaps" below for the specific failure mode observed.
- **Should the retrieval threshold be relative/adaptive rather than a
  fixed scalar?** The sweep in docs/DECISIONS.md picked 0.55 as the
  highest value with 100% recall@10 on 30 cases, but a margin-below-top-
  score rule might generalize better as the corpus grows. Needs a larger
  eval set to tell the two apart.
- **Resolved: a Browse page (`/captures`) lists every capture newest-
  first with a delete button per capture**, backed by a new
  `GET /api/captures` list endpoint (`GET /api/captures/{id}` already
  existed for a single capture). No pagination -- not warranted yet at
  this MVP's expected capture volume. Deletion asks for confirmation
  client-side (a plain `confirm()`) before calling the existing
  `DELETE /api/captures/{id}`; there was no product ambiguity left here
  once the question was "should this exist", the engineering approach
  was evident.
- **Resolved: `MAX_CLARIFICATION_ROUNDS` is now 3** (was 1), with up to
  `MAX_QUESTIONS_PER_ROUND = 3` candidate questions surfaced per round
  (asked one at a time) -- see docs/DECISIONS.md. Still a judgment call,
  not calibrated against real usage; revisit if it turns out to
  interrogate users more than it helps.
- **Should retrieval-session-gained information ever be promotable to
  durable memory?** Answer in principle: yes (user-confirmed). Not
  implemented -- deliberately deferred, since the retrieval-session
  architecture itself may still change (see the interactive-vs-one-shot
  benchmark finding below) before a promotion mechanism is worth
  designing. Open: what exactly gets promoted, in whose words, with what
  provenance.
- **Narrower case of the above, raised by the user: a follow-up/
  discriminating-question answer during a `RecallSession` round
  (`app/retrieval/discriminate.py`, `app/retrieval/cues.py`) should be
  added as a new memory if it reveals something not already captured
  anywhere.** Not formalized, just written down. Today that answer is
  only ever consumed as a disambiguation cue for the current session
  (see `RecallSession.cues` in `app/retrieval/session.py`) and never
  persisted as a `MemoryUnit` -- even when the user volunteers a
  genuinely new fact while answering, it's discarded once the session
  ends. Open: how to detect "not already captured" (dedup against
  existing units -- by what, embedding similarity?), whether this
  happens automatically or needs the user to confirm first, and in
  whose words the new memory gets written (the literal answer text, or
  something derived from it).
- **A "forget" process driven by causal relationships between captures.**
  Raised by the user, not formalized -- written down as a direction,
  not a spec. Idea: a new capture can causally resolve or consume an
  earlier one, and that should be reflected in storage somehow (delete,
  archive, or some new status) rather than just adding another memory
  alongside the stale one. Example: logging "went grocery shopping"
  should somehow act on a previously stored shopping-list memory --
  roughly delete or archive the items that were just bought. This is
  distinct from the generic time-decay "human-like forgetting" listed
  under "Explicitly deferred" below -- this is triggered by the
  semantic content of a *new* capture, not by elapsed time. Currently
  nothing in the pipeline looks for this relationship at all: every
  capture is additive, and the only existing deletion path is the
  user's own explicit `DELETE /api/captures/{id}`. Open, unformalized
  questions this would need to answer before it could be designed:
  how a causal/consuming link between two memories would even be
  detected (an LLM judgment per new capture against candidate related
  memories? something cheaper?), what "resolved" should mean for a unit
  that wasn't deleted outright (archived-but-still-searchable? excluded
  from recall like `awaiting_clarification`?), how wrong/false-positive
  matches get corrected or undone, and whether this should ever run
  automatically vs. always surfacing as a confirmation to the user first.
- **Eventually, fuse the memory-retrieval and attachment-upload UI
  workflows into one surface that can retrieve and attach memory
  simultaneously.** Raised by the user, not formalized -- written down
  as a direction. Today "Remember" (`/`, creating captures + attaching
  files) and "Recall" (`/recall`, querying) are two separate pages with
  no interaction between them; attaching a file is only ever something
  you do while writing a *new* memory (`POST /api/captures` with
  `files`) or adding to an existing one you already know the id of
  (`POST /api/captures/{id}/attachments`, via `add_attachment()`).
  There's no flow where recalling something and attaching a file to it
  happen in the same interaction -- e.g. asking "what's my rent
  agreement" and, in that same place, being able to attach an updated
  copy to whatever it finds. Open, unformalized questions this would
  need to answer before it could be designed: what the unified
  interaction even looks like (search-then-attach? a single compose box
  that can both ask and attach?), whether an attachment added this way
  attaches to a specific existing capture/unit or starts a new one,
  and how this interacts with the capture/memory-unit distinction
  (`docs/ARCHITECTURE.md`) if attaching during recall doesn't cleanly
  map to either flow.

## Known technical gaps

- **To fix: a terse, context-free capture (one or two words, no stated
  intent) is stored but not retrievable under any query describing the
  *purpose* those words were captured for.** Verified live: storing
  "Bananas.", "Pepper.", and "Garlic." as three separate captures (each
  correctly reaches "ready" with no clarification question, since
  they're common real words, not invented/ambiguous ones -- see the
  `generate_questions()` gap above for why that mechanism doesn't help
  here either) produces semantic enrichment like "A brief mention of
  bananas." with topics like "produce"/"healthy snacks" -- accurate,
  but with zero connection to "buying," "shopping," or "a list."
  Querying "What do I need to buy?" afterward **abstained entirely**
  (0 sources) -- and a differently-phrased query that happened to
  score well against *other*, unrelated stored memories (which
  literally contained the words "buy"/"need" in fuller sentences)
  returned a confident but completely wrong answer, with none of the
  three grocery items mentioned at all. The items remain findable via
  their own literal words ("bananas pepper garlic" finds all three
  correctly) -- the gap is specifically that nothing links them to the
  *intent* behind writing them down.

  **Root cause:** segmentation/clarification/enrichment only ever see
  each capture's own words; a bare noun carries no "this is something
  I need to buy" signal for the model to work with, and -- separately
  -- storing each item as its *own* capture means they don't even share
  a `capture_id` for `sibling_context()` (the same mechanism that lets
  "related context" from one unit help interpret another) to stitch
  them together at answer time.

  **Verified fix, already supported, no code change:** writing it as
  *one* capture with the intent stated once --
  `"Need to buy: bananas, pepper, and garlic."` -- segments into one
  coherent unit and is correctly found by "What do I need to buy?"
  afterward, listed alongside (not instead of) the other genuinely
  "need to buy" memories. This is a real, usable workaround today, not
  a hypothetical fix.

  **Open question this raises, not yet designed:** should the app ever
  intervene when a capture is this terse -- e.g. a clarifying question
  along the lines of "what's this for?" even for an ordinary real word,
  specifically when the capture is very short and carries no stated
  purpose? That would be a deliberate carve-out from the existing
  "don't ask about well-known real-world words" rule
  (`app/ingestion/clarify.py`'s system prompt), traded against the
  spec's explicit goal of *not* turning storage into an interrogation
  -- a product call, not an obvious engineering one.

- **To fix: ingestion-time clarification/enrichment never checks what's
  already stored.** Confirmed directly in code: `generate_questions()`
  (`app/ingestion/clarify.py`) and `enrich()`
  (`app/ingestion/enrichment.py`) take no `db`/session parameter at
  all -- there is no `hybrid_search()` or any other memory-store lookup
  anywhere in the ingestion path (`app/ingestion/pipeline.py`).
  Retrieval only ever happens at recall time
  (`POST /api/recall`/`RecallSession`), never during storage. Each
  capture is enriched in total isolation from every other one.
  Concretely: if an earlier capture explains an invented/unfamiliar
  term (answered via clarification, or just named in passing), and a
  *later*, unrelated capture mentions that same term again, the system
  has no way to recognize it's already known -- `generate_questions()`
  may ask about it again, and `enrich()` has no prior context to draw
  on when deciding what it means, since `unit.clarifications` is scoped
  to that one `MemoryUnit` row, never shared across captures. Needs
  real design work, not just wiring in a DB session: what would even
  count as "the same term" (exact string match? semantic similarity?),
  how much of a prior match's context to inject into the prompt
  without ballooning it, and whether this should only suppress
  redundant *questions* or also inform `enrich()`'s actual semantic
  output (aliases/meaning) with what a prior capture already
  established.
- **No retry mechanism if `finish_attachment_chunking` is dropped by a
  server restart.** Unlike unit enrichment (`finish_unit_processing`,
  recoverable via `POST /memory-units/{id}/retry`), an attachment whose
  background chunking task never ran has no equivalent recovery
  endpoint -- it stays fully full-text searchable via its
  `extracted_text` (that's never affected), just without a fine-grained
  chunk-level index, with no way to trigger re-chunking short of
  deleting and re-adding the attachment. Acceptable at this MVP's
  expected restart frequency; noted so it isn't silently forgotten. See
  docs/DECISIONS.md "offloading to background processing".
- **To fix: no handling of contradictory memories at all.** Verified
  live, not theoretical: storing "3 is my favourite number." and later
  "4 is my favourite number." produces two independent, fully valid
  memory units -- nothing flags them as contradictory, nothing merges
  or supersedes the older one, every capture is purely additive (see
  the "forget" direction below, which is related but distinct: that's
  about a *causal* link between captures, e.g. a shopping trip
  resolving a shopping list; this is about two memories that are just
  factually incompatible with no causal relationship at all).
  `hybrid_search` scored both candidates almost identically in testing
  (fused scores differing by ~0.002, i.e. noise, not a meaningful
  preference), so both were passed to answer generation as evidence,
  and the model happened to handle it well on its own ("You have
  recorded both 3 and 4 as your favourite numbers.") -- but there is no
  code-level guarantee of that. The `_ANSWER_SYSTEM` prompt
  (`app/retrieval/pipeline.py`) has no instruction at all for
  contradictory evidence; a different phrasing or more lopsided
  retrieval scores could just as easily produce a confident, wrong
  single answer with nothing in the system to catch it. Needs real
  design work, not just a prompt tweak: should storing a new fact that
  contradicts an existing one surface the conflict to the user at
  write time? Should retrieval/answer generation be required to
  flag disagreement explicitly rather than leaving it to model judgment
  per-query? Should older contradicted memories be superseded,
  archived, or just left to coexist with a visible conflict marker?
- **Interactive recall (`RecallSession`) is not the default UI** --
  reverted after `eval/run_interactive_benchmark.py` measured it
  underperforming both the semantic baseline and the one-shot agent
  overall (see docs/DECISIONS.md). Root cause identified:
  `apply_negative_cues`/`apply_positive_cue_boost`/
  `generate_discriminating_question` (`app/retrieval/discriminate.py`)
  only check `people`/`organizations`/`locations`/`aliases`/`topics` --
  an entity name that lands in `relations` or `definitions` instead
  (common for a project/thing name, per the benchmark trace) is invisible
  to all three. Fixing this and re-running the benchmark is the natural
  next step before restoring it as the default; code, tests, and the
  `/api/recall/sessions/*` API are untouched and fully usable in the
  meantime.

- **Ambiguity detection is sensitive to full-text-search score
  asymmetry.** Observed directly during manual testing: two candidates
  that are equally plausible to a human can get very different fused
  scores purely because one sentence happens to share more literal
  words with the query (`websearch_to_tsquery` AND-matching), not
  because it's actually the better answer; conversely two candidates
  with genuinely different relevance can end up close by coincidence.
  The eval cases in `eval/session_cases.yaml` were deliberately phrased
  to produce symmetric FTS matches where ambiguity is the point of the
  case -- real stored memories won't always cooperate this way.
- **Cue-to-candidate matching (`apply_negative_cues`/
  `apply_positive_cue_boost`) is an exact-ish metadata match**, not
  fuzzy. "Anna" matches a candidate whose `semantic.people` contains
  exactly "Anna" (case-insensitive); it won't match "Annalisa" or a
  misspelling. Works in testing because the same enrichment call that
  populates `semantic.people` is reasonably consistent, but isn't
  linguistically robust.
- **Cue extraction's "bare confirmation" fallback is a judgment call
  the model doesn't always make correctly.** Observed during manual
  testing: a reply that both confirms/denies the question *and* adds a
  genuinely new detail ("Yes, it was at ETH") initially caused the
  model to extract only the confirmation and drop the new detail,
  silently stalling disambiguation. The prompt was strengthened
  (`app/retrieval/cues.py`) to explicitly require extracting new
  information even when a yes/no is also present, and this now works
  correctly in testing -- but like the enrichment clarification-
  triggering gap below, it's model judgment, not a hard guarantee.
- **Re-embedding is whole-row, not incremental.** `scripts/reembed.py`
  re-embeds a unit/chunk entirely if its embedding is missing or
  stale-model; it has no concept of "only the aliases changed, reuse the
  rest," which is fine at this scale but wouldn't be at a much larger one.
- **Segmentation's verbatim-fallback is a blunt instrument.** It protects
  against promoting paraphrased content to authoritative status, but a
  model that produces clean, correctly-punctuated verbatim sub-sentences
  while still cutting across a pronoun reference (observed during eval:
  "I met Anna... and she recommended a paper" split into two
  grammatically-clean verbatim halves) will pass the check and still
  lose the antecedent in the *primary* evidence block. Mitigated at the
  answer-generation layer (the prompt explicitly directs resolving
  pronouns using sibling "related context" evidence, confirmed fixing
  the one eval failure this caused), but segmentation itself has no
  linguistic awareness of reference chains.
  - **Same bug, different shape, newly observed:** segmentation
    deterministically (3/3) splits "My trip to Japan last year was
    amazing, especially the food in Osaka." into "My trip to Japan last
    year was amazing" / "the food in Osaka" -- the second fragment loses
    the elided "amazing" it depended on, no pronoun involved at all. This
    made `test_paraphrased_query_retrieves_relevant_memory` flip from
    reliably passing to failing more often than not after the
    multi-prompt enrichment change (see docs/DECISIONS.md) shifted an
    already-borderline grounding judgment for this exact evidence
    combination. Not fixed -- same root class of problem as the pronoun
    case, this time an elided qualifier instead of a referent; the
    sibling-context mitigation only targets explicit pronouns today.
- **Clarification question targeting/triggering is inconsistent** across
  model calls for borderline cases. Bounded to 1 round per unit, so the
  blast radius is small, but not fully reliable -- and now that it
  gates recall, an unnecessary question has a bigger UX cost than before
  (the memory is invisible to recall until answered or discarded).
- **Chunk size (1200 chars, 150 overlap) is a fixed, untuned default.**
  Works for the eval set's short attached notes. A genuinely large
  (50+ page) PDF's total *processing cost* is now bounded
  (`membase_max_chunks_per_attachment`, see docs/DECISIONS.md "Large
  attachments") -- but the 1200/150 split itself, and whether 600
  (raised from 300 by user request) is the right chunk-count ceiling,
  remain untuned against a real large document, just no longer
  unbounded.
- **Image captions are not empirically quality-tested the way
  segmentation/enrichment prompts were** (see docs/DECISIONS.md "Gemma 4
  migration" -- qwen's choice was validated by direct manual A/B testing
  before being trusted; the image-captioning prompt in
  `app/ingestion/pipeline.py`'s `_IMAGE_CAPTION_PROMPT` has not had the
  same treatment yet). Functionally wired up and covered by fast tests
  with a fake model, but caption *quality* against real images (does it
  reliably transcribe visible text? describe enough detail to be
  findable later?) hasn't been evaluated the way the text pipeline was.
- **A RecallSession round can be more expensive than the one-shot
  agent's**, since `automatic_search` re-searches the original query
  plus every accumulated positive cue from scratch each round (by
  design -- see docs/DECISIONS.md on why candidates must not be
  "forgotten" between rounds), on top of the same per-round ≤3-iteration
  bound the one-shot agent already has. Not measured under load; only
  observed to resolve within 1-2 rounds for the 6-case iterative eval
  set.
- **No rate limiting / request size limits** beyond the per-file size
  cap. Not a concern for a single-user localhost MVP.
- **UI was validated via direct HTTP calls against the running server
  (including a real restart) and rendered-HTML inspection, not a live
  browser session** -- no browser automation tool was available in the
  environment this was built in, same limitation as the previous phase.

## Explicitly deferred (not evaluated as gaps, just out of scope)

- Autonomous background agents, general knowledge-graph storage, memory
  consolidation, human-like forgetting, multi-agent systems, advanced
  temporal reasoning, multimodal (image) understanding, cloud
  deployment, mobile apps, multi-user auth, a learned/trained reranker.
  (Multi-turn *recall* is now implemented -- see "Resolved this phase"
  above; general open-ended multi-turn chat about anything is still out
  of scope, since the conversational surface here exists specifically
  for narrowing down a memory, not as a chat assistant.)
