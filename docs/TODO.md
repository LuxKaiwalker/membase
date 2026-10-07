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
- **Image understanding: deferred** (unchanged from the previous phase).
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
- **Should there be a UI affordance for browsing/deleting individual
  captures**, not just creating and recalling? The API supports it
  (`GET`/`DELETE /api/captures/{id}`); the web UI doesn't expose it yet.
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

## Known technical gaps

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
- **No image understanding.** Images are validated and stored but never
  captioned/OCR'd.
- **Chunk size (1200 chars, 150 overlap) is a fixed, untuned default.**
  Works for the eval set's short attached notes; untested against a
  genuinely large (50+ page) PDF.
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
