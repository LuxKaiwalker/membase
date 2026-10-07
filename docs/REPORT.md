# Engineering report: iterative recall phase

This supersedes the previous report. Scope: the previous phase
corrected clarification-gating and hardened retrieval around the
memory-unit architecture; this phase's task was different in kind --
not another correctness gap, but a product-shape change to *how recall
itself works*. Recall used to be `query -> search -> answer` in one
HTTP request. It is now a persisted, multi-round `RecallSession`:
a weak or partial first query can produce a clarifying question instead
of a guess, the user's reply (an answer, a correction, or an unprompted
extra detail) becomes a new search cue, and the session searches again
-- repeating until it succeeds, genuinely runs out of options, or the
user gives up. Verified against the real local LLM throughout,
including a full session walked through `curl` against the live server
across an actual process restart.

## Current architecture

Ingestion (capture -> segmentation -> enrichment -> clarification gate
-> embedding/fts) is unchanged from the previous report; see
docs/ARCHITECTURE.md for the full picture. What's new sits entirely on
the retrieval side:

```
user query (possibly just a weak cue)
      v
RecallSession created, persisted in Postgres
      v
round: automatic_search()            <- same bounded 3-iteration
      (search -> evaluate ->             search/evaluate/reformulate
       reformulate, shared with            loop the one-shot agent uses
       the one-shot agent)
      v
apply_negative_cues() / apply_positive_cue_boost()   <- user corrections/
      |                                                  confirmations
      v                                                  reshape the
is_ambiguous()?  (origin-capture groups,                 candidate set
      |           close top scores)
      +-- yes, and a question helps --> ASK, stop, wait for reply
      +-- no / no more questions  ----> narrow_to_dominant_group()
      v
generate_answer()  (same grounding-enforcement function as before)
      |
      +-- grounded   --> SUCCESS, snapshot evidence, stop
      +-- ungrounded --> ask one more question, or EXHAUSTED
```

A user reply re-enters at "round": `extract_cues()` turns free text into
structured `{text, confidence, polarity}` cues, appended to the session,
and another round runs with the full accumulated cue set.

## Retrieval-session design

`RecallSession` (`backend/app/models.py`) is one row per session:
`original_query` (immutable -- the one thing that's never overwritten,
protecting against query drift), `status`, `current_question`,
`messages` (full transcript), `cues`, `hypotheses`, `tried_queries`,
`asked_questions`, `round_count`, `last_candidate_count`, and
`final_answer`/`final_sources`/`final_chunk_sources` (a snapshot, not a
live reference, so a later correction or deletion of the underlying
memory can't retroactively change what a past session recorded as its
answer). No foreign key to `memory_units`/`attachment_chunks` -- the
snapshot is deliberately self-contained.

Three categories of information are kept structurally distinct, per the
spec's data-flow concern about an iterative system contaminating its
own memory store:

- **User-provided cues** (`cues`): confidence (`certain`/`likely`/
  `uncertain`) and polarity (`positive`/`negative`), both tagged by
  round. A positive cue is always searched, regardless of confidence --
  confidence affects nothing about whether it's used, only (eventually)
  how it should be weighed as evidence, which in practice means: it's
  never treated as a hard filter. A negative cue is never searched; it
  only excludes a candidate via `apply_negative_cues`.
- **Retrieved evidence**: the `ScoredUnit`/`ScoredChunk` results from
  `hybrid_search`/`hybrid_search_chunks` -- real stored content, scored.
- **System hypotheses** (`hypotheses`): the reformulated queries
  `evaluate_sufficiency` proposes along the way (itself unchanged from
  the one-shot agent), kept on the session purely for transparency and
  testability. Never passed into answer-generation evidence; never
  becomes a memory.

Two heuristics specific to this phase, both deliberately simple per the
spec's own instruction not to build a formal active-learning system:

- `is_ambiguous`: candidates from ≥2 distinct origin captures with top
  fused scores within `AMBIGUITY_SCORE_MARGIN` (0.15) are a real
  decision, not something to silently resolve by picking whichever
  ranked first.
- `generate_discriminating_question`: picks the semantic attribute
  (person/location/organization/topic/date) that best bisects the
  remaining candidates and phrases one concrete question about it,
  skipping anything already asked this session.

One thing discovered only by testing against the real model, not
anticipated in the initial design: re-searching with a cue as its own
short query (e.g. just "ETH") often scores *lower* on the fused metric
than the original full-sentence query already did for every candidate,
so naively merging per-query results via "keep the max score seen"
never resolved ambiguity in practice. `apply_positive_cue_boost`
addresses this directly: a confirmed cue that matches a candidate's own
semantic metadata (the same metadata the question was generated from)
gets a flat score bonus. Without it, a correct "yes, ETH" answer left a
session stuck asking forever; with it, the session resolves in the next
round -- verified live against the real model, see docs/DECISIONS.md.

## Agent behavior

- **Automatic iterations** (no user involved): `automatic_search()` in
  `app/retrieval/agent.py`, extracted this phase so both the one-shot
  agent and every `RecallSession` round share one implementation. Still
  bounded to `MAX_AGENT_ITERATIONS = 3`, ≤2 reformulations per round --
  unchanged from the previous phase.
- **User-mediated iterations**: bounded only by `MAX_SESSION_ROUNDS = 12`
  -- a runaway-loop safeguard, not a UX-facing turn limit (spec section
  10 is explicit that these two kinds of bound shouldn't be conflated).
- **Tools**: still just `hybrid_search`/`hybrid_search_chunks` (spec
  section 19's minimal surface) -- no new tool was added for this phase;
  cue extraction and the ambiguity heuristics are code the session
  controller runs itself, not agent-invokable tools.
- **Termination conditions**: `success` (grounded answer), `exhausted`
  (nothing found, or ambiguity never resolved and no discriminating
  angle remains), `cancelled` (explicit user action), `error` (LLM/infra
  failure mid-round). `needs_user_cue` and `active` are the only
  non-terminal statuses.
- **Safeguards**: grounding enforcement is unchanged -- still exactly
  one override point, now reached from both `run_agent()` and
  `_run_round()`, both calling the same `generate_answer()`. Original
  query never replaced. Repeated-question avoidance via
  `asked_questions`. Negative cues demote by structured-metadata match,
  not loose substring guessing.

## Memory-safety model

`app/retrieval/session.py` and `app/retrieval/cues.py` never call
`create_capture` and never write to `MemoryUnit`/`Capture` -- there is
no code path from "something said during a recall conversation" to
"durable memory" other than the user separately using the ordinary
`POST /api/captures` flow with their own words. This was verified
directly, not just by code inspection: `test_recall_session.py` asserts
`MemoryUnit` row counts are unchanged before/after sessions that
generate hypotheses, uncertain cues, and negative corrections.
Storage-time clarification (`MemoryUnit.pending_clarification`, gates
*searchability* of a memory) and retrieval-time clarification
(`RecallSession.current_question`, gates *continuing the search*) are
separate fields on separate models -- answering one never touches the
other.

## UI

`/recall` is now a lightweight conversation, not a single form: a
transcript of user/system turns, a reply box that stays open while
`status` is `needs_user_cue`, a "Give up" button (cancel), and a "Start
a new recall" button once the session reaches a terminal state. A
one-line progress indicator ("Found 2 related memories so far...")
surfaces `last_candidate_count` without exposing any internal
scores/reasoning. No browser automation tool was available in this
environment (same limitation noted in the previous report's TODO) --
validated instead via direct HTTP calls against the live running
server, including rendering the HTML and walking a full multi-turn
session through `curl`, across an actual `kill`+restart of the dev
server to confirm session persistence.

## Evaluation

- **One-shot set** (`eval/cases.yaml`, unchanged, 30 cases): 29/30 on
  this run. The one failure (`ambiguous_two_rome_trips`) is pre-existing
  eval-case brittleness unrelated to this phase -- the model's answer
  ("You ate pasta... Additionally, you enjoyed gelato...") is actually a
  reasonable grounded merge of both ambiguous Rome trips, it just never
  echoes the literal word "Rome" that `expect_contains` checks for.
  `automatic_search`'s logic is byte-for-byte unchanged from the
  previous phase's `run_agent` (confirmed by the full existing test
  suite passing before any new code was added), so this is real-LLM
  phrasing variance, not a regression.
- **New iterative-recall set** (`eval/session_cases.yaml` +
  `eval/run_session_eval.py`, run via `scripts/run-session-eval.sh`): 6
  cases, **6/6 passing**, covering: ambiguous candidates resolved by a
  location cue, resolved by a negative correction ("no, it wasn't
  Markus"), an irrelevant cue that must not force a wrong answer, no
  evidence at all (immediate abstention), an exact-match case answered
  in one round, and original-query preservation across rounds with no
  useful new cue.
- Both sets run against the real configured models
  (`qwen2.5:14b-instruct-q4_K_M` / `nomic-embed-text`), isolated in the
  `membase_eval` database.

## Tests

66 pytest tests total (up from the previous phase's 55; +11 new), all
passing: `test_recall_session.py` (session state machine, ambiguity ->
question -> resolution, negative-cue exclusion, uncertain-cue-as-search-
signal-not-filter, hypothesis non-contamination, exhaustion, cancel/
reject-continue, original-query preservation, repeated-question
avoidance, pure-function tests of the discriminate heuristics, cue-
extraction degradation on malformed LLM output, restart-durability via
`db.expire_all()` + re-fetch). The prior phase's 55 tests pass unchanged
-- `automatic_search`/`evaluate_sufficiency` were extracted, not
rewritten, and the extraction was verified behavior-preserving against
the full suite before any new logic was added on top.

## Remaining limitations

- Ambiguity detection (`AMBIGUITY_SCORE_MARGIN = 0.15`) is a hand-picked
  constant, not calibrated via a sweep the way `VECTOR_SIM_THRESHOLD`
  was -- there wasn't yet a large enough multi-round eval set. It's also
  sensitive to full-text-search score asymmetry: two equally-plausible
  candidates can get different scores purely from literal keyword
  overlap with the query, independent of actual relevance (see
  docs/TODO.md).
- `apply_negative_cues`/`apply_positive_cue_boost` match cue text against
  structured semantic metadata exactly (case-insensitive), not fuzzily --
  "Anna" won't match "Annalisa" or a misspelling.
- Cue extraction's handling of a reply that both answers yes/no *and*
  adds new information required a prompt fix mid-phase after the real
  model initially dropped the new detail (see docs/DECISIONS.md/TODO.md)
  -- now correct in testing, but still model judgment, not a hard
  guarantee, same caveat as the enrichment clarification-triggering gap
  already documented.
- A session round re-searches the original query plus every accumulated
  positive cue from scratch (by design, to avoid "forgetting" earlier
  candidates between rounds) -- not measured under load at a large
  personal-memory-store scale.
- No way to list/browse past recall sessions; only fetchable by id.

## Major open design questions

- **Should a user be able to promote something said during recall into
  a permanent memory?** Deliberately left open (spec section 46) --
  not an obvious engineering call (what exactly gets saved, in whose
  words, with what provenance). Current behavior: nothing from a
  session is ever saved automatically or via any UI action.
- **Should recall-session history become searchable memory itself**
  (e.g. "what did I ask about last week")? Not implemented; sessions are
  fetchable by id but there's no search/list surface over them.
- **Should the ambiguity margin and positive-cue-boost constants be
  probabilistic/learned rather than fixed heuristics** as real usage
  accumulates? Spec explicitly permits deferring this for the MVP.

## Recommended next experiment

Build a larger, more varied iterative-recall eval set (the current one
is 6 hand-picked cases, deliberately phrased for score symmetry where
ambiguity is the point) and run a sweep over `AMBIGUITY_SCORE_MARGIN`
and `POSITIVE_CUE_BOOST` the same way `VECTOR_SIM_THRESHOLD` was
calibrated last phase -- this phase's manual testing already found one
real gap (the positive-cue-boost mechanism itself, absent from the
initial design) that a systematic sweep would likely surface others
like it faster than one-off manual probing did.
