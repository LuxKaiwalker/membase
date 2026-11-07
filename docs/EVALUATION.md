# Evaluation: the multi-strategy retrieval benchmark

What this document covers: the one experiment in this repo that compares
*retrieval strategies* against each other on identical stored memories, so
the question "does associative/iterative recall actually beat conventional
retrieval?" has an answer with numbers behind it — including the answer
"no".

It is deliberately separate from the other eval scripts, which measure
different things:

| script | question it answers |
| --- | --- |
| `scripts/run-eval.sh` | does one-shot recall get the 30 hand-written cases right? |
| `scripts/run-session-eval.sh` | does iterative recall handle multi-round/ambiguous cases? |
| `scripts/run-interactive-benchmark.sh` | does *asking a clarifying question* recover what single-shot cannot? (3 modes) |
| `scripts/sweep-threshold.sh` | what should `VECTOR_SIM_THRESHOLD` be? |
| `scripts/benchmark-recall-latency.sh` | where does a recall round's wall-clock time go? |
| **`scripts/run-strategy-benchmark.sh`** | **which of four retrieval strategies is actually better, and at what cost?** |

Files:

- `eval/memory_corpus.yaml` — the corpus (44 memories)
- `eval/strategy_queries.yaml` — the query set (40 queries, 8 categories)
- `eval/strategy_benchmark.py` — the runner
- `scripts/run-strategy-benchmark.sh` — the wrapper (isolated `membase_eval` DB)
- `eval/.corpus_state.json` — the corpus cache (gitignored)
- `eval/strategy_benchmark_results.json` — raw per-query per-strategy results (gitignored)

## How to run it

```
nix develop
scripts/run-strategy-benchmark.sh --self-check   # validate the YAML only, no DB, no model
scripts/run-strategy-benchmark.sh --limit 8      # fast smoke run (8 memories, the queries they can answer)
scripts/run-strategy-benchmark.sh                # full run
scripts/run-strategy-benchmark.sh --repeat 3     # full run, 3x, with the noise floor
scripts/run-strategy-benchmark.sh --reseed       # throw the cached corpus away and re-ingest
```

The wrapper is a verbatim copy of `scripts/run-eval.sh` with the final line
swapped, so the same database-isolation guarantee applies: `PGDATABASE` is
`membase_eval` before Python starts, and `assert_eval_database()` re-checks
the live connection rather than the env var. The benchmark `TRUNCATE`s its
tables when it seeds, which is why that guard is not optional.

### Corpus caching

Seeding goes through the real ingestion pipeline — roughly 6 generation
calls plus one embed per memory, ~20–25s of model time each (see
docs/DECISIONS.md "Benchmark: fast pipeline vs. structurer wall-clock
time"). A 44-memory corpus is therefore 15–25 minutes *before any query
runs*, and every other script in `eval/` re-ingests from scratch on every
invocation. That makes iterating on a retrieval strategy impractical, so
this benchmark caches:

1. After seeding it writes `eval/.corpus_state.json`: a SHA256 fingerprint
   of the corpus YAML (plus the `--limit`), the database name, the number of
   ready units, how long seeding took, and the `M-id -> capture_id` map.
2. On startup it re-seeds **unless** all of these hold: the fingerprint
   matches the current YAML, the cache was written for the database now
   connected, the recorded memory count matches the corpus, the database
   still reports exactly the recorded number of ready units, and every
   cached `capture_id` still exists. Any mismatch prints the specific reason
   and re-seeds.
3. The output always says which path was taken — `CACHE HIT -- seeding
   SKIPPED` (with when the reused ingest happened and how long it took) or
   `CACHE MISS` followed by per-memory seeding progress and the total.

`--limit N` is part of the fingerprint, so an 8-memory smoke corpus can
never satisfy a full run's cache check or be reported as a full run.

**Per-memory seeding cost is extremely sensitive to concurrent load on the
single local model, which is the main reason this cache exists.** Measured on
this machine on the same corpus within one hour: ~77s per memory while the
GPU was shared with other work (8 memories in 617s), degrading to ~70–95s and
producing repeated `Segmentation LLM call failed; falling back to a
deterministic sentence split` warnings when several processes were hitting
Ollama at once. There is one generation model on one 11GB GPU
(CLAUDE.md "Environment facts"); every concurrent caller queues behind the
same llama.cpp slot, and a call that exceeds `llm_timeout_seconds` (90s) is
*dropped*, degrading that memory's ingestion rather than merely slowing it.
Two consequences: seed on an idle machine if you can, and check the
degraded-unit count afterwards if you could not.

One related footgun the benchmark works around in `eval/`: `app/db.py` sets
`idle_in_transaction_session_timeout=180000`, and `create_capture_and_finish()`
ends with a `refresh` that autobegins a read transaction. Left open across the
next memory's minutes of model time, Postgres terminates the connection
mid-seed — observed for real here under contention. `seed_corpus()` and the
query loop therefore `db.commit()` after each unit of work; it costs nothing
(`expire_on_commit=False`) and removes the failure mode.

Either path then prints the association graph's size (entities, unit-entity
links, unit associations) and how many ready units have
`enrichment_status='unavailable'`, with a warning if either number is bad.
Both can silently invalidate a run: an empty graph makes `associative` look
like it adds nothing when in fact there was nothing to walk, and a unit whose
enrichment never completed was matched against incomplete information, is
penalised by `rerank`, and flips `should_expand`'s verdict. Either is grounds
to `--reseed` on an idle machine rather than to believe the numbers.

`--limit N` also **filters the query set**: a query whose
`relevant_memories`/`ground_truth_memory` are not all seeded is skipped and
named in the output, because running it would score as a retrieval failure
for all four strategies and that would be a lie about the strategies.
Unanswerable queries always run; they have no ground truth to be missing.

The cache does not survive a corpus edit, an `alembic` downgrade that drops
rows, or someone else truncating `membase_eval`. It is a speed optimisation
with a conservative validity check, not a database.

## The corpus

44 first-person notes, each ingested as its own independent capture (no
hand-grouping), so retrieval has to find the right note among real
competition. Entity names are globally unique across the corpus — `eval/cases.yaml`
has documented cross-contamination from reusing "Anna" between cases, and
none of `eval/interactive_benchmark.yaml`'s names appear here either.

Deliberately present, because these are the structures the four strategies
are supposed to differ on:

- **Recurring entities.** Eight people, five projects, six locations, two
  recurring events, appearing across many memories — the raw material the
  association graph is projected from.
- **Temporal variety.** Explicit day (`3 February 2025`), month-only
  (`in March`), season-only (`in the spring`, `in late summer`), year-only,
  and a deliberate set of **undated** memories. The undated case must be
  represented: a date-free memory is the normal case in real notes and the
  one most likely to break date-sensitive evidence collection.
- **Causal statements** — "we abandoned X because Y" (M2, M15, M31).
- **Three ambiguous pairs**, each differing in *exactly one* attribute:
  M8/M9 (person), M13/M14 (location), M23/M24 (year). Word-for-word
  identical otherwise. These are what a clarification mechanism has to
  resolve; nothing in the text itself can.
- **Lexical distractors** (M26–M30): memories that share salient query
  words but are about something else. The sharpest is Project *Kestrel* (a
  data migration) against an actual kestrel bird (M27) and a conference talk
  on bird *migration* (M26). This is what separates retrieval from keyword
  matching.
- **Fragmentary recollections** (M18, M37, M38, M39): terse notes with
  almost no context.
- **Multi-hop chains**, four of them, where memory A names an entity and
  memory B holds the answer, and B shares no wording with the query:
  M10→M11 (who taught me to centre clay → she broke her wrist),
  M3→M4 (who supplied the hardware → that supplier is defunct),
  M16→M17 (who sold me the lathe → he rents kayaks now),
  M6→M44 (who owns the migration → she is my manager).
- **Two contradiction pairs** stating incompatible facts: M21/M22 (one
  child: a daughter vs. a son) and M34/M35 (boiler replaced in 2019 vs.
  never replaced). The answer prompt is required to *state* a conflict
  rather than silently pick a side (docs/DECISIONS.md "Contradiction
  detection"), so these test that too.

Each memory carries a `theme` tag, used for nothing in scoring — it exists
so a human reading the corpus can see what each block is for.

## The query set

40 queries, every one tagged with exactly one category:

| category | n | what it tests |
| --- | --- | --- |
| `exact` | 5 | the query uses the memory's own words / proper nouns |
| `semantic` | 5 | paraphrase with little or no lexical overlap |
| `indirect_association` | 5 | the fact lives in a memory *linked* to the one the query matches |
| `multi_hop` | 4 | two steps: A names the entity, B holds the answer |
| `temporal` | 5 | date- and ordering-sensitive, including season-only |
| `ambiguous` | 5 | several stored memories remain plausible; a clarifying question genuinely resolves it |
| `vague_episodic` | 6 | a feeling or fragment rather than a fact |
| `unanswerable` | 5 | nothing stored answers it; `expect: abstain` |

Per-query fields:

- `relevant_memories` — **every** acceptable correct target, scored against
  by recall@k. For an ambiguous pair this contains *both* members:
  retrieving either is correct *retrieval*, and choosing between them is
  what answer correctness measures. Keeping those two judgements separate is
  the point.
- `ground_truth_memory` — the single best target. Used for exactly one
  thing: grounding the simulated user's replies.
- `expect_contains` — a list of case-insensitive substrings; **any** match
  on a grounded answer counts as correct.
- `expect: abstain` — unanswerable queries only, and they carry no ground
  truth at all (the self-check enforces this).

The five unanswerable queries come in two varieties, which is intentional:
Q38/Q39 have nothing related stored at all (the easy case), while
Q36/Q37/Q40 have a *related memory that does not contain the answer* — a
fragment mentioning a yellow door, a note saying "blue folder, third
drawer… I no longer know why". The hard variety is where a strategy that
retrieves aggressively is most tempted to answer anyway.

### Self-check

Before touching the database, the runner validates both YAML files and
**aborts** on any problem. It checks: duplicate ids, corpus size, missing
theme/text, duplicate query ids, minimum query count, category membership,
full category coverage, the minimum unanswerable count, that unanswerable
queries carry no ground truth, that answerable queries have
`relevant_memories` + `ground_truth_memory` + `expect_contains`, that every
id referenced resolves to a real corpus memory, and that each
`ground_truth_memory` is a member of its own `relevant_memories`.

This exists because the failure it catches is otherwise invisible: a query
pointing at a memory id that does not exist scores zero recall for every
strategy, which reads as a retrieval finding rather than as a typo.
`--self-check` runs just this and exits.

## The four strategies

All four run against the same stored memories, read-only with respect to
memories, in one process, with nothing truncated between them. Each returns
the same dict: `{answer, grounded, retrieved_memory_ids, questions_asked,
iterations, llm_calls, llm_seconds, wall_seconds}`. Normalising at the
*strategy* boundary is deliberate — `eval/run_interactive_benchmark.py`'s
scorer has to branch on which mode produced a result in order to read its
sources, and that branch is a wart this runner does not reproduce.

### 1. `vector` — the conventional baseline

Pure pgvector cosine top-k: embed the query, take the nearest `status='ready'`
units, keep those at or above `VECTOR_SIM_THRESHOLD`, truncate to
`TOP_K_FOR_ANSWER`, then one `generate_answer` call. No full-text, no
reformulation, no sufficiency reasoning, no graph.

Written as SQL inside the benchmark rather than as a new mode on
`hybrid_search`, for two reasons stated in the code: (a) `hybrid_search` has
no vector-only mode and adding one would mean editing production code purely
to be benchmarked, and (b) its fused score is `0.65*vector + 0.35*normalised-FTS`,
so with FTS zeroed every score would be uniformly scaled by 0.65 and the
relevance bar would mean something different. Here the cosine similarity *is*
the score. The threshold and top-k are imported from the production pipeline
rather than re-picked, so this is the real system's vector half at the real
system's settings — the baseline is not handicapped by a different bar.

It does get `sibling_context`, as all four do: that is evidence *assembly*,
not retrieval, and holding it constant is what keeps the compared variable
the retrieval strategy.

### 2. `semantic_rerank`

The existing one-shot agent, `run_agent()` — `hybrid_search`'s vector+FTS
fusion plus `evaluate_sufficiency`-driven reformulation, bounded to
`MAX_AGENT_ITERATIONS` rounds. No graph, no clarification. Production code,
unmodified.

### 3. `associative`

`automatic_search()` for seeds, then the level-6 engine from
`backend/app/retrieval/associative.py`: `should_expand` →
`expand_with_associations` → `collect_evidence` → `rerank` → `apply_budget`,
answered via `generate_answer` with `format_evidence_block`. Multi-hop, but
**single-shot**: it never asks the user anything.

This mirrors the candidate-building half of `session.py::_run_round` minus
everything user-mediated (negative cues, positive-cue boost, ambiguity
narrowing), specifically so that the measured difference between this and
`adaptive` is clarification and nothing else.

Association-route candidates count toward `retrieved_memory_ids` — reaching
a memory by inference is the whole claim this strategy makes, so it has to
be scored on it. (Those candidates are still capped at `weak` confidence and
labelled as not-directly-matched in the answer prompt; the benchmark does
not change that.)

### 4. `adaptive`

The real `RecallSession` — `start_session()` then `continue_session()` —
driven by a simulated user, capped at `MAX_SIMULATED_TURNS = 5` replies
(same bound as `eval/run_interactive_benchmark.py`).

The simulated user is the same oracle as the precedent: it reverse-maps the
session's generated question against `_ATTRIBUTE_TEMPLATES` to recover which
semantic field was asked about, then answers yes/no by checking the
*actual post-ingestion* semantic metadata of the query's `ground_truth_memory`
via `values_for_field()`. It is not a hand-authored Q&A script, so it keeps
working whatever phrasing `discriminate.py` picks. One addition the precedent
has no equivalent of: on an unanswerable query there is no ground truth, so
the simulated user says it does not know rather than inventing a cue that
would help narrow a search for a memory that is not there.

Retrieved ids come from `session.final_evidence` (the reranked candidate
list, direct + association, that the answer was generated from) so they are
comparable with `associative`'s, falling back to `session.final_sources` on
the paths that set only that.

## Metrics, exactly

Let *A* be the answerable queries and *U* the unanswerable ones.

- **recall@1** — over *A*: fraction where
  `relevant_memories ∩ retrieved_memory_ids[:1] ≠ ∅`.
- **recall@5** — over *A*: same with `retrieved_memory_ids[:5]`.
  `retrieved_memory_ids` is the strategy's own ranked source list mapped to
  corpus ids and deduped, order preserved.
- **answer correctness** — over *A*: fraction where the answer is grounded
  **and** at least one `expect_contains` string appears (case-insensitively)
  in it. An abstention is never correct on an answerable query.
- **false retrieval rate** — over *U*: fraction answered (grounded) instead
  of abstained. **Reported separately and prominently**, because it is the
  only metric that stops "answer everything" from looking like a win: a
  strategy can buy recall and correctness by never abstaining, and it pays
  for it here.
- **abstention rate on answerable queries** — over *A*: fraction where the
  strategy abstained. The mirror image of the above; together they say
  whether a strategy's grounding decision is calibrated or just biased.
- **mean clarifying questions** — over all queries. Structurally 0 for the
  first three strategies; only `adaptive` can be non-zero.
- **mean iterations** — over all queries. For `adaptive` this is the
  session's own `round_count`. For `vector` it is 1 by construction. For
  `semantic_rerank` and `associative` it is `1 + (number of
  evaluate_sufficiency calls)`, **inferred from the LLM call log** rather
  than reported by the retrieval code, which has no iteration counter to
  read. That inference is exact for the current `automatic_search` loop
  (one sufficiency call per non-final round) but would silently drift if
  that loop gained another sufficiency call per round.
- **mean LLM calls / LLM seconds / wall seconds** — over all queries, from a
  `TimingLLM` decorator wrapping the real provider per strategy invocation
  (the `eval/benchmark_recall_latency.py` pattern). `llm_seconds` is summed
  per-call latency; `wall_seconds` includes the SQL and Python work too.
  Call kind is inferred from the response schema's properties, which is
  enough to separate sufficiency reasoning, cue extraction, answering, and
  embedding.
- **per-category breakdown** — correctness and recall@5 per category; for
  `unanswerable`, correct-abstention rate instead.
- **pairwise win sets** — for every ordered pair of strategies, the query
  ids where the first succeeded and the second failed, where "succeeded"
  means *correct answer* on an answerable query and *correct abstention* on
  an unanswerable one. This is the most informative construct in the
  existing report (docs/REPORT.md): an aggregate delta of a few points says
  nothing about whether two strategies fail on the same queries or on
  different ones, and only the win sets distinguish "strictly better" from
  "differently wrong".

`--repeat N` runs the whole query set N times against the same cached
corpus and reports the mean plus the observed `[min-max]` range per metric,
and `(k/N)` counts in the win sets. That range is the sampling-noise floor
for this corpus and model. docs/TODO.md calls out that no existing script
can establish it; with `gemma4:12b`'s non-determinism, a one- or two-case
flip on a 35–40 query set is plausible with no code change at all. **A
between-strategy gap smaller than the within-strategy range is not a
finding.** A single run cannot distinguish a real difference from noise, and
the report says so in its own output.

Everything above is also written to `eval/strategy_benchmark_results.json`
(per-query, per-strategy, per-repeat, including the answer text and the
clarifying questions actually asked) so a later run can be diffed against an
earlier one rather than re-read from a terminal scrollback.

## Caveats

In the project's established style (docs/DECISIONS.md, docs/RECALL_LATENCY.md):
these are the reasons not to over-read the numbers this produces.

- **The corpus was authored by the same system being evaluated.** Every
  memory, every query, and every `expect_contains` token in
  `eval/memory_corpus.yaml` / `eval/strategy_queries.yaml` was written by an
  LLM in the course of building this benchmark. The ambiguous pairs, the
  multi-hop chains and the lexical distractors are therefore *the hard cases
  this system's author could imagine*, not a sample of the hard cases a real
  user's notes contain. A strategy could score well here by being good at
  synthetic structure. This is the single largest caveat and it is not
  fixable by running the benchmark more times.
- **`expect_contains` is a substring proxy for correctness, not a judge.**
  It will mark a correct answer wrong for using a synonym the list does not
  contain, and a wrong answer right for naming the expected token in
  passing ("you have no memory about Thamesreach Instruments" contains
  "thamesreach"). It was chosen over an LLM judge because a judge would add
  its own unmeasured error and a second model's bias to every number here;
  the trade is that individual per-query verdicts should be read from the
  JSON's answer text, not trusted blindly.
- **A single run cannot distinguish a real difference from noise.** See
  `--repeat` above. Treat any one-run ordering of the four strategies as a
  hypothesis.
- **The simulated user is an upper bound on a real user's helpfulness.** It
  answers from the ground-truth memory's real post-ingestion semantic
  fields, so it is always consistent, never wrong, never vague, and never
  misremembers. A real person being asked "Did this involve Priyanka Venn?"
  about a memory they are struggling to recall may well not know. Whatever
  `adaptive` gains from clarification here is the ceiling, not the expected
  value.
- **Clarification at ingestion time is suppressed.** The corpus is seeded
  with `allow_clarification=False`. A unit parked in
  `awaiting_clarification` is excluded from `hybrid_search` by the status
  gate, which would silently remove part of the corpus from every
  strategy's reach for reasons unrelated to retrieval. This keeps it a
  retrieval benchmark, but it is a real deviation from production
  ingestion: in production some of these memories would have been enriched
  by a user's answer before ever being searched.
- **`vector` is not production code.** It is a baseline assembled in the
  benchmark; production has no reason to offer a non-fusing, non-iterating
  search path. Same disclaimer as mode A in
  `eval/run_interactive_benchmark.py`.
- **The whole corpus shares one database.** A query belonging to one
  structure can match a memory authored for another (that is the point of
  the distractors), but it also means recall@k is measured against a
  hand-declared `relevant_memories` set that may not be exhaustive — a
  strategy could retrieve a genuinely reasonable memory that nobody listed
  and be scored as a miss. Same class of problem as the precision
  undercount documented for the threshold sweep.
- **`mean iterations` for two strategies is inferred, not measured.** See
  the definition above.
- **Association expansion is occasionally truncated by its own wall-clock
  budget, and that is production behaviour, not a benchmark artefact.**
  `RetrievalBudget.deadline_seconds` is 10s and describes itself as a
  ceiling on "the non-LLM work in one round", but
  `app/retrieval/session.py::_run_round` passes `started_at=round_started` —
  a timestamp taken *before* `automatic_search`'s LLM calls. So whenever the
  search loop needed a reformulation round, the clock is already past 10s and
  `expand_with_associations` returns only what it had found before the first
  deadline check. The benchmark passes `started_at` the same way on purpose,
  so `associative` and `adaptive` are affected identically and the measured
  difference between them stays "clarification". Observed in the `--limit 8`
  run: the budget warning fired on 2 of 8 answerable queries, while the other
  6 expanded 1–5 association candidates normally. On a busier machine (slower
  LLM calls) the suppressed fraction will be higher, which means **a run on a
  contended box systematically understates what association contributes.**
- **recall@k is a lower bound for `adaptive` on some queries.** A session
  still in `needs_user_cue` after the 5-turn simulated cap has snapshotted
  neither `final_evidence` nor `final_sources` — `RecallSession` only records
  candidates when it reaches a decision — so those queries report an empty
  retrieved set even though the session did retrieve candidates each round.
  Re-running retrieval in the benchmark to fill that in would measure
  something other than what the session actually delivered to its user, which
  was nothing, so the lower bound is kept deliberately.
- **One machine, shared with everything else.** `llm_seconds` and
  `wall_seconds` are real measurements of this box at that moment, including
  whatever Ollama's prompt cache was doing (docs/RECALL_LATENCY.md documents
  observing "forcing full prompt re-processing due to lack of cache data"
  under load). The *ratios* between strategies are the finding; the absolute
  seconds are not portable.
- **44 memories is a small store.** Association-graph traversal, the
  relevance threshold's behaviour, and the fused score's absolute scale all
  depend on corpus size. Nothing here predicts behaviour at 10,000 memories.
