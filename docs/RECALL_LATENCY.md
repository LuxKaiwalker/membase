# Recall latency: benchmark and the optimizations that came out of it

Goal: one recall round (`POST /api/recall`, `run_agent()`) should take at
most 10 seconds end to end. This document records the benchmark that
diagnosed where the time was going, and the four optimizations it led
to -- all four are now implemented and live in
`backend/app/retrieval/agent.py` and `app/retrieval/pipeline.py`; see
docs/DECISIONS.md "Deterministic sufficiency short-circuit", "Reduce
automatic retrieval iterations", and "Dedupe query embeddings" for the
decision record of each.

## Benchmark

`eval/benchmark_recall_latency.py` (run via
`scripts/benchmark-recall-latency.sh`, same isolated `membase_eval`
database convention as the other eval scripts) ingests a shared corpus
from `eval/cases.yaml` once, then runs `run_agent()` for each case's
query with the real model (`gemma4:12b` / `nomic-embed-text`), wrapping
the `LLMProvider` to record `(call_kind, seconds)` for every individual
`generate`/`generate_structured`/`embed` call so the time can be
attributed, not just totaled.

### Results (8 cases from `eval/cases.yaml`, measured against the pre-optimization baseline: `MAX_AGENT_ITERATIONS=3`, `MAX_REFORMULATIONS_PER_ROUND=2`, no sufficiency short-circuit, no embed dedupe)

| case | total | LLM calls | iterations |
|---|---|---|---|
| exact_lookup | 8.48s | 4 | 1 |
| paraphrase_japan_cuisine | 24.05s | 13 | 3 (exhausted) |
| paraphrase_work_location | 6.83s | 4 | 1 |
| indirect_eth_person | 9.22s | 4 | 1 |
| indirect_pet_name | 6.17s | 4 | 1 |
| multiconcept_anna_tactile_paper | 16.64s | 13 | 3 (exhausted) |
| multiconcept_combo | 8.45s | 4 | 1 |
| distractor_heavy_paris | 17.15s | 13 | 3 (exhausted) |

```
n = 8
mean   = 12.12s
median =  8.85s
p90    = 24.05s
max    = 24.05s
min    =  6.17s
rounds over 10s = 3/8
```

**Time by call kind:**

| kind | total | calls | avg/call | share of LLM time |
|---|---|---|---|---|
| `evaluate_sufficiency` | 56.37s | 11 | 5.12s | 58% |
| `generate_answer` | 38.99s | 8 | 4.87s | 40% |
| `embed` | 1.54s | 40 | 0.04s | 2% |

### The one finding that mattered most

**Rounds that converge in a single iteration already met the goal.** The
5 cases that got a "sufficient" verdict on the first try averaged
**7.83s** (6.17-9.22s) -- comfortably under 10s. The 3 cases that never
got a "sufficient" verdict and ran all `MAX_AGENT_ITERATIONS=3` rounds
averaged **19.28s** (16.64-24.05s). There was no middle ground in this
sample: a round was either clearly fast or clearly slow, split exactly
along whether `evaluate_sufficiency` said "sufficient" the first time it
was asked.

That reframed the problem from "make everything faster" to the narrower
"stop `evaluate_sufficiency` from being asked at all when the answer is
obviously going to be yes, and/or cut the cost of the cases where it's
genuinely still searching" -- which is exactly what optimization 1 below
does.

## Where the time was going (call-stack view)

`run_agent()` → `automatic_search()` → per iteration: for each query
(seed query + reformulations), `hybrid_search()` *and*
`hybrid_search_chunks()` each independently called `llm.embed([query])`
for the exact same query string -- then, unless it was the last
iteration, `evaluate_sufficiency()` (one structured-output call). After
the loop, `generate_answer()` (one structured-output call) always runs
once.

Call-count arithmetic, confirmed by the benchmark (under the
pre-optimization 3-iteration/2-reformulation budget):

- **Converges immediately** (5/8 cases): 2 embeds (unit + chunk search
  for the one seed query) + 1 `evaluate_sufficiency` + 1
  `generate_answer` = **4 calls**.
- **Exhausts all 3 iterations** (3/8 cases): iteration 0 (2 embeds +
  eval) + iteration 1, up to 2 reformulated queries (4 embeds + eval) +
  iteration 2, last, up to 2 more reformulated queries (4 embeds, no
  eval since it's the last iteration) + 1 `generate_answer` = **13
  calls**.

`embed` was cheap (40ms average) regardless of call count -- the two
structured-output calls (`evaluate_sufficiency`, `generate_answer`) were
where nearly all the real time went.

## The four optimizations (all implemented)

### 1. Skip `evaluate_sufficiency` when the top result is clearly good enough

The biggest lever, since it removes one ~5s call from the majority case
unconditionally. Implemented as `_is_high_confidence()` in
`app/retrieval/agent.py`: skips the `evaluate_sufficiency()` LLM call
when the top-ranked candidate's `fused_score` clears
`HIGH_CONFIDENCE_FUSED_SCORE = 0.80`, its margin over the runner-up (if
any) clears `HIGH_CONFIDENCE_MARGIN = 0.20`, and its semantic analysis
isn't a degraded fallback. See docs/DECISIONS.md "Deterministic
sufficiency short-circuit" for the full reasoning and why this can only
ever fire on the confident side (never skips a genuinely-needed
reformulation).

### 2. Reduce `MAX_AGENT_ITERATIONS` from 3 to 2

Implemented -- `app/retrieval/agent.py`'s `MAX_AGENT_ITERATIONS = 2`.
See docs/DECISIONS.md "Reduce automatic retrieval iterations" for the
measured quality effect (28/30 vs. 30/30 on the eval set, judged an
acceptable trade-off, not reverted).

### 3. Reduce `MAX_REFORMULATIONS_PER_ROUND` from 2 to 1

Implemented alongside optimization 2 -- `app/retrieval/agent.py`'s
`MAX_REFORMULATIONS_PER_ROUND = 1`. Halves the embed-call count in
non-first iterations; since embed was only ~2% of total time this alone
barely moved wall-clock time, but it compounds with optimization 2's
iteration-count reduction.

### 4. Deduplicate the two `embed()` calls per query

Implemented via `embed_query()` and an optional `query_vector` parameter
on `hybrid_search()`/`hybrid_search_chunks()` (`app/retrieval/
pipeline.py`); `automatic_search()` embeds each query string once per
iteration and passes the resulting vector into both calls. See
docs/DECISIONS.md "Dedupe query embeddings". A clean, zero-trade-off
correctness improvement -- pure removed redundancy, not a quality
trade-off like 1-3.

### Not available: a faster second model

Investigated during the Gemma 4 migration (see docs/DECISIONS.md "Gemma
4 migration"): this GPU's 11GB VRAM has no realistic room for a second
generation model alongside `gemma4:12b` (measured ~8.7GB) and
`nomic-embed-text` (~0.25GB) without risking the eviction/reload
thrashing already documented. Model-level speedup isn't on the table
without different hardware.

## Caveats on the benchmark numbers above

- 8 cases, one run each, on a machine shared with everything else that
  session was doing -- not a large enough or repeated-enough sample to
  treat the exact seconds as precise, only as directionally real (the
  1-iteration vs. 3-iteration split is the actual finding, not the
  specific "7.83s" and "19.28s" averages).
- Ollama's own prompt cache was under real pressure from other activity
  on the same machine at the time (observed via `journalctl -u ollama`:
  repeated "forcing full prompt re-processing due to lack of cache
  data"), which adds variable overhead per call unrelated to this app's
  own code. A clean-machine re-run would likely show lower, more
  consistent numbers across the board.
- These numbers predate optimizations 1-4 above -- they describe the
  problem the optimizations were built to fix, not the current
  post-optimization latency. A fresh benchmark run
  (`scripts/benchmark-recall-latency.sh`) against the current code would
  measure the actual current numbers; not re-run for this document.
