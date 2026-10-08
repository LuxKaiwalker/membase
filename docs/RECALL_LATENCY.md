# Recall latency: benchmark and options to reach <10s/round

Goal (user-set): one recall round (`POST /api/recall`, `run_agent()`)
should take at most 10 seconds end to end.

## Benchmark

New script, `eval/benchmark_recall_latency.py` (run via
`scripts/benchmark-recall-latency.sh`, same isolated `membase_eval`
database convention as the other eval scripts). It ingests a shared
corpus from `eval/cases.yaml` once, then runs `run_agent()` for each
case's query with the real model (`gemma4:12b` / `nomic-embed-text`),
wrapping the `LLMProvider` to record `(call_kind, seconds)` for every
individual `generate`/`generate_structured`/`embed` call so the time
can be attributed, not just totaled.

Building this required fixing a real, pre-existing gap first:
`eval/run_eval.py`, `eval/run_session_eval.py`,
`eval/run_interactive_benchmark.py`, and `eval/threshold_sweep.py` all
called `create_capture()` directly and then either queried units
immediately or ran retrieval against them -- none of them drove the
now-backgrounded enrichment step to completion first (see
docs/DECISIONS.md "offloading to background processing" and the
earlier gemma4-migration work). Since `hybrid_search()` only returns
`status="ready"` units, every one of these scripts had been silently
broken (0% recall, every case failing) since that architecture
change landed earlier in this project's history, and nobody had re-run
them since. Fixed by adding `eval/_helpers.py` (`finish_processing()`,
mirroring `backend/tests/helpers.py`'s `create_capture_and_finish()`)
and calling it from all four scripts right after each `create_capture()`.

### Results (8 cases from `eval/cases.yaml`, run against the live `gemma4:12b`)

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

**Time by call kind** (corrected after fixing a counting bug in the
benchmark script itself -- the first run's per-kind `calls`/`avg`
numbers were wrong, counting "queries that used this kind at least
once" instead of actual call count; `total` seconds were already
correct, only the denominator was off):

| kind | total | calls | avg/call | share of LLM time |
|---|---|---|---|---|
| `evaluate_sufficiency` | 56.37s | 11 | 5.12s | 58% |
| `generate_answer` | 38.99s | 8 | 4.87s | 40% |
| `embed` | 1.54s | 40 | 0.04s | 2% |

### The one finding that matters most

**Rounds that converge in a single iteration already meet the goal.**
The 5 cases that got a "sufficient" verdict on the first try averaged
**7.83s** (6.17-9.22s) -- comfortably under 10s. The 3 cases that
never got a "sufficient" verdict and ran all `MAX_AGENT_ITERATIONS=3`
rounds averaged **19.28s** (16.64-24.05s). There is no middle ground in
this sample: a round is either clearly fast or clearly slow, split
exactly along whether `evaluate_sufficiency` said "sufficient" the
first time it was asked.

That means the real question isn't "how do we make everything faster,"
it's narrower: **how do we stop `evaluate_sufficiency` from being asked
at all when the answer is obviously going to be yes, and/or cut the
cost of the cases where it's genuinely still searching.**

## Where the time actually goes (call-stack view)

`run_agent()` → `automatic_search()` → per iteration (max 3): for each
query (seed query + up to 2 reformulations), `hybrid_search()` *and*
`hybrid_search_chunks()` each independently call `llm.embed([query])`
for the exact same query string (`app/retrieval/pipeline.py`) -- then,
unless it's the last iteration, `evaluate_sufficiency()` (one
structured-output call). After the loop, `generate_answer()` (one
structured-output call) always runs once.

Call-count arithmetic, confirmed by the benchmark:

- **Converges immediately** (5/8 cases): 2 embeds (unit + chunk search
  for the one seed query) + 1 `evaluate_sufficiency` + 1
  `generate_answer` = **4 calls**.
- **Exhausts all 3 iterations** (3/8 cases): iteration 0 (2 embeds +
  eval) + iteration 1, up to 2 reformulated queries (4 embeds + eval)
  + iteration 2, last, up to 2 more reformulated queries (4 embeds, no
  eval since it's the last iteration) + 1 `generate_answer` = **13
  calls**.

`embed` is cheap (40ms average) regardless of how many times it's
called -- doubling it up (`hybrid_search` and `hybrid_search_chunks`
separately embedding the identical query string) is real, measurable
waste, but at 2% of total LLM time, eliminating it wouldn't move any
of the numbers above in a way that matters for the 10s goal. The two
structured-output calls are where literally all the real time is.

## Options to reach <10s/round, in order of actual impact

### 1. Skip `evaluate_sufficiency` entirely when the top result is clearly good enough (biggest lever)

Add a cheap, deterministic short-circuit before the LLM call: if the
top-ranked candidate's `fused_score` clears a high-confidence threshold
(well above `VECTOR_SIM_THRESHOLD`, picked empirically the same way
that constant was -- see docs/DECISIONS.md's sweep), treat it as
"sufficient" without asking the model at all. This removes one ~5s
call from the *majority* case (5/8 in this sample) unconditionally,
which is the single biggest lever available: it would bring the
"converges immediately" bucket's ~7.8s average down to roughly
~2.5-3s (embed + answer only).

**Trade-off, not free:** `evaluate_sufficiency` isn't pure overhead --
it's also where the agent decides a result is *insufficient* and
reformulates. A score-threshold shortcut only ever fires on the
confident side (never skips a genuinely-needed reformulation), so it
can't make retrieval *worse*, but it does mean the "is this actually
enough evidence" judgment is sometimes made by a hand-picked number
instead of the model's own reasoning -- the same category of judgment
call `VECTOR_SIM_THRESHOLD` already represents, and the project's own
convention (docs/DECISIONS.md) is to calibrate such a threshold against
a real eval set before trusting it, not guess at one. This needs the
same sweep treatment as the retrieval threshold got, not a number
picked from this one benchmark run.

### 2. Reduce `MAX_AGENT_ITERATIONS` from 3 to 2

Cuts the worst case from 13 calls to ~8 (removes one full
reformulation round and its `evaluate_sufficiency` call). Estimated
worst-case saving: roughly 5s (one fewer eval call) off the ~17-24s
tail, landing around 12-19s -- better, but **likely still over 10s**
on the genuinely hard cases. This is a pure quality/latency trade-off
(less chance to find evidence via reformulation) with no deterministic
guardrail softening it, unlike option 1 -- would need the same kind of
empirical quality check `eval/run_eval.py`'s cases give before
shipping, now that that script actually works again.

### 3. Reduce `MAX_REFORMULATIONS_PER_ROUND` from 2 to 1

Halves the embed-call count in non-first iterations (worst case 13 →
9 calls), but since embed is only 2% of total time, this barely moves
wall-clock time -- the `evaluate_sufficiency` call count per iteration
is unchanged. **Not worth it on its own** for this specific goal; only
useful combined with option 1 or 2.

### 4. Deduplicate the two `embed()` calls per query

`hybrid_search()` and `hybrid_search_chunks()` both independently
embed the identical query string. Fixing this (embed once per unique
query string per iteration, reuse the vector for both searches) is a
clean, zero-trade-off correctness improvement -- pure removed
redundancy, not a quality trade-off like options 1-3. Worth doing
regardless, but on these numbers it would save on the order of tens of
milliseconds per round (2% of total time, and only half of that is
actually redundant) -- **not a meaningful lever for the 10-second
goal** by itself.

### 5. Not available: a faster second model

Already investigated during the Gemma 4 migration (see
docs/DECISIONS.md): this GPU's 11GB VRAM has no realistic room for a
second generation model alongside `gemma4:12b` (measured ~8.7GB) and
`nomic-embed-text` (~0.25GB) without risking the eviction/reload
thrashing already documented. Model-level speedup isn't on the table
without different hardware.

## Recommendation

Implement option 1 (deterministic sufficiency short-circuit on a high
score) as the primary fix -- it's the only option in this list that
helps the common case without a latency/quality trade-off that needs
weighing, and the data says the common case is most of the problem.
Pair it with option 4 (dedupe the redundant embed) since it's free and
correct regardless. Leave options 2/3 (lowering the iteration/
reformulation bounds) as a secondary lever specifically for the
genuinely-ambiguous tail that option 1 can't help -- but treat that as
a deliberate quality trade-off requiring the same kind of before/after
check against `eval/run_eval.py`'s cases that every other retrieval
tuning decision in this project got, not something to ship on this
benchmark's 8 cases alone.

None of the above is implemented yet -- this is the analysis and
options the user asked for; implementing is a separate step.

## Caveats on this benchmark

- 8 cases, one run each, on a machine shared with everything else this
  session was doing -- not a large enough or repeated-enough sample to
  treat these exact seconds as precise, only as directionally real
  (the 1-iteration vs. 3-iteration split is the actual finding, not
  the specific "7.83s" and "19.28s" averages).
- Ollama's own prompt-cache was under real pressure from this session's
  broader activity (observed directly via `journalctl -u ollama`:
  repeated "forcing full prompt re-processing due to lack of cache
  data"), which adds variable overhead per call unrelated to anything
  in this app's own code. A clean-machine re-run would likely show
  lower, more consistent numbers across the board.
- The full 30-case `eval/cases.yaml` set was not used for this specific
  latency run (it was for validating the `eval/_helpers.py` fix, which
  independently took over 40 minutes against this same loaded machine
  and was stopped rather than waited out) -- 8 cases were enough to see
  the 1-iteration/3-iteration split clearly, but a larger sample would
  give tighter percentile estimates.
