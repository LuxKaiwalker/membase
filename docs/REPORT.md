# Engineering report: memory-unit architecture phase

This supersedes the previous report. Status: the core architectural
correction (capture/memory-unit decomposition, clarification-as-semantic-
enrichment, bounded retrieval agent) is implemented and verified against
the real local LLM, including the spec's acceptance tests. Not
production-ready -- see "Known limitations" and "Architectural risks."

## What changed from the previous MVP

The previous MVP stored one `Memory` row per submission with a flat
`derived` metadata blob and a single clarification question. This phase
replaced that with:

- **`Capture` + `MemoryUnit`**, decoupling "what the user submitted" from
  "what's independently retrievable," with an LLM-driven, verbatim-
  validated segmentation step deciding the split per capture.
- **A real semantic representation per unit** (`semantic` jsonb: meaning,
  aliases, entities, topics, definitions, relations) instead of a flat
  people/location/topics dict, plus a deterministically-assembled
  `retrieval_text` that's what actually gets embedded and full-text
  indexed -- aliases and definitions are retrieval cues now, not just
  display metadata.
- **Clarification as semantic enrichment**: an answer re-triggers the
  full enrichment pass (not just an appended text blob), so a clarified
  term's aliases become genuinely embedded/indexed, not just present in
  one past answer.
- **A bounded retrieval agent** (`app/retrieval/agent.py`) replacing a
  single-shot hybrid search: up to 3 rounds of search, with an LLM
  deciding whether to reformulate the query using likely synonyms/aliases.
- **A two-phase-commit ingestion pipeline** with an explicit `status`
  (`draft`/`ready`/`failed`) so a bug in derived processing can never
  silently lose or corrupt the user's original input.
- **Sibling/context expansion**: a primary hit's answer can draw on
  other units from the same capture for context (e.g. resolving a
  pronoun), bounded and clearly distinguished from primary evidence.

## Local LLM setup

- Ollama: system service, already running at `127.0.0.1:11434`.
- Generation model: `qwen2.5:14b-instruct-q4_K_M`, used for *every*
  generation/structured-output call (segmentation, enrichment, agent
  evaluation, answers). Chosen over the originally-suggested
  `qwen2.5:7b-instruct` after direct testing showed 7b reliably
  mis-segmenting coherent episodes and never asking a clarification
  question even for genuinely invented terms (always fabricated a
  confident-sounding definition instead). Using one model everywhere
  also avoids a measured ~7-14s Ollama model-swap stall that occurs if a
  second generation model is mixed in on this GPU's 11GB VRAM budget.
- Embedding model: `nomic-embed-text`, 768 dims, unchanged.
- Setup/diagnostic: `scripts/model-doctor.sh` -- checks reachability,
  pulls missing models, verifies generation and embedding calls actually
  work, and flags an embedding-dimension mismatch against the schema.
  Safe to rerun.

## Evaluation

`eval/cases.yaml`: 12 hand-written cases against the real model, covering
exact, paraphrased, indirect, multi-concept, distractor, no-answer
(carried over from the previous phase) plus new categories this phase
specifically targets: segmentation (compound-facts vs. coherent-episode),
clarification-driven alias retrieval (using an invented term, "Zorblex
code," specifically so the model cannot fall back on world knowledge),
user-specific terminology, relation retrieval, and cross-memory retrieval
across separate captures. **12/12 passing** as of this report
(`scripts/run-eval.sh`).

One real failure was found and fixed during this work, not just assumed
away: the coherent-episode segmentation case initially split "I met Anna
at ETH... and she recommended a paper" into two verbatim-valid halves,
and the answer came back as "she" instead of "Anna" -- the sibling
context *was* retrieved and included in the evidence, but the
answer-generation prompt didn't instruct the model to actually use it to
resolve the pronoun. Fixed by strengthening that prompt; confirmed fixed
by rerunning the eval case, not just by inspection.

## Tests

40 pytest tests, all passing: 35 fast (FakeLLMProvider, dispatches by
schema shape into segmentation/enrichment/evaluate/answer canned
responses) covering storage, segmentation (verbatim fallback, cap,
compound vs. coherent end-to-end), clarification (question generation,
answer persistence, re-enrichment, bounded rounds, alias retrieval via
fakes), retrieval (exact/distractor/sibling-context), the agent (direct
search, reformulation, iteration limit, malformed/unavailable LLM,
no-answer), grounding override, and two-phase-commit failure/retry
(via `monkeypatch` forcing an unexpected exception mid-enrichment). 5
integration tests against the real Ollama service, including the
spec's own acceptance-test shape (clarification -> alias -> retrieval,
using the invented "Zorblex code" term specifically so the test can't
pass via world knowledge) and the segmentation acceptance test
(compound facts split, coherent episode doesn't).

## Remaining limitations

- Segmentation's verbatim-fallback check has no linguistic awareness of
  pronoun/reference chains -- it only prevents paraphrasing, not
  reference-breaking splits. Mitigated at the answer-generation layer
  (see above), not eliminated at the source.
- Clarification question triggering is still inconsistent for borderline
  cases across different runs of the same input.
- No image understanding; no attachment chunking for large documents; no
  re-embedding/backfill tool for stale embeddings.
- Retrieval threshold (0.55 cosine similarity cutoff) remains a
  hand-picked value, not swept/tuned against a large eval set.
- Agent worst-case latency (~7 search calls + up to 4 LLM structured
  calls per recall) is unbounded-feeling even though iteration-bounded;
  not measured under load.

## Deferred work

Multi-turn conversation, autonomous background agents, general
knowledge-graph storage, memory consolidation/forgetting, multimodal
understanding, multi-user auth, a trained/learned reranker, deletion UI,
post-creation metadata correction UI. See docs/TODO.md for the complete,
categorized list -- several of these (deletion, correction, re-embedding)
are flagged as likely near-term work, not permanently out of scope.

## Open product questions

Whether clarification should gate recall availability (current: it
doesn't) is the most consequential one -- see docs/DECISIONS.md for the
reasoning and docs/TODO.md for what would need to change to flip it.

## Architectural risks

1. **Retrieval threshold calibration** remains the single scalar most
   likely to need rework with more usage data -- unchanged risk from the
   previous phase, now compounded by the fact that `retrieval_text` (the
   embedding input) is richer/longer than before, which may shift the
   unrelated-pair similarity baseline that the 0.55 cutoff was picked
   against.
2. **Segmentation quality is now load-bearing** in a way it wasn't
   before: a bad split can break reference resolution (see the fix
   above) or merge things that shouldn't be merged. The verbatim check
   is a safety net against fabrication, not a quality guarantee.
3. **Single generation model dependency**: the whole pipeline's
   judgment quality (segmentation, clarification triggering, grounding)
   rests on one 14B local model's instruction-following. No fallback or
   ensemble; a future smaller/faster model swap would need the same kind
   of manual A/B testing documented in docs/DECISIONS.md, not just a
   config change.

## Recommended next experiment

Expand `eval/cases.yaml` to ~30-50 cases specifically targeting
segmentation/reference-resolution edge cases (the exact failure mode
found and fixed this phase) and use it to decide whether the verbatim-
fallback mechanism needs a second, explicit check for dangling pronouns/
references, rather than relying on the answer-generation prompt to paper
over it after the fact.
