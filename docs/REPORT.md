# Engineering report: clarification-gating and retrieval-hardening phase

This supersedes the previous report. Scope: this phase corrected the
single most important architectural gap identified in review --
clarification was enriching memories but not gating their visibility --
and hardened retrieval (attachment chunking, a calibrated threshold,
deletion, correction) around the memory-unit architecture built in the
prior phase. Verified against the real local LLM throughout, including
all 6 of the spec's final acceptance tests.

## Current architecture

```
human capture
      v
one or more memory units (LLM-decided segmentation, verbatim-validated)
      v
semantic enrichment (meaning, aliases, entities, definitions, relations)
      v
optional bounded clarification -- GATES recall until resolved/discarded
      v
durable authoritative memory (two-phase commit: text saved before any
      LLM call in the derived stage runs)
      v
derived retrieval representations (retrieval_text, embedding, fts;
      attachments additionally chunked + independently indexed)
      v
hybrid retrieval (vector + full-text, over units AND attachment chunks)
      v
bounded local retrieval agent (search -> evaluate -> reformulate, <=3 rounds)
      v
grounded answer (hard abstention override in exactly one place)
```

User corrections (`POST /memory-units/{id}/correct`) and explicit
deletion (`DELETE /api/captures/{id}`) are the two sanctioned ways a
memory changes or disappears after creation; nothing else does either
automatically.

## Architectural changes this phase

- **Clarification gate**: `status` gained `awaiting_clarification` and
  `discarded`. A unit with an open question is durable but excluded from
  `hybrid_search()`'s `status='ready'` filter until answered or
  discarded. This reverses the prior phase's "enrich but don't gate"
  decision -- see docs/DECISIONS.md for why, including why the prior
  reasoning is kept rather than deleted.
- **Attachment chunking**: new `attachment_chunks` table, page-based for
  PDFs, fixed-window+overlap otherwise, each chunk independently
  full-text and vector indexed. The agent now searches chunks alongside
  memory units every round; answers can cite a specific attachment +
  page rather than an undifferentiated excerpt.
- **Post-hoc attachments now update semantic retrieval**, not just
  full-text -- a real gap in the prior phase (spec section 28 is explicit
  about this) fixed by re-embedding affected units from a retrieval_text
  that includes a fresh attachment excerpt.
- **Re-embedding/backfill tool** (`scripts/reembed.py`): finds
  missing/stale-model embeddings across both units and chunks, backfills
  idempotently, refuses to run if the configured embedding dimension
  doesn't match the schema's fixed `vector(768)` column.
- **Deletion and correction**: `DELETE /api/captures/{id}` (cascades DB
  rows + removes files); `POST /memory-units/{id}/correct` (free-text,
  reuses the clarification-enrichment mechanism, doesn't consume the
  bounded clarification-round budget).
- **Retrieval threshold calibration**: `scripts/sweep-threshold.sh`
  swept `VECTOR_SIM_THRESHOLD` against the (now 30-case) eval set;
  confirmed 0.55 as the highest value with 100% recall@10, see
  docs/DECISIONS.md for the full numbers and caveats.

## Local LLM setup

Unchanged from the prior phase: Ollama system service at
`127.0.0.1:11434`; `qwen2.5:14b-instruct-q4_K_M` for every
generation/structured-output call; `nomic-embed-text` (768 dims) for
embeddings. `scripts/model-doctor.sh` verifies both are present and
working; safe to rerun.

## Retrieval evaluation

`eval/cases.yaml` grew from 12 to 30 cases, adding categories the spec
explicitly called out: alias/definition (direct, not just
clarification-derived), entity-mediated ("what did Anna tell me"),
ambiguous (two candidates, must ground in a real one rather than
abstain or conflate), relation (forward and reverse), and more
no-answer/distractor variants. **30/30 passing.**

`scripts/sweep-threshold.sh` ingested all 30 cases into one shared DB and
swept `VECTOR_SIM_THRESHOLD` from 0.35 to 0.70:

| threshold | recall@10 | precision* |
|---|---|---|
| 0.35-0.55 | 100% | 13-19% |
| 0.60 | 96% | 25% |
| 0.65 | 89% | 33% |
| 0.70 | 78% | 34% |

*(precision here undercounts -- see docs/DECISIONS.md caveat on
cross-case contamination in a shared corpus.)* 0.55 kept as the value
that gives up no measured recall while maximizing precision among
candidates that don't.

Representative failure found and fixed during the *previous* phase's
eval work (not reintroduced here): a coherent-episode segmentation split
left a dangling pronoun ("she recommended a paper") whose antecedent was
in a sibling unit; fixed by instructing the answer-generation prompt to
resolve references using sibling context. No new unfixed failures
surfaced in this phase's 30-case run.

## Tests

55 pytest tests (up from 40), all passing: the prior phase's 40 plus new
coverage for the clarification gate (awaiting-state durability, discard,
gated->searchable transition), chunking (pagination, plain-text
splitting, embedding, post-hoc indexing), deletion (DB row + file
removal), correction (semantic representation actually changes, doesn't
consume the clarification budget), and the backfill tool (run as a real
subprocess against a live DB, since it deliberately always uses the real
configured LLM provider -- marked `integration`, skipped if Ollama is
unreachable like the other real-model tests).

## Remaining limitations

- Segmentation's verbatim check has no linguistic awareness of pronoun/
  reference chains (see docs/TODO.md) -- mitigated, not eliminated.
- Clarification triggering is still inconsistent for borderline terms,
  and now that clarification gates recall, a spurious question has a
  real UX cost (the memory is invisible until answered/discarded).
- Chunk size (1200 chars) is untuned against a genuinely large document;
  the eval set's attachments are all short.
- Agent worst-case latency roughly doubled (chunks searched alongside
  units every round) -- not measured under load, only observed to
  resolve in round 1 for all 30 eval cases in practice.
- Retrieval threshold calibration used 30 cases in one shared corpus;
  real per-user data at larger scale could shift the picture.

## Deferred work

Multi-turn conversation, autonomous background agents, general
knowledge-graph storage, memory consolidation/forgetting, multimodal
understanding, multi-user auth, a trained/learned reranker, a web UI
for browsing/deleting individual captures (API supports it, UI doesn't
expose it yet).

## Open product questions

Whether the retrieval threshold should become a relative/adaptive rule
instead of a fixed scalar as the corpus grows; whether
`MAX_CLARIFICATION_ROUNDS` should ever exceed 1 given `correct_unit()`
already offers an unbounded user-initiated path to the same effect. See
docs/TODO.md for the full list.

## Architectural risks

1. **Clarification gating raises the stakes of a bad question.** Before,
   a spurious clarification was just noise appended to a log; now it
   makes a memory temporarily unsearchable. The system's bias toward
   *not* asking (see enrichment prompt) is doing more load-bearing work
   than before.
2. **Chunking + per-round dual search (units + chunks) is the most
   latency-sensitive part of the system** and the least load-tested --
   fine at 30 eval cases, unverified at real personal-memory-store scale
   (hundreds of memories, dozens of attachments).
3. **Single generation model dependency** (unchanged risk from the prior
   phase): all judgment quality rests on one 14B local model.

## Recommended next experiment

Load-test the agent's latency with a synthetic corpus of ~200 memory
units and ~20 chunked attachments (vs. today's ~30-unit eval corpus) to
see whether the dual unit+chunk search per round, times up to 3 rounds,
stays within an acceptable interactive latency budget -- this is the
risk most likely to surface as "feels slow" before it surfaces as an
accuracy problem.
