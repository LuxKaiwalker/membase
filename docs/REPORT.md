# Final engineering report

Status: working MVP. The full loop (store -> optionally clarify -> later
ask an indirect question -> get a grounded answer with sources -> restart
without losing data) was verified end-to-end against the real local LLM,
not just unit-tested. Not production-ready -- see "Architectural risks."

## Implemented

- Web UI (Jinja2 + vanilla JS) for memory creation (with clarification
  flow) and recall, talking only to a JSON API.
- `POST /memories`, `GET /memories/{id}`, `POST /memories/{id}/clarify`,
  `POST /memories/{id}/attachments`, `POST /recall`.
- File attachments: PDF/TXT/MD with text extraction, images
  stored+validated (no captioning). Real content sniffed from bytes;
  wrong/dangerous types rejected at upload.
- Automatic metadata extraction (people/location/topics/event_date/
  summary) via local LLM structured output, with a simple one-question
  clarification mechanism that enriches but never blocks persistence.
- Embeddings (nomic-embed-text, 768d) + Postgres full-text search,
  fused by a weighted-score rerank.
- Grounded answer generation that deterministically abstains
  ("I don't have a stored memory about this.") when evidence doesn't
  support an answer, with full provenance (which memories were used).
- `LLMProvider` abstraction with one implementation (Ollama); ingestion
  and retrieval degrade gracefully (never crash) when the LLM is down,
  times out, or returns malformed output.
- Alembic-migrated Postgres schema distinguishing authoritative data
  (caption, files) from derived data (metadata, embeddings, search index).
- 19 pytest tests (fast fakes for storage/failure-mode logic + real-LLM
  integration tests for retrieval/grounding quality) and a 6-case manual
  eval set covering exact/paraphrased/indirect/multi-concept/distractor/
  no-answer -- all passing as of this report.
- `flake.nix` devShell (Python entirely from nixpkgs, no pip; project-local
  Postgres+pgvector) and `CLAUDE.md` documenting the NixOS-specific
  footguns hit while building this.

## Partially implemented

- **Clarification quality**: works, but the 7B generation model
  occasionally asks a spurious or mistargeted question (see docs/TODO.md).
  Low severity since it's purely additive.
- **Attachments added after creation** (`POST /memories/{id}/attachments`)
  extend the full-text index but don't re-embed or re-run metadata
  extraction -- a scope cut, not a bug (see docs/DECISIONS.md).
- **Metadata extraction** is reasonably reliable for people/topics/
  event_date but inconsistent for `location` in manual testing (sometimes
  left blank when the text clearly names one). Doesn't block retrieval
  since full-text/embedding search index the raw caption regardless.

## Not implemented (deferred, per spec section 3)

Multi-turn conversational memory, autonomous background agents, knowledge
graphs, memory consolidation/forgetting, multi-agent systems, advanced
temporal reasoning, image understanding (captioning/OCR), cloud
deployment, mobile apps, multi-user auth, a reranker model, an
embedding-model backfill/reindex tool, per-chunk attachment retrieval.

## Known limitations

- **Retrieval threshold is unvalidated.** `VECTOR_SIM_THRESHOLD = 0.55`
  came from a handful of manual trials against one embedding model
  (nomic-embed-text), not a tuned value against a real eval set at scale.
- **No re-embedding path.** A memory whose embedding failed at creation
  time (LLM was down), or that predates a `MEMBASE_EMBED_MODEL` change,
  stays full-text-searchable but not vector-searchable until someone
  writes a backfill job.
- **Small-model grounding is imperfect.** The 7B model occasionally gives
  an over-literal "the evidence doesn't say X" (slightly too conservative,
  since tuned) or an overly verbose hedge; the prompt was iterated against
  manual cases but not against a large held-out set.
- **No image/vision model wired in** -- images are stored and validated
  but contribute nothing to search.
- **UI click-tested via API+rendered-HTML inspection, not a live browser
  session** -- no browser automation tool was available in the build
  environment. The JS logic is simple DOM/fetch code traced by hand
  against the verified API contract.
- **Single-user, localhost-only, no auth** -- appropriate for the MVP's
  privacy goals, not for any shared/networked deployment.

## Open product questions

See docs/TODO.md "Open product questions" for the full list:
whether clarification should gate persistence, whether metadata should be
user-editable, whether attachments should be independently searchable,
whether memories should ever expire/be forgotten.

## Architectural risks

1. **Retrieval threshold calibration.** The single scalar cutoff on
   cosine similarity is the most likely thing to need rework once real
   usage data exists -- it was picked from ~10 manual query/memory pairs.
   A relative/adaptive threshold (margin below the top score, or a
   learned calibration) is the likely fix; needs a bigger eval set first.
2. **Single embedding model, no versioning migration path.** The schema
   stores `embedding_model`/`embedding_dim` per row specifically so a
   future model change is *detectable*, but there's no tool that acts on
   that yet. Not urgent at MVP scale, but would bite the first time
   someone actually changes `MEMBASE_EMBED_MODEL` with existing data.
3. **Small local generation model's judgment calls** (what counts as
   "grounded," when to ask a clarification question) are the least
   predictable part of the system and the hardest to regression-test
   without a much larger eval set. The current eval set (6 cases) is
   enough to catch gross regressions, not subtle ones.

## Suggested next steps

1. **Build a larger eval set and use it to calibrate the retrieval
   threshold.**
   - *Problem:* the 0.55 cutoff is a guess from a handful of trials.
   - *Current state:* 6 eval cases, all passing, but not enough to tune
     a numeric threshold responsibly.
   - *Why it matters:* this threshold directly controls the
     recall/precision tradeoff for the project's core value proposition
     (indirect/associative retrieval).
   - *Possible approaches:* expand eval/cases.yaml to 30-50 cases
     covering more paraphrase distance; sweep the threshold against it;
     consider a relative-margin rule instead of an absolute cutoff.
   - *Recommended next investigation:* start by writing 10 more
     paraphrased/indirect cases from real-ish personal memories and
     seeing where the current threshold already fails.

2. **Decide whether clarification should gate persistence.**
   - *Problem:* current behavior (persist immediately, enrich later) is
     a deliberate scope call, but diverges from the spec's own diagram.
   - *Current state:* implemented as "enrich-after" (docs/DECISIONS.md).
   - *Why it matters:* affects the UI's perceived flow and whether
     "pending" memories should be visually distinguished.
   - *Possible approaches:* keep as-is; or add a lightweight `status`
     field (`complete`/`awaiting_clarification`) surfaced in the UI
     without actually blocking persistence.
   - *Recommended next investigation:* user feedback on whether the
     current (silent, non-blocking) clarification flow feels right.

3. **A re-embedding/backfill tool.**
   - *Problem:* no path to fix memories with missing/stale embeddings.
   - *Current state:* detectable (`embedding_model`/`embedding_dim`
     columns) but not actionable.
   - *Why it matters:* becomes necessary the first time the embedding
     model changes or Ollama was down during a batch of memory creation.
   - *Possible approaches:* a `scripts/reembed.py` that finds rows where
     `embedding IS NULL OR embedding_model != current_model` and re-runs
     just the embedding step.
   - *Recommended next investigation:* low effort, could be done
     immediately after step 1; not urgent before then.
