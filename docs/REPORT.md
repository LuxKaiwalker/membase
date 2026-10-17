# Engineering report: restore the multi-call pipeline, instant save, image structured data, attachment access

This supersedes the previous report (the "streamlining phase," summarized
below where it's still relevant). Scope: four related changes, driven by
one premise the streamlining phase's own trade-off no longer holds under
-- interactive recall now lives entirely in Browse/Remember, so saving a
memory no longer needs to return a synchronously-analyzed result at all.

1. **Restore the multi-call ingestion pipeline**, now fully backgrounded,
   so the quality the streamlining phase traded away can come back at
   zero perceived latency cost.
2. **Make saving instant**: `POST /api/captures` returns the moment raw
   text + attachments are durably persisted, before any LLM call.
3. **Fix lost structured data in images** (the concrete bug that
   motivated this work: a photographed glasses prescription's numeric
   values were getting summarized into prose, or lost, instead of
   becoming queryable facts).
4. **Attachment view/download**, previously missing entirely -- a
   retrieved memory with a photo attached had no way to actually see
   the photo anywhere in the UI.

Verified against the real local model throughout: the full fast pytest
suite (161 tests, all passing, against fakes) and the real-`gemma4:12b`
integration suite (5 tests, all passing -- one needed a prompt fix,
documented in section 4).

## 1. Current architecture

Unchanged in *shape* from docs/ARCHITECTURE.md's description: a
`Capture` decomposes into one or more `MemoryUnit`s; each unit carries
authoritative content (verbatim, never rewritten), an append-only
clarification history, a derived semantic representation, and a status
gate (`draft -> awaiting_clarification -> processing -> ready`, or
`failed`/`discarded`). What changed is *where* the LLM-dependent work
happens, not what it produces: all of it (image understanding,
segmentation, clarification, enrichment, embedding, structuring) now
runs in one background task, `process_capture_fully()`, scheduled the
instant `POST /api/captures` persists the raw text. `create_capture()`
itself no longer calls the LLM at all.

## 2. Reversed: the streamlining-phase ingestion collapse

The previous phase collapsed segmentation + clarification-need +
enrichment into one call per capture (`analyze_capture()`) specifically
to cut *synchronous-request* latency, trading some retrieval quality
for it (documented there: 28/30 vs 30/30 on the eval set, one
regression explicitly attributed to losing a dedicated clarification
call's focused attention). That trade's entire premise -- a capture's
full analysis has to finish before the HTTP response can return --
stopped being true once saving became fully backgrounded. The restored
`app/ingestion/segmentation.py`/`clarify.py`/`enrichment.py` (reconstructed
from pre-collapse git history, adapted to the current `Semantics` shape
and multi-image/prior-context support that didn't exist back then) cost
more LLM calls per capture, but none of them are on any critical path a
human is waiting on any more.

`resolve_clarification()` and `correct_unit()` were rewired to the same
clarify/enrich composition. One correctness gap in the collapsed design
was fixed along the way, not just inherited: `resolve_clarification()`
now *always* eventually calls `enrich_unit()` once no further question
will be surfaced, closing a path where a unit's only candidate question
got suppressed (round budget exhausted, or the field was already marked
declined/unknown) and the unit could previously reach `"ready"` with
semantics that were never actually enriched.

## 3. Performance

LLM call counts, by inspection:

| Flow | Streamlining phase | Now |
|---|---|---|
| Ordinary memory, no clarification | 1 generation call + 1 embed (synchronous) | up to 5 generation calls + 1 embed (segment, clarify, 3x enrich) -- **all backgrounded, zero synchronous calls** |
| Memory needing 1 clarification round | 1 (initial) + 1 (`reanalyze_unit`) + 1 embed | 1 segment + 1 clarify (round 1) + 1 clarify (round 2, settles) + 3 enrich + 1 embed -- **still entirely backgrounded** |
| Image attachment | captioned in a separate, later-scheduled background task, blind to the capture's first analysis pass | captioned (+ a new structured-extraction call) in the SAME background task, BEFORE segmentation/clarification/enrichment ever run, so they see its content |
| `POST /api/captures` response latency | bounded by the synchronous analysis call (up to `llm_timeout_seconds`, 90s) | bounded by a DB commit only -- no LLM call in the request path at all |

The trade is the opposite of the streamlining phase's: more total LLM
calls, in exchange for zero of them being on the critical path. Not
measured: whether the background task's own total wall-clock time grew
enough to matter for anything (nothing is waiting on it, so this was
judged not worth instrumenting yet -- see docs/TODO.md if it ever is).

## 4. Retrieval quality

Re-running the real-model integration suite surfaced one live
regression from restoring a *dedicated* clarification call: asked in
isolation (no enrichment competing for its attention), `gemma4:12b`
asked an unnecessary clarification question about a memory's missing
optional detail (a recommended paper's title/author) -- directly
violating its own prompt's explicit instruction that "a missing field
is not, by itself, grounds for a question." This gated the unit out of
search entirely, which is exactly the failure mode
`test_indirect_multiconcept_relationship_query` exists to catch. Fixed
by adding a concrete negative example matching this exact case to
`clarify.py`'s system prompt; re-run confirmed fixed, deterministically
(structured-output calls run at `temperature=0.0`, so this was a
reliable prompt-adherence gap, not sampling noise). All 5 real-model
integration tests pass after the fix.

This is the same category of risk the streamlining phase itself
flagged going the other direction ("a dedicated call, with no other
task competing for its attention, [may ask more]") -- confirmed real
here, and addressed by prompt wording rather than by abandoning the
dedicated-call design, since the alternative (collapsing calls back
together) is exactly the trade-off this whole effort exists to undo.

## 5. Image structured-data extraction

`_describe_image()`'s free-text caption is unchanged and still the
image's general-purpose searchable description. New alongside it:
`_extract_image_structured_data()`, a schema-constrained call
(`has_structured_data: bool`, `fields: [{label, value}]`) prompted to
transcribe any form/table/prescription/measurement data exactly as
shown, appended to the caption as a `"Structured data:\n..."` block.
`enrichment.py`'s retrieval-aids sub-prompt gained one instruction:
promote each such labeled pair into its own `relations` entry rather
than summarizing the block in prose -- this is the line that makes a
photographed prescription's numbers land as queryable
`semantic.relations` instead of staying inert text a free-text caption
might also have garbled. Two images in one capture are labeled
`"Image 1:"`/`"Image 2:"` in the combined attachment text so one
photo's values and another's can be merged into one unit's `relations`
by a single enrichment call.

## 6. Attachment view/download

Previously missing entirely: no backend route served an attachment's
original bytes, and the UI only ever rendered a filename as plain text.
New `GET /api/attachments/{id}/file` (no new auth code -- nginx's
existing Basic Auth already covers it). `SourceOut`/`ChunkSourceOut`
extended to carry attachment info so a retrieved memory's sources can
link to their originals too. The frontend gained one shared
fetch-as-blob-then-open-or-download helper, wired into the Store page,
the Browse tab, and the Recall page.

## 7. Multi-file upload

Investigated as part of this work (reported as single-file-only) and
found already fully correct end-to-end -- `<input multiple>` with
cross-picker accumulation on the frontend, `list[UploadFile]` on both
upload endpoints on the backend. No code change. If still observed as
single-file in the deployed app, the most likely cause is a stale
published GitHub Pages build (the frontend is published separately via
`scripts/build_frontend.py`, not served by this backend) -- rebuild and
republish before assuming a code regression.

## 8. Local LLM setup

Unchanged: `gemma4:12b` for every generation/structured-output call,
`nomic-embed-text` (768-dim) for embeddings, both via the system Ollama
service. No new model, no new config. The per-image caption call is
joined by one new structured-extraction call per image (same model,
same multimodal path) -- still well within the single-generation-model
VRAM budget documented in CLAUDE.md, since nothing here changes which
models are loaded, only how many `generate_structured` calls happen per
image.

## 9. Deferred / not done

- **`docs/REPORT.md`'s own historical sections** (this file, previous
  phase) on contradiction handling, recall-session deferred items, and
  the eval-threshold sweep are unaffected by this work and weren't
  re-verified here -- see the superseded report's sections 5/8/9/10 if
  still relevant.
- **No wall-clock baseline** for the background task's own total
  duration (see section 3) -- nothing is waiting on it, so not
  instrumented.
- **GitHub Pages republish** -- code-correct already, but the live
  site wasn't rebuilt/republished as part of this work (see section 7);
  that's a deploy step, not a code change.

## 10. Remaining risks

- The dedicated clarify call's prompt-adherence fix (section 4) was
  validated against one concrete failure mode found live, not swept
  across the 30-case eval corpus (`scripts/run-eval.sh`) -- worth a full
  eval re-run before treating this as fully closed, the same caveat the
  streamlining phase's own regressions carried.
- Image structured-data extraction (section 5) was validated by
  targeted reasoning about the prompt/schema and the existing test
  suite, not against a real photographed prescription -- the acceptance
  test for "does this actually work" is trying it with a real photo
  through the dev server, not yet done here.
