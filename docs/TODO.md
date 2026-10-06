# TODO / open questions

Things that are unfinished, provisional, technically uncertain, or
waiting on a product decision. See docs/REPORT.md for a prioritized,
explained version of the top few; this file is the flat list.

## Open product questions

- **Should clarification gate persistence, or only enrich after the
  fact (current)?** See docs/DECISIONS.md. Revisit if users expect a
  memory to be "incomplete" until clarified rather than immediately
  searchable.
- **Should memory metadata (`derived`) be user-editable?** Currently
  read-only/machine-generated, displayed as-is. No UI or API support for
  correcting a wrong extracted person/date/topic other than via the
  one-shot clarification flow at creation time.
- **Should files be independently searchable from their parent memory?**
  Currently an attachment's extracted text only ever surfaces as part of
  its parent memory's full-text index/evidence block -- there's no
  "search within my PDFs" mode.
- **Should memories ever be automatically deleted/archived?** No
  forgetting/expiry logic exists; everything persists forever until
  manually removed from the database.

## Known technical gaps

- **No backfill/reindex tool.** If `MEMBASE_EMBED_MODEL` changes, or a
  memory's embedding failed at creation time (LLM was down), there is no
  job that re-embeds existing rows. `embedding_model`/`embedding_dim` are
  stored per-row specifically so this is at least *detectable* later.
- **Retrieval threshold (`VECTOR_SIM_THRESHOLD = 0.55`) is a hand-picked
  value from a handful of manual trials**, not a tuned one. Empirically,
  nomic-embed-text puts unrelated short English sentences at ~0.42-0.65
  cosine similarity, so there's no principled zero point -- this needs a
  larger eval set to calibrate properly, and will behave differently with
  a different embedding model.
- **Clarification question targeting is inconsistent.** The 7B generation
  model occasionally asks a plausible-sounding but oddly-aimed or
  spurious clarification question (e.g. asking to confirm a date that was
  already resolved correctly). Low severity since clarification is purely
  additive (see docs/DECISIONS.md), but noticeable in manual testing.
- **No image understanding.** Images are validated and stored but never
  captioned/OCR'd -- they contribute nothing to search. Documented as
  allowed-to-defer in spec section 7.
- **No chunking for large attachments.** A very long PDF is fully indexed
  for full-text search but only a 2000-character excerpt is considered
  for the metadata-extraction prompt and attachment evidence shown to the
  answer-generation LLM; most of a large document is effectively
  invisible to the LLM-facing parts of the pipeline.
- **Post-hoc attachments (`POST /memories/{id}/attachments`) don't
  re-embed or re-run metadata extraction** -- only the full-text index is
  extended. See docs/DECISIONS.md rationale inline in
  `app/ingestion/pipeline.py:add_attachment`.
- **No rate limiting / request size limits beyond the per-file size cap**
  (`MEMBASE_MAX_UPLOAD_MB`). Not a concern for a single-user localhost
  MVP, but would matter before any multi-user or network-exposed use.
- **UI was validated via rendered-HTML inspection and direct API calls,
  not a live browser session** (no browser automation was available in
  the environment this was built in). The JS was traced manually against
  the verified API contract but hasn't been click-tested end-to-end in a
  real browser.
