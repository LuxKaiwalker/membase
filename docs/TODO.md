# TODO / open questions

## Open product questions

- **Should clarification gate recall availability, not just enrich
  after the fact (current)?** See docs/DECISIONS.md. The spec's own
  ingestion diagram suggests gating was the intended default; current
  behavior prioritizes durability/simplicity instead. Revisit with real
  usage feedback -- does a user expect a memory to be "incomplete" until
  clarified, or is immediate availability the right default?
- **Should memory semantic representation (`semantic` jsonb) be
  user-editable?** Currently read-only/machine-generated. No UI/API for
  correcting a wrong extracted alias/relation other than re-triggering
  enrichment via a new clarification round (bounded to 1).
- **Should attachments be independently searchable from their parent
  capture's units?** Currently an attachment's extracted text only ever
  surfaces as part of every sibling unit's full-text index.
- **Should memories ever be automatically deleted/archived?** No
  forgetting/expiry logic exists.
- **Should there be a UI affordance for inspecting/deleting individual
  memory units or whole captures?** Not built; only create+recall flows
  exist in the web UI (API supports GET by id, nothing for delete).

## Known technical gaps

- **No backfill/reindex tool.** If `MEMBASE_EMBED_MODEL` changes, or a
  unit's embedding failed at creation time, nothing automatically
  re-embeds existing rows. `embedding_model`/`embedding_dim` are stored
  per-row specifically so this is at least *detectable*.
  `scripts/model-doctor.sh` checks the *configured* model's dimension
  matches the schema, but doesn't touch existing data.
- **Retrieval threshold (`VECTOR_SIM_THRESHOLD = 0.55` in
  `app/retrieval/pipeline.py`) is a hand-picked value from a handful of
  manual trials**, not a tuned one against a real eval set. The eval set
  in `eval/cases.yaml` (12 cases) is large enough to catch gross
  regressions, not to calibrate a numeric threshold responsibly -- that
  would need ~30-50 cases and a deliberate sweep.
- **Segmentation's verbatim-fallback is a blunt instrument.** It protects
  against promoting paraphrased content to authoritative status, but a
  model that produces clean, correctly-punctuated verbatim sub-sentences
  while still cutting across a pronoun reference (observed during eval:
  "I met Anna... and she recommended a paper" split into two
  grammatically-clean verbatim halves) will pass the check and still
  lose the antecedent in the *primary* evidence block. Mitigated at the
  answer-generation layer (the prompt now explicitly directs resolving
  pronouns using sibling "related context" evidence, confirmed fixing
  the one eval failure this caused), but segmentation itself has no
  linguistic awareness of reference chains.
- **Clarification question targeting/triggering is inconsistent** across
  model calls for borderline cases (sometimes asks about an
  already-resolved detail, sometimes doesn't ask about a genuinely
  invented term depending on exact phrasing). Bounded to 1 round per
  unit, so the blast radius of a bad question is small, but it's not
  fully reliable.
- **No image understanding.** Images are validated and stored but never
  captioned/OCR'd.
- **No chunking for large attachments.** A long PDF is fully indexed for
  full-text search but only a 2000-character excerpt feeds the
  enrichment prompt and the answer-generation evidence block.
- **Post-hoc attachments don't re-embed or re-run enrichment** for
  sibling units -- only the full-text index is extended.
- **Agent latency is unbounded-ish in the worst case.** 3 rounds × up to
  2 reformulated queries each means up to ~7 search calls plus up to 3
  evaluate calls plus 1 answer call per recall request in the worst
  case. Observed in practice to resolve in 1 round for most queries
  (see docs/REPORT.md eval results), but a pathological query could take
  tens of seconds.
- **No rate limiting / request size limits** beyond the per-file size
  cap. Not a concern for a single-user localhost MVP.
- **UI was validated via rendered-HTML inspection and direct API calls,
  not a live browser session** (no browser automation tool was available
  in the environment this was built in).

## Explicitly deferred (not evaluated as gaps, just out of scope)

- Multi-turn conversational memory, autonomous background agents,
  general knowledge-graph storage, memory consolidation, human-like
  forgetting, multi-agent systems, advanced temporal reasoning,
  multimodal (image) understanding, cloud deployment, mobile apps,
  multi-user auth, a learned/trained reranker.
