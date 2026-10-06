# TODO / open questions

## Resolved this phase (spec section 44 decisions)

- **Clarification gates recall availability.** `status="awaiting_clarification"`
  is excluded from search until answered or discarded. See docs/DECISIONS.md.
- **Derived metadata is not directly editable**; corrections go through
  `POST /memory-units/{id}/correct` (natural-language, re-runs enrichment),
  not a metadata editor.
- **Attachments are semantically searchable through derived chunks**
  (`attachment_chunks`) linked to the parent capture; chunks are not
  independent top-level memories.
- **No automatic deletion/expiry.** Explicit user deletion
  (`DELETE /api/captures/{id}`) is supported and removes files too.
- **Image understanding: deferred** (unchanged from the previous phase).
- **Knowledge graph: deferred** -- `semantic.relations` stays a flat list
  of subject/predicate/object triples, not a queryable graph.
- **Multi-agent architecture: deferred** -- one bounded retrieval agent.

## Open product questions

- **Should the retrieval threshold be relative/adaptive rather than a
  fixed scalar?** The sweep in docs/DECISIONS.md picked 0.55 as the
  highest value with 100% recall@10 on 30 cases, but a margin-below-top-
  score rule might generalize better as the corpus grows. Needs a larger
  eval set to tell the two apart.
- **Should there be a UI affordance for browsing/deleting individual
  captures**, not just creating and recalling? The API supports it
  (`GET`/`DELETE /api/captures/{id}`); the web UI doesn't expose it yet.
- **Should `MAX_CLARIFICATION_ROUNDS` ever be >1?** Bounded to 1
  deliberately (spec: avoid interrogation); unclear if real usage ever
  needs a second round, or whether `correct_unit()` (unbounded,
  user-initiated) already covers that need well enough.

## Known technical gaps

- **Re-embedding is whole-row, not incremental.** `scripts/reembed.py`
  re-embeds a unit/chunk entirely if its embedding is missing or
  stale-model; it has no concept of "only the aliases changed, reuse the
  rest," which is fine at this scale but wouldn't be at a much larger one.
- **Segmentation's verbatim-fallback is a blunt instrument.** It protects
  against promoting paraphrased content to authoritative status, but a
  model that produces clean, correctly-punctuated verbatim sub-sentences
  while still cutting across a pronoun reference (observed during eval:
  "I met Anna... and she recommended a paper" split into two
  grammatically-clean verbatim halves) will pass the check and still
  lose the antecedent in the *primary* evidence block. Mitigated at the
  answer-generation layer (the prompt explicitly directs resolving
  pronouns using sibling "related context" evidence, confirmed fixing
  the one eval failure this caused), but segmentation itself has no
  linguistic awareness of reference chains.
- **Clarification question targeting/triggering is inconsistent** across
  model calls for borderline cases. Bounded to 1 round per unit, so the
  blast radius is small, but not fully reliable -- and now that it
  gates recall, an unnecessary question has a bigger UX cost than before
  (the memory is invisible to recall until answered or discarded).
- **No image understanding.** Images are validated and stored but never
  captioned/OCR'd.
- **Chunk size (1200 chars, 150 overlap) is a fixed, untuned default.**
  Works for the eval set's short attached notes; untested against a
  genuinely large (50+ page) PDF.
- **Agent latency is unbounded-ish in the worst case.** 3 rounds, each
  now searching both units and chunks for every query in play, means up
  to ~14 search calls plus up to 3 evaluate calls plus 1 answer call per
  recall request in the worst case. Observed in practice to resolve in
  1 round for most queries (30/30 eval cases), but a pathological query
  could take tens of seconds.
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
