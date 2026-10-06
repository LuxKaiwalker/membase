# Decisions

Only decisions with real alternatives and real consequences are recorded
here — not every implementation detail.

---

**Decision:** A single user submission ("capture") may decompose into
multiple independently-retrievable memory units, or stay as one; the LLM
decides this per-capture via `app/ingestion/segmentation.py`, with a hard
verbatim-content check as a safety net.

**Reason:** The project's core value proposition is recovering information
via indirect/associative cues. A single giant embedding for "my AHV number
is X, Anna studies robotics, and I need to renew my passport" would blur
three unrelated retrieval targets into one noisy vector. But forcing every
sentence into its own atomic fact would shred coherent episodes (destroying
the context that makes "who recommended that paper" answerable). Neither
extreme works; the split has to be semantic, not mechanical.

**Alternatives considered:** Always one memory per submission (the original
MVP's model) -- simple, but conflates unrelated facts into one search unit
and was the most-cited architectural gap in review. Always split by
sentence -- destroys context for coherent episodes (see the "she
recommended" pronoun-resolution bug found during eval, fixed in the
answer-generation prompt rather than by abandoning segmentation).

---

**Decision:** Segmentation only accepts a proposed unit if its content is
a verbatim (whitespace/case-normalized) substring of the original text;
if any proposed unit fails this check, the whole capture falls back to
one unit.

**Reason:** This is the one thing standing between "the LLM organizes the
user's own words" and "the LLM silently rewrites what the user said and
we call it authoritative." A paraphrased "unit" would promote model
wording to the same authority level as user input, which the hierarchy in
docs/ARCHITECTURE.md explicitly rules out.

**Alternatives considered:** Character-offset span tracking into the
original text instead of a copied string -- more precise, but meaningfully
more implementation complexity (every downstream consumer needs to resolve
spans) for an MVP where whole-text fallback is an acceptable safety net.

---

**Decision:** Clarification answers are folded back into the unit's
semantic representation (aliases, definitions, meaning) by *re-running
enrichment with the answer as context*, not by just appending the Q&A as
inert text.

**Reason:** This is the actual mechanism that makes "AHV number" and
"Swiss insurance number" cross-retrievable after a clarification -- the
alias has to end up in `retrieval_text` (and therefore in the embedding
and the full-text index), not just sit in a `clarifications` log that
nothing searches.

**Alternatives considered:** Appending clarification text to
`searchable_text` only (what the original MVP did) -- works for literal
keyword overlap but not for semantic/paraphrased retrieval of the
clarified concept, since the embedding never changes to reflect it.

---

**Decision:** Clarification enriches a unit but does not gate its
availability for recall -- a unit is `status="ready"` (searchable)
immediately after its first enrichment pass, whether or not a
`pending_clarification` is still open.

**Reason:** Durability and simplicity: gating would require a staging
state machine (draft -> awaiting_clarification -> ready) and a decision
about what happens if the user never answers. Immediate availability
means the authoritative content is never at risk of being "stuck"
unsearchable, and most memories need no clarification at all (by design
-- see the enrichment prompt's bias toward not asking).

**Alternatives considered:** Gating recall on clarification completion.
This is a legitimate, debatable product decision rather than a purely
technical one -- flagged explicitly in docs/TODO.md as something to
revisit with real usage feedback, since the spec's own ingestion diagram
suggests gating was the intended default.

---

**Decision:** `qwen2.5:14b-instruct-q4_K_M` is used for every
generation/structured-output call (segmentation, enrichment, agent
evaluation, answer generation) -- not `qwen2.5:7b-instruct`, and not a mix
of the two.

**Reason:** Tested both directly (see CLAUDE.md / git history for the
transcripts). 7b reliably over-segmented coherent episodes against
explicit few-shot instruction to the contrary, and never once recognized
an invented/unfamiliar term as worth a clarification question -- it
always fabricated a confident-sounding definition instead, which is
exactly the failure mode the clarification mechanism exists to prevent.
14b handled both correctly. Separately: this GPU has 11GB VRAM; 14b-q4
(~9GB) + nomic-embed-text (~0.3GB) coexist without eviction, but adding a
7b model (~4.7GB) into the same rotation would exceed available VRAM and
cause a measured ~7-14s reload stall every time Ollama swaps between them
-- so using a second generation model isn't just unnecessary, it's a
measured latency regression.

**Alternatives considered:** 7b for cheap/fast paths (segmentation) and
14b only for the agent -- rejected once the VRAM-swap cost was measured;
the quality gain from always using 14b outweighs that cost, which turned
out to be ~0 once nothing else is competing for VRAM.

---

**Decision:** PostgreSQL + pgvector + built-in full-text search, one
database, no separate vector store.

**Reason:** Keeps the architecture simple and colocates structured
metadata, full-text search, and vector search in one place with one
transaction model.

---

**Decision:** Dependencies come entirely from `nixpkgs.python313Packages`
via the flake's `python3.withPackages`; no `requirements.txt`/`venv`/`pip`.

**Reason:** Binary wheels from PyPI routinely fail on NixOS (missing
dynamic linker paths/shared libs). Everything this project needs happens
to already be packaged in nixpkgs.

---

**Decision:** One FastAPI process serves both the JSON API and
server-rendered (Jinja2) HTML; browser-side JS only calls the JSON API.

**Reason:** Satisfies "separate frontend from memory service" without a
second process/build toolchain for an MVP.

---

**Decision:** No separate reranker model/service; the hybrid fused score
(0.65 × vector + 0.35 × normalized full-text rank) is the MVP's
"reranking" step, and the agent's bounded reformulation loop is the
MVP's "retrieval-oriented reasoning," not a general tool-calling agent
framework.

**Reason:** A cross-encoder reranker and a general agent framework are
each real additional systems to host and evaluate; for an MVP at
dozens-to-hundreds-of-memories scale, a deterministic bounded loop with
narrow structured LLM decisions gets most of the benefit at a fraction of
the complexity and failure surface.
