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

**Decision (superseded, see below):** An earlier iteration of this
project let a unit become searchable immediately after its first
enrichment pass regardless of any open clarification question, reasoning
that gating would require a staging state machine and durability was the
higher priority. **This was reversed**: a unit with a genuinely open
question is now `status="awaiting_clarification"` and is excluded from
`hybrid_search()` until answered or explicitly discarded -- durably
saved the whole time (a crash or restart never loses it), just not
recall-visible. Kept here rather than deleted because the reasoning for
*why* gating matters is worth keeping: a memory whose interpretation is
still genuinely unresolved (what does this term/identifier mean?) risks
answering confidently from an incomplete understanding if it's
searchable before that's settled -- durability and searchability turned
out to be separable goals, not a package deal. Discarding an awaiting
unit (`POST /memory-units/{id}/discard`) is a status flip, not a hard
delete, since the user might reconsider; actual deletion
(`DELETE /api/captures/{id}`) is a separate, explicit action.

---

**Decision:** A user-initiated correction (`POST
/memory-units/{id}/correct`, free-text) reuses the exact same
"append to `clarifications`, re-run enrichment" mechanism as answering a
system-asked clarification question, but does not consume
`clarification_round`.

**Reason:** No reason to build a second enrichment-update code path when
the only real difference is *who* initiated it. Not counting it against
the round budget matters because that budget exists specifically to stop
the *system* from pestering the user with questions -- it has nothing to
do with how often a user chooses to fix something themselves.

---

**Decision:** Attachments are chunked (page-based for PDFs, fixed
character window + overlap otherwise) with each chunk independently
full-text and vector indexed, rather than only ever contributing a single
blind excerpt to a parent unit's evidence.

**Reason:** A fact buried on page 40 of an attached PDF was previously
only reachable if it happened to fall within the first ~500 characters
blindly copied into evidence. Chunking + independent indexing means the
*relevant* chunk gets found by the same hybrid search as everything else,
and provenance can point at a specific attachment + page rather than
"somewhere in this file."

**Alternatives considered:** A full document-processing pipeline
(semantic chunking, overlap tuning per content type, re-ranking chunks
against the parent memory) -- explicitly out of scope per spec section
27 ("do not build a sophisticated document-processing pipeline"); fixed-
size chunking is the simplest thing that makes large-document content
findable at all.

---

**Decision:** `VECTOR_SIM_THRESHOLD = 0.55`, kept at its prior
hand-picked value, but now backed by `scripts/sweep-threshold.sh` run
against the 30-case eval set rather than a handful of manual trials.

**Reason:** The sweep (0.35 through 0.70 in steps of 0.05) showed
recall@10 flat at 100% from 0.35 up through 0.55, then dropping (96% at
0.60, 89% at 0.65, 78% at 0.70) as real matches stopped clearing the bar.
0.55 is the highest threshold in the flat-100%-recall region, which this
project's grounding philosophy (prefer abstention over a wrong answer,
but *also* prefer not drowning the LLM's evidence block in irrelevant
noise) treats as the right side to err on. Precision (own-case hits /
total hits returned) rose monotonically with threshold across the whole
sweep (13% at 0.35 up to 34% at 0.70), so there's a real recall/precision
trade available above 0.55 -- 0.55 is specifically the point that doesn't
give up any measured recall to get it.

**Caveats:** 30 cases is enough to catch a threshold that's clearly too
loose or too tight, not enough to distinguish 0.50 from 0.55 with
confidence. The precision metric specifically undercounts: all 30 cases'
data shares one database for the sweep (to avoid re-ingesting per
threshold), so a query belonging to one case can match another case's
*own* units (e.g. several cases mention "Anna") without being flagged as
either a hit or a miss for either case, pulling precision down for
reasons unrelated to the threshold itself. Recall is unaffected by this
(it only checks whether a case's own units were found), which is why
recall, not precision, drove the final choice.

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
