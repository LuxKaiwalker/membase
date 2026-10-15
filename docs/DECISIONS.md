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

---

**Decision:** Recall is a persisted, multi-round `RecallSession`
(`app/retrieval/session.py`), stored in Postgres as one row per session,
not in-memory server state and not a new document store.

**Reason:** The project's central premise is that a user's first
recollection is often a weak, partial cue -- not a wording close enough
for one search call to land on. A stateless one-shot endpoint has no way
to ask a clarifying question and incorporate the answer. Postgres is
already the app's only state store (see the pgvector/FTS decision
above); a session is architecturally the same kind of thing a
MemoryUnit's `pending_clarification` already is -- durable process state
-- so reusing the same database, rather than adding Redis/in-memory
session state, needed no new infrastructure and gets restart durability
for free (verified: start a session, kill and restart the dev server,
`GET /api/recall/sessions/{id}` returns identical state).

**Alternatives considered:** In-memory server-side session dict keyed by
id -- rejected because it doesn't survive a restart (this project treats
that as a real property, not an edge case -- see the restart-durability
bugs already fixed in CLAUDE.md) and would need its own expiry/cleanup
logic for no benefit over a DB row. A dedicated `recall_messages` table
(one row per turn) instead of a JSONB `messages` array on the session
row -- rejected as unnecessary relational structure for data that's
always read/written as a whole conversation, never queried turn-by-turn
(same reasoning already applied to `MemoryUnit.clarifications`).

---

**Decision:** A `RecallSession`'s `cues`, `hypotheses`, and `messages`
are stored *only* on the session row. Nothing in `app/retrieval/
session.py` or `app/retrieval/cues.py` ever calls `create_capture` or
writes to `MemoryUnit`/`Capture`.

**Reason:** This is the load-bearing guarantee against an iterative
retrieval system contaminating the memory store with its own guesses
(spec section 5/29). An agent hypothesis ("maybe this was about tactile
sensing") or an uncertain user cue ("maybe Anna") must stay exactly that
-- search input and session bookkeeping -- never promoted to a stored
fact just because a recall conversation touched on it. The only
sanctioned way something discussed during recall becomes a durable
memory is the user separately using the ordinary `POST /api/captures`
flow with their own words.

**Open question (deliberately unresolved, see docs/TODO.md):** whether
the UI should ever offer an explicit "save this as a memory" action
during recall. Not implemented this phase -- it's a real product
decision (what exactly gets saved: the cue? the final answer? in whose
words?) rather than an obvious engineering call, so it's recorded as
open rather than decided by default.

---

**Decision:** Ambiguity between candidate memories is detected by a
flat score-margin heuristic over origin-capture groups
(`is_ambiguous`/`AMBIGUITY_SCORE_MARGIN = 0.15` in `app/retrieval/
discriminate.py`), not a probabilistic/Bayesian model.

**Reason:** Spec section 13 explicitly permits (and section 14 asks for)
"a simple heuristic based on candidate attributes... sufficient for the
MVP" rather than formal active-learning. Two candidates from different
origin captures (likely different real-world events) whose top fused
scores are close are treated as a real decision the user should make,
not something to silently resolve by picking whichever ranked first --
sibling units of the *same* capture are explicitly not "candidates" for
this purpose (they're one episode, not competing answers).

**Alternatives considered:** A fixed top-K cutoff ("always show the top
2 as candidates") -- rejected because it would ask a discriminating
question even when one memory massively outranks every other
unit-that-happens-to-share-a-keyword, which is exactly the "don't ask
when you don't need to" cost the project already treats as real (see
the clarification-gating risk note in the prior report).

**Caveat (known limitation, see docs/TODO.md):** 0.15 is hand-picked,
not calibrated via a sweep the way `VECTOR_SIM_THRESHOLD` was -- there
wasn't yet a large enough multi-round eval set to run that experiment
against. Empirically, two topically-different candidates can end up with
a *wider* score gap than two genuinely-plausible ones purely because of
full-text-search asymmetry (one sentence happens to share more literal
words with the query than the other, independent of which one is
actually more plausible) -- observed directly during manual testing of
this phase; see docs/TODO.md.

---

**Decision:** A confirmed positive cue boosts a candidate's effective
score by a flat bonus (`apply_positive_cue_boost`,
`POSITIVE_CUE_BOOST = 0.3`) when the cue text matches that candidate's
own structured semantic metadata (people/organizations/locations/
aliases/topics) -- it is not just added as another search query and
left to the normal hybrid-search fusion to sort out.

**Reason:** Discovered empirically, not anticipated in the original
design: re-searching with a user's cue as its own short query (e.g. just
"ETH") frequently produces a *lower* fused score for the correct
candidate than the original full-sentence query already gave it -- a
short, low-information query embeds and full-text-ranks differently
than a full sentence, so merging per-query results via "keep the max
score seen for this unit across any query tried" never moved the needle
on disambiguation. Since `generate_discriminating_question` already
asks about a specific attribute *value* (e.g. "Did this happen at
ETH?"), a "yes" is structured confirmation of that exact value, not just
another vague search term -- treating it as a direct metadata match and
rewarding it accordingly is what actually closes the loop. Verified
against the real local model: without this, a correct "yes, ETH"
answer left the session still ambiguous indefinitely; with it, the
session resolves in the next round.

**Alternatives considered:** Re-weighting the vector/FTS fusion
coefficients specifically for single-word queries -- rejected as a more
invasive, harder-to-reason-about change to code shared with ordinary
one-shot recall, for a problem that's specifically about *confirmed
cues*, not search quality in general.

---

**Decision:** Storage-time clarification is split into two separate LLM
calls -- `app/ingestion/clarify.py:generate_questions()` (fast, narrow:
only decides whether to ask and about what) runs *first*; full semantic
enrichment (`app/ingestion/enrichment.py:enrich()`) only runs once no
clarification question is pending. The combined single-call design from
the previous phase is gone.

**Reason:** Two problems with asking one combined call to both derive
the full semantic representation *and* decide on a clarification
question: (1) the user never sees a question until the whole (larger)
enrichment schema finishes generating, even though the question is the
time-sensitive part; (2) if a question *was* needed, that enrichment
work was wasted anyway, since resolving the clarification re-runs
enrichment from scratch with the answer as context. Splitting means a
unit reaches `awaiting_clarification` after one small, fast call, and
full enrichment runs exactly once, after clarification is fully settled
-- strictly less wasted LLM work for any unit that ends up needing a
question, at the cost of one extra fast call for units that don't.

**Alternatives considered:** Keep one combined call but reorder its
internal reasoning (ask the model to decide clarification "first" within
the same response) -- rejected because the user still can't see anything
until the whole response is generated; a single LLM call has no
meaningful partial-result boundary to exploit without streaming, which
this project doesn't use.

---

**Decision:** `enrich()` itself (full semantic enrichment, run once
clarification is settled) is further split into three independent calls
-- narrative (`meaning`, `event_date`), entities (`people`,
`organizations`, `locations`), and retrieval aids (`aliases`, `topics`,
`definitions`, `relations`) -- instead of one combined 9-field schema.
Each degrades independently on LLM failure (a failed slice doesn't
discard what the other slices already extracted).

**Reason:** A single call asking for nine fields at once spreads the
model's attention and output budget across all of them regardless of how
much any one actually warrants; three narrower, single-purpose prompts
let each slice get the model's full attention. Explicitly requested as a
direction to try, on top of (not in place of) the clarify/enrich split
above -- an empirical A/B comparison of the *old* single-call prompt
against the *trimmed* single-call prompt (done when the clarify/enrich
split landed) found no quality regression for either, but neither of
those tested this specific question (multi-prompt vs. single-prompt
extraction), which is a different axis.

**Cost:** Two extra sequential LLM calls per unit (~3x enrichment
latency instead of ~1x) -- see the ingestion-timing documentation for
what that costs in wall-clock terms. Accepted as the explicit trade-off
of this decision.

**Caveat, observed directly while verifying this change:** one existing
real-LLM integration test
(`test_paraphrased_query_retrieves_relevant_memory`) went from reliably
passing to failing more often than not (3 of 4 runs) after this change.
Traced to its root cause, not just observed: `segment_capture()`
deterministically splits "My trip to Japan last year was amazing,
especially the food in Osaka." into two verbatim-valid units, "My trip
to Japan last year was amazing" and "the food in Osaka" -- the second
fragment loses the elided "amazing" it grammatically depended on in the
original sentence (an *elided-qualifier* case, the same underlying class
of bug as the already-documented pronoun-reference segmentation gap, see
docs/TODO.md, just without a pronoun). This segmentation behavior is
unrelated to and unchanged by this decision -- confirmed deterministic
(3/3) in isolation -- but the *different* semantic content the new
multi-prompt enrichment produces for the orphaned fragment apparently
shifts the downstream grounding call's already-borderline judgment for
this specific case from usually-grounded to usually-not. Not fixed here:
the actual bug is in segmentation/grounding's handling of elided
references, which is pre-existing, separately scoped work -- see
docs/TODO.md.

**Alternatives considered:** Reverting to single-call enrichment --
rejected; this was an explicit direction to try, and the one observed
regression traces to a pre-existing, different-layer bug this change
merely made more visible, not one this change itself introduced.

---

**Decision:** Clarification now supports multiple questions per round
(`MAX_QUESTIONS_PER_ROUND = 3`, asked one at a time via
`MemoryUnit.clarification_queue`, never shown as a batch) and multiple
rounds (`MAX_CLARIFICATION_ROUNDS` raised from 1 to 3) -- an answer that
reveals new ambiguity can trigger a fresh round of questions, the same
iterative principle `RecallSession` uses for retrieval-time
clarification, applied here to storage-time clarification. A round only
counts against the budget if it actually surfaced a question; a unit
that never needed clarification stays at `clarification_round = 0`
indefinitely, not "one check used up".

**Reason:** The old single-question cap meant a unit with two genuinely
unclear terms could only ever be asked about one of them. Generating
several candidates at once and ranking them lets the fast question-check
call surface everything worth asking in one pass, while still asking the
user one concrete question at a time (never an interrogation-style
batch, same UX principle as `RecallSession`'s discriminating questions).
Allowing a follow-up round after an answer closes a real gap: an answer
can introduce a *new* unclear term the model couldn't have known to ask
about up front.

**Caveat:** Raising the round/question bounds increases the *number* of
questions a unit could in principle generate before this was measured
against real usage -- this is a judgment call, not a calibrated number
(unlike `VECTOR_SIM_THRESHOLD`, which had a sweep). If real usage shows
this interrogates users more than it helps, the bounds should come back
down; see docs/TODO.md.

---

**Decision (open, deliberately not implemented):** Information gained
during a *retrieval* session (user cues, confirmed facts) should, in
principle, eventually be promotable into durable memory -- the answer to
"should this ever happen" is yes. This is **not implemented**: no code
path exists for it, and none should be added yet.

**Reason:** The retrieval-session architecture itself may still change
(see the interactive-vs-one-shot benchmark findings below) before it's
worth designing a promotion mechanism on top of it -- building that now
risks designing against a shape that doesn't hold. Recorded here
explicitly per the user's instruction, so the "yes in principle" answer
isn't lost, without committing to an implementation prematurely. See
docs/TODO.md for what a future design would need to resolve (what
exactly gets promoted, in whose words, with what provenance).

---

**Decision:** The default user-facing recall experience reverted to the
one-shot agent (`run_agent`, `POST /api/recall`); the interactive
multi-round `RecallSession` code, tests, and API
(`/api/recall/sessions/*`) all remain in the codebase and fully
functional, just not wired into the `/recall` page's default UI.

**Reason:** The interactive-retrieval benchmark
(`eval/interactive_benchmark.yaml`, `eval/run_interactive_benchmark.py`)
measured the interactive mode *underperforming* both the semantic
baseline and the one-shot agent overall (44% vs. 61%/67% on an
18-question, deliberately ambiguous synthetic corpus) -- including
asking 5 clarifying questions on a direct, one-hop lookup that the other
two modes answered immediately, the specific failure the spec called
"crucial" to avoid. Root-caused (not just observed): the discriminating-
question/cue-boost heuristics only inspect `people`/`organizations`/
`locations`/`aliases`/`topics`, but an entity name frequently lands in
`relations` or `definitions` instead, which neither function checks (see
the trace visualization built from that benchmark run). Shipping the
underperforming mode as the default wasn't justified by the evidence;
shipping it at all (rather than deleting it) is, since the root cause is
specific and fixable, not fundamental to the approach.

**Alternatives considered:** Fix the field-coverage gap first, then
decide -- rejected for now because the fix needs validating against a
rerun of the same benchmark before it earns back default status, and
that's follow-up work, not something to rush ahead of reverting the
default.

---

**Decision:** `MAX_SESSION_ROUNDS = 12` is a runaway-loop safeguard on a
`RecallSession`, deliberately much larger than `MAX_AGENT_ITERATIONS = 3`
(the bounded *automatic* search loop within one round) and not exposed
as a user-facing "you get N questions" limit.

**Reason:** Spec section 10 explicitly distinguishes bounding automatic,
no-user-interaction iterations (latency/cost-sensitive, needs a small
bound) from bounding a user-mediated conversation (should continue as
long as it's actually productive). 12 is large enough that no normal
multi-round recall should ever hit it -- it exists purely so a
pathological case (e.g. a bug that keeps regenerating "fresh" attribute
values forever) can't spin forever, not to cap genuine back-and-forth.

---

**Decision:** Answering a storage-time clarification question
(`POST /memory-units/{id}/clarify`) is optional, enforced only as a
client-side preference (the "obligatory" toggle in the settings menu,
`app.js`'s `isObligatory()`/`localStorage`), never on the API itself. A
blank/omitted `answer` is accepted unconditionally by
`resolve_clarification()` and treated as "skip this question": nothing
is appended to `clarifications`, the unit just proceeds through
`_advance_unit()` exactly as if the question had never surfaced. When
the setting is "obligatory", the UI sets the answer input's HTML
`required` attribute and hides "Discard this one" -- obligatory means no
escape hatch at all (must answer), not just "discard is now the only
way out" (which was the *old* implicit behavior, when answering was
unconditionally required).

**Reason:** User-requested. The server-side requirement that an answer
be non-empty was never load-bearing for anything downstream -- enrich()
already tolerates a unit with zero clarifications (same code path as
"no question was ever asked"), so there was no correctness reason to
keep rejecting a blank answer at the API layer. Making the preference
purely client-side also means it's just UX, not a data-integrity rule;
nothing about stored data depends on which mode was active when a
question was skipped.

**Alternatives considered:** Enforcing "obligatory" server-side too (e.g.
a user-level setting persisted in Postgres, checked in
`resolve_clarification()`) -- rejected: this is a single-user localhost
MVP with no auth/user model to hang a server-side preference off of, and
the setting has no bearing on data correctness, so adding persistence
and a new guard clause for it would be solving a problem that doesn't
exist yet.

**Superseded:** the "obligatory" toggle was removed once clarification
moved entirely to the Browse tab (see "Clarification moved to Browse")
-- there's no longer a live, in-flight question on the Store page for
"obligatory" to gate at all. The underlying API behavior this decision
describes (a blank answer is always accepted, `resolve_clarification()`
never requires one) is unchanged; only the client-side toggle and its
effect on `required`/"Discard this one" visibility are gone -- Browse's
per-unit clarification form now always offers both answering and
discarding, with no obligatory mode.

---

**Decision:** Migrated the single generation model from
`qwen2.5:14b-instruct-q4_K_M` to `gemma4:12b` (Google, released June
2026), used for every generation/structured-output call exactly as qwen
was -- segmentation, the enrichment sub-calls, clarification-question
generation, agent decisions, answer generation, and now also image
captioning (see the next decision). `nomic-embed-text` is unchanged.

**Reason:** User-requested, specifically to gain native multimodal
(image/audio) understanding without a separate vision model -- gemma4
has an encoder-free architecture that understands images directly
through the same generation path already used for everything else,
rather than bolting on OCR/captioning as a third model. The user
explicitly considered and rejected pairing it with a second, smaller
Gemma 4 variant (e.g. `gemma4:e2b`) for "faster" sub-calls, discarding
that idea themselves before this was implemented.

**Alternatives considered / measured:** A dual-model setup (`gemma4:12b`
for heavy calls + a smaller `gemma4:e2b`/`e4b` for lighter ones) was
evaluated on paper before being discarded -- see the next bullet, VRAM
measurement makes this moot anyway. Published quantized-size estimates
for `gemma4:12b` suggested ~6.7GB VRAM, which looked like it might leave
room for a second small model alongside it; **measured directly**
(`nvidia-smi` while the model was loaded and actively serving a real
request, not just `ollama list`'s on-disk size) it actually uses ~8.7GB
-- the published estimate undercounted real usage by ~2GB, consistent
with this project's standing rule to trust direct measurement over
documentation (same lesson as the original qwen 7b/14b VRAM call). That
leaves only ~1GB of headroom on this 11GB card, confirming a second
generation model was never viable here regardless of the user's
decision to drop the idea -- it's now also an empirically closed
question, not just a judgment call.

---

**Decision:** Uploaded images are described by the generation model
itself (`app/ingestion/pipeline.py`'s `_describe_image`, via a new
`images: list[bytes]` parameter added to `LLMProvider.generate()`/
`generate_structured()`) immediately after the attachment is durably
stored, writing the caption into the same `Attachment.extracted_text`/
`extraction_status` fields PDF/text extraction already use. No new
schema, no new search code path -- an image becomes full-text and
chunk-searchable exactly the way a PDF page already is, as soon as it
has a caption instead of `extraction_status="unsupported"`.

**Reason:** This is the smallest change that makes images genuinely
searchable (closing the "No image understanding" gap in docs/TODO.md)
precisely because every downstream consumer of `extracted_text`
(`_attachment_text_for`, `chunk_attachment`, `_build_searchable_text`,
`hybrid_search_chunks`) is already format-agnostic -- it was written to
not care *how* the text arrived, so an image caption slots in with zero
changes to retrieval code.

**Large attachments -- how processing time is bounded:** Two independent
caps, both configurable in `app/config.py`, both only ever bound
*derived* processing cost, never the original file or its full
extracted text:
- `membase_max_images_captioned_per_capture` (default 8): one captioning
  call already costs O(1) regardless of an image's resolution (the
  vision path's own fixed tiling handles that internally) -- this instead
  bounds how many images in a single *upload batch* get that call made
  at all, so attaching dozens of images in one request can't turn into
  dozens of sequential vision calls with no visible limit. Images beyond
  the cap are stored normally, just without a caption (`extraction_
  status` stays `"unsupported"`) -- the same "best-effort, never lose
  the original" tolerance as every other derived step in this pipeline,
  not a new failure mode. Enforced per call to `_caption_new_images()`
  -- `create_capture()`'s batch of uploads, or one `add_attachment()`
  call -- not as a cumulative budget across a capture's lifetime;
  attaching images one at a time via repeated `add_attachment()` calls
  each gets its own fresh budget. The risk this exists for (many images
  in one request) is covered either way.
- `membase_max_chunks_per_attachment` (default 600, raised from 300 by
  user request): bounds how many
  `AttachmentChunk` rows one attachment's text gets split into and
  embedded as, so a single enormous document (a 1000-page PDF, say)
  can't turn into an unbounded-size `embed()` batch call. Critically,
  this never truncates the attachment's own full extracted text --
  `_attachment_text_for()` and therefore each memory unit's
  `searchable_text` still carry *all* of it, so coarse full-text search
  against the whole document is never diminished. Only the fine-grained,
  independently-chunk-embedded search granularity is capped -- content
  past the cap is still findable via the parent unit's full-text match,
  just not as its own standalone chunk hit.

Both are per-attachment/per-capture bounds, not global or time-based --
deliberately simple limits rather than a token-budget or wall-clock
timeout, since those would need the kind of empirical tuning (how long
is "too long"?) this project prefers to do only once there's a real
workload to tune against (same reasoning as the untuned chunk-size
constants already in `app/ingestion/chunking.py`).

**Alternatives considered:** Resizing/downsampling images client- or
server-side before sending them to the model -- rejected for now, since
`membase_max_upload_mb` (50MB, raised from 25MB by user request) already
bounds worst-case payload size
and the model's own vision pipeline handles arbitrary input resolution
internally; revisit only if a real file is observed to cause a
problem. A wall-clock timeout on the whole captioning/chunking phase --
rejected in favor of the simpler count-based caps above, consistent
with not reaching for time-based tuning without a real workload yet.

---

**Decision:** `OllamaProvider` calls Ollama's `/api/chat` endpoint
(a `messages` list, one role per entry) for both `generate()` and
`generate_structured()`, not `/api/generate` (a flat `system` +
`prompt` pair) -- a change made specifically for the gemma4 migration,
not carried over from the qwen setup.

**Reason:** Found by direct empirical testing, not assumed -- the first
full integration-test run after switching to gemma4:12b failed
`test_segmentation_acceptance_compound_vs_coherent`: three unrelated
facts that qwen reliably split into 3 units came back as 1, *despite*
the segmentation prompt containing a few-shot example using that exact
input text mapped to the correct 3-unit output. Reproduced directly
against the live model outside the app entirely: the identical
system+prompt+schema sent to `/api/generate` truncated an array-typed
structured output (a 3-item array collapsed to 1 item; a simplified
array-of-strings schema came back empty) -- but the same system
message, user message, and schema sent to `/api/chat` instead
reliably returned the full, correct array every time. This is
specific to how Ollama's chat-template substitution interacts with
gemma4's Modelfile template combined with grammar-constrained (JSON
schema) decoding via `/api/generate`'s flat fields -- `/api/chat`'s
explicit message roles sidestep whatever that interaction is. After
switching, the full integration suite (7 tests, all against the real
model) passed cleanly, including the originally-failing case.

This matters beyond segmentation: every structured-output call in this
app (`enrich()`'s three sub-calls, `generate_questions()`, the agent's
evaluate/answer steps, cue extraction) uses `generate_structured()`
with a JSON schema, and several of those schemas have array-typed
fields (aliases, people, topics, relations, definitions, questions) --
had this not been caught by running the real integration suite (not
just the fast suite's fakes, which can't exhibit this at all), the
migration would have shipped with a silent, pervasive quality
regression across most of the app's actual output, not just
segmentation.

**Alternatives considered:** Working around it by restructuring the
segmentation schema (e.g. a single delimited string instead of a JSON
array) -- rejected once `/api/chat` was confirmed to fix the root
cause cleanly for every schema shape, not just this one; patching
around a single call site would have left the same latent bug waiting
in every other array-typed schema in the app.

**Lesson for future model migrations on this project:** the fast
test suite's `FakeLLMProvider` cannot catch this class of bug by
construction -- it never talks to Ollama at all. Only the real-model
integration suite (`pytest -m integration`) can, and it must be run to
completion (and alone -- see the "shared dev database" caution in
`backend/tests/conftest.py`'s `db` fixture, which `TRUNCATE`s on every
test and will corrupt a concurrently-running suite sharing the same
Postgres instance) before trusting a generation-model swap, no matter
how plausible the plumbing looks.

---

**Decision:** `clarify.generate_questions()` and `enrich()` both take a
required `attachment_text` parameter (not an optional kwarg) -- every
caller must pass whatever's already known about attached files,
including an uploaded image's caption (`_describe_image`), so a vague
reference in the memory's text that an attachment already resolves is
never flagged as needing a question, and semantic extraction
(meaning/aliases/relations) isn't blind to attachment content either.

**Reason:** User-raised with a concrete example: uploading a photo of a
red shoe with the text "I like shoes of this colour" previously still
triggered a clarification question asking which colour was meant, even
though the caption (`_describe_image`, run before segmentation --
see "Gemma 4 migration" above) already answers it. Root cause:
`attachment_text` was computed in `pipeline.py` (`_attachment_text_for`)
and threaded into `_run_full_enrichment` for building `searchable_text`
(a plain string concatenation), but never into the two LLM calls that
actually make judgment calls about the memory's content --
`generate_questions()` and `enrich()`. Full-text search over an
attachment already worked; the model's own understanding of what the
memory means did not.

**Deliberately minimal prompt engineering, maximal code-level
guarantee** (explicit user direction): the fix is not an elaborate
few-shot example tuned to the shoe scenario -- the system prompts
gained one short, general instruction each ("if attached file content
is given, treat it as already-known information, not something to ask
about"). The actual guarantee is structural: `attachment_text` is a
*required* positional parameter, so a caller cannot forget to pass it
without a `TypeError` at call time, not a silent gap; and pipeline.py's
call order (`_caption_new_images()` always completes, inside the same
request, before segmentation or `_advance_unit()`/`generate_questions()`
ever runs) guarantees the text is complete and available by the time
either call happens. Verified against the live model with the user's
own example (red-shoe photo + "I like shoes of this colour"): no
clarification question was asked, and the resulting semantic meaning
correctly said "red".

**Alternatives considered:** A deterministic heuristic to detect vague
references ("this colour", "that thing") and suppress clarification
when an attachment is present -- rejected as exactly the kind of
brittle, false-positive-prone pattern-matching the "minimal prompt
engineering, maximal hard guarantee" direction was steering away from;
judging whether an attachment actually resolves a given reference is
inherently a semantic question with no reliable non-LLM shortcut, so
the LLM call stays, but the *data it's given* and *when it's called*
are now both hard-coded guarantees rather than hoped-for prompt
behavior.

---

**Decision:** Offload non-decision-making derived processing out of the
synchronous request path, to minimize latency from a capture's first
submission through its last answered clarification question.
Specifically: `_index_attachment_chunks()` (chunk-splitting + embedding
an attachment's text for fine-grained search) no longer runs inline
during `create_capture()` -- it's now `finish_attachment_chunking()`,
a background task scheduled by `api/captures.py` exactly like unit
enrichment already was.

**Reason:** User-requested: "minimize the time it takes to process a
memory from the first user input... to the last user input... offload
most of the calculation to the memory processing stage." Chunk
embedding has no bearing on segmentation or clarification decisions
(unlike image captioning, which must stay foreground -- see the
decision above), so there was no correctness reason for it to block
the response the user is waiting on. `add_attachment()` (attaching a
file to an already-"ready" capture, a separate interaction from the
create-then-clarify flow this request was about) was deliberately left
unchanged -- it's a single file at a time, not part of the latency path
the user described, and moving it would have meant either changing its
return signature (breaking callers) or inventing a side-channel for a
case that isn't the actual complaint.

**What stays foreground, and why:** segmentation (`segment_capture`)
and the clarification-question check (`generate_questions`) are the
irreducible part -- the whole point of the earlier clarify/enrichment
split was already to make the user-facing decision as fast as possible
(see the clarification/enrichment split decision elsewhere in this
file). Image captioning also stays foreground, specifically *because*
of the decision above: it must complete before `generate_questions()`
and `enrich()` run, or the attachment-context fix doesn't work. This is
a real, acknowledged trade-off, not an oversight -- captioning is
bounded (`membase_max_images_captioned_per_capture`, default 8) so its
worst case is known, but it is real latency in the critical path.

**Recovery gap, noted not fixed:** unlike unit enrichment
(`finish_unit_processing`, recoverable via `POST /retry` if a server
restart drops the background task), there is currently no equivalent
retry endpoint for a dropped `finish_attachment_chunking` task -- an
attachment would just stay without chunks (still fully full-text
searchable via its `extracted_text`, just missing the fine-grained
chunk-level index) until the next whole-attachment reprocessing
mechanism exists. Acceptable for now at this MVP's expected restart
frequency; flagged in docs/TODO.md rather than silently left unnoted.

---

**Decision:** A client disconnecting mid-request does not abort
server-side processing or lose work -- confirmed by inspecting the
code (no `request.is_disconnected()` checks or cancellation handling
anywhere in `app/`), not assumed. An `async def` endpoint handler that
is simply `await`-ing a sequence of calls keeps running to completion
even if the client goes away; only the final HTTP response write is
wasted, not the work itself. Authoritative content was already durably
committed before any derived/LLM-dependent step runs (the two-phase-
commit pattern, see docs/ARCHITECTURE.md), so even a disconnect during
the slowest part of ingestion never loses the user's own words -- the
capture just finishes processing invisibly, and the `/captures` Browse
page is the way to find it again afterward.

**Reason:** User asked for "general robustness checks and edge case
handling... since the memory storage process can be very time
intensive and disconnections could happen in between." Rather than add
explicit disconnect-detection/cancellation (which would require
deciding what to do with partially-committed work mid-transaction, a
much larger and riskier change), the simpler and already-true property
is: this app never had a mechanism that would lose work on disconnect
in the first place, because it never explicitly cancels anything.
Documented here so this property doesn't get accidentally broken by a
future change that starts checking `request.is_disconnected()` without
realizing it would need to handle a half-finished transaction.

---

# Streamlining phase ("too bloaty" pass)

Everything below this point was made under one combined mandate: "reduce
unnecessary complexity and latency while preserving the architectural
ideas that make this an augmented-memory system rather than a
conventional notes/RAG application." The five abstractions the whole
pass was required to justify itself against: Capture, MemoryUnit,
Semantic representation, RecallSession, LLM/Database. Each decision
below names which of those it serves, or what it removed.

---

**Decision:** Collapse the five-call ingestion pipeline (`segment_capture`
+ `generate_questions` + `enrich()`'s three narrative/entities/
retrieval-aids sub-calls) into two calls total across a unit's lifecycle:
`analyze_capture()` (once per capture -- segmentation + full per-unit
semantic analysis + clarification decision, all in one structured-output
call) and `reanalyze_unit()` (once per clarification round -- re-derive
semantics and decide if a further question is warranted). Lives in the
new `app/ingestion/analysis.py`, replacing `clarify.py`/`enrichment.py`/
`segmentation.py` (all deleted).

**Reason:** An ordinary memory with no clarification need previously
cost 4 generation calls (segmentation, question-check, 3x enrichment)
plus 1 embed; it now costs 1 generation call plus 1 embed. This was the
single largest latency/call-count lever available, and directly serves
the "storage should be cheaper than recall" pressure the spec asked for.

**What was preserved, deliberately:** the verbatim-excerpt safety check
(`is_verbatim`, moved not rewritten) still rejects any proposed unit that
isn't an exact substring of the raw capture text, falling back to one
whole-text unit -- the collapse changes how many calls decide
segmentation, never the safety property that segmentation can't rewrite
authoritative content.

**Alternative considered:** Keep 3 calls (segmentation, one combined
enrichment+clarification). Rejected -- once clarification-need detection
and semantic enrichment are looking at the same text for the same
purpose (deciding what's worth storing/asking about), splitting them
into two calls bought no quality the eval set could detect, only an
extra round-trip.

---

**Decision:** Clarification is strictly sequential -- one question
surfaced at a time, answered, the unit re-analyzed exactly once
(`reanalyze_unit`), which both re-derives semantics AND decides whether a
*further* question is still needed. The old `clarification_queue` JSONB
column (a pre-generated batch of future questions) is dropped from the
schema entirely; `MAX_QUESTIONS_PER_ROUND` no longer exists because
there's no queue to bound.

**Reason:** Pre-generating several questions up front commits to asking
all of them even if the first answer makes the rest moot (e.g. answering
"what's Flimvex?" with "it's my gym locker PIN" can retroactively make a
second pre-generated question about "which gym" either still necessary or
suddenly answerable from context) -- re-evaluating after every answer is
strictly more adaptive, at the cost of one call per round instead of
zero marginal cost for a queued question. `MAX_CLARIFICATION_ROUNDS = 3`
(unchanged) still bounds how many rounds the system may *initiate*.

**Clarification outcomes:** a deterministic, regex-based (no LLM call)
classifier (`_classify_outcome` in `app/ingestion/pipeline.py`) recognizes
`declined` ("rather not", "none of your business", ...) vs. `unknown`
("I don't know", "not sure", ...) vs. `answered` from the free-text
answer. A field marked declined/unknown is hard-coded into
`declined_or_unknown_fields` and the next round's question is
*suppressed* if it targets that same field, even if the model proposes it
again -- this is a code-level guarantee, not a prompt instruction, because
the entire point of tracking the outcome is to make re-asking impossible,
not merely discouraged. (`UNCLEAR`/`CORRECTION` from the spec's suggested
outcome vocabulary are represented as: an empty/ambiguous answer that
doesn't match either pattern falls through to `answered` as a best-effort
outcome rather than a fourth bucket, and `correct_unit`'s existing
free-form correction mechanism already covers `CORRECTION` -- adding a
fourth JSON-level outcome tag would have doubled the state space for no
behavioral difference the current UI/eval set exercises.)

---

**Decision:** Direct structured metadata entry (Location, Date, People,
Organizations, Topics) is accepted as an optional JSON-encoded `metadata`
form field on `POST /api/captures`, stored verbatim on
`MemoryUnit.explicit_metadata`, and always takes authority over whatever
the same analysis call separately infers for the same semantic field. A
genuine conflict (both sides non-empty and different) is surfaced as a
synthetic clarification question (`_apply_explicit_metadata` in
`pipeline.py`) rather than silently resolved either way; in the interim,
both values are kept in the semantic field (not just the explicit one)
so nothing the model inferred is silently discarded while the question is
still open.

**Reason:** Spec-mandated explicit authority ordering. Minimal API
surface: no new endpoint, one optional form field, parsed defensively
(malformed JSON is treated as "none given," not a 400) since metadata is
optional and shouldn't be able to fail the whole submission.

**Verified against the real model:** submitting `{"location": "Zürich"}`
against text whose body read like an office memo (model inferred
`location: "office"`) produced exactly the designed clarification --
`"You entered location='Zürich', but the text seems to say
location='office'. Which should be associated with this memory?"` -- with
`semantic.locations` holding both values pending the answer. Confirmed
live against `gemma4:12b` via the dev server, not just the fake-LLM test
suite.

---

**Decision:** A cheap, pre-LLM lexical lookup (`_find_related_context` in
`pipeline.py`) scans the new text for capitalized/proper-noun-shaped
terms and does a plain `ILIKE` search against `status='ready'` units'
`retrieval_text`, returning up to 3 short snippets. These are injected
as a labeled "potentially related earlier memories" context block into
both `analyze_capture()` and `reanalyze_unit()`'s prompts, explicitly
marked as context the current text overrides on conflict -- never as
authority.

**Reason:** Prevents the system from re-asking about a term it already
has an explanation for (e.g. "Zorblex" clarified once, mentioned again
later) without paying for a semantic/vector search or a second LLM call
-- regex + `ILIKE` is pure SQL. Deliberately NOT `hybrid_search()`: a
full retrieval round for every ingested memory would undo the "storage
should be cheap" pressure this whole phase was about.

**Prior memory is context, not authority:** this is enforced by prompt
wording only (there is no code-level override, unlike the explicit-
metadata case above) -- a deliberate difference, since prior memory is
inherently soft background ("have you perhaps already explained this
term"), not a structured field with a definite right answer the way
explicit metadata is.

---

**Decision:** Lightweight contradiction detection at storage time:
after a unit embeds successfully, `_detect_conflicts()` extracts
`(subject, predicate) -> object` triples from its own `semantic.relations`
and searches for other `ready` units whose relations share a
(subject, predicate) key but disagree on the object. A match adds a
purely advisory `{"unit_id", "note"}` entry to `potential_conflicts` on
*both* units -- never deletes, supersedes, or hides either side.

**Reason:** Spec-mandated; "two memories disagree" must be surfaced, not
silently resolved by picking one. Reuses the existing `hybrid_search()`
call (no new retrieval primitive) and does the actual conflict
comparison in plain Python (string equality on already-extracted
fields) -- no extra LLM call for the comparison itself.

**Bug found and fixed during testing:** the first implementation
searched using the new unit's full `retrieval_text` as the query. That
text *includes the differing object itself* (e.g. "...1400 dollars..."),
so a full-text search for it can structurally never match a prior memory
that says "...1200 dollars..." -- the one word that would prove the
match is exactly the word guaranteed not to appear in the thing you're
searching for. Fixed by searching on the relation's subject+predicate
only (e.g. "rent is"), which is stable across the disagreement and
actually finds the other side. Caught by a dedicated test
(`tests/test_contradiction.py`) that deliberately exercises two
differently-valued memories about the same subject, not by inspection --
the original version passed every "no false positive" test and quietly
failed every true positive until this was added.

**Retrieval-side surfacing:** `_format_unit()` in
`app/retrieval/pipeline.py` appends a `[Conflict] ...` line (plain string
formatting, no LLM call) for any evidence unit carrying
`potential_conflicts`, and the answer-generation system prompt was
extended with one explicit instruction: a `[Conflict]` line means two
stored memories disagree and the answer **must** state that disagreement
rather than silently picking a side. This is the only place conflict
presentation happens, keeping "represent disagreement explicitly" a
single code path rather than something every caller has to remember to
handle.

**Not built, and why:** no truth-maintenance database, no temporal
belief revision, no automatic "newest wins" rule, no supersession. See
docs/TODO.md for what evidence would justify any of those.

---

**Decision:** `RecallSession` (`app/retrieval/session.py`) is now the
canonical retrieval architecture end to end, including for the
"stateless" one-shot endpoint. `POST /api/recall` no longer calls
`run_agent()` directly -- it calls `start_session()`, runs exactly the
same first automatic round a multi-turn session would (same ambiguity
narrowing, same grounding decision), and reports the outcome in the
flat legacy response shape. The session row is still persisted (it's
recall history, same as any other session), but its id is never
returned to a one-shot caller, so there's no way to continue it from
that endpoint.

**Reason:** Spec-mandated: "No two independent retrieval algorithms."
Before this change, the one-shot path and the session path made the
same *searches* (both called `automatic_search()`) but reached different
*decisions* from the results -- the one-shot agent had no ambiguity
detection, no discriminating-question logic, and no cue handling at all,
so a genuinely ambiguous query got a possibly-wrong best-guess answer via
`/api/recall` and a clarifying question via `/api/recall/sessions`,
depending purely on which endpoint happened to be called for the exact
same stored memories.

**`run_agent()` is retained as an internal primitive** -- a one-shot
search-and-answer with no ambiguity narrowing and no persisted session
row -- for callers that deliberately want exactly that and not the
session machinery: the eval harness (`eval/run_eval.py`) and latency
benchmarks. It is no longer used by any HTTP-facing path.

---

**Decision:** A deterministic bypass (`_is_high_confidence` in
`app/retrieval/agent.py`) skips the `evaluate_sufficiency()` LLM call
entirely when the top candidate's fused score clears `0.80`, its margin
over the runner-up (if any) clears `0.20`, and its semantic analysis
isn't a degraded (`enrichment_status == "unavailable"`) fallback. All
three conditions must hold.

**Reason:** Spec-mandated "deterministic high-confidence short-circuit
...using a combination of absolute relevance + margin to second
candidate + candidate evidence completeness (not one arbitrary score
alone)" -- a single-field check (score alone) would risk accepting a
confidently-wrong top hit; requiring margin too means a close second
candidate always still gets the reasoning step; requiring non-degraded
evidence means a unit that only exists because the LLM was unavailable
during ingestion never gets to skip the one place that was going to
sanity-check it.

**Calibration status:** thresholds are an initial engineering default,
sanity-checked (not swept) against the real model via
`scripts/run-eval.sh` -- the full 30-case eval set passed 28/30 with this
shortcut active (see "Performance and quality snapshot" below). A
genuine threshold sweep (vary both constants, measure false early
acceptance / missed early acceptance / latency delta across the eval
corpus) is in docs/TODO.md as the natural next experiment, not done here
-- the corpus is small enough that a swept threshold would risk
overfitting to 30 hand-written cases rather than reflecting a real
distribution.

---

**Decision:** `MAX_AGENT_ITERATIONS` reduced 3 -> 2, and
`MAX_REFORMULATIONS_PER_ROUND` reduced 2 -> 1 (`app/retrieval/agent.py`).

**Reason:** Spec-mandated initial defaults ("a query that still isn't
answerable after one reformulation is better handed to the user as a
cue request than burned on a third automatic guess"). Not re-tuned
against the eval corpus beyond the one full run reported below.

**Measured effect, honestly reported:** the same 30-case eval set that
passed 30/30 in earlier phases under the old 3/2 budget passed 28/30
under the new one-call-ingestion + 2/1-iteration architecture combined.
The two new failures:
- `clarification_alias_second_term`: the single unified `analyze_capture`
  call chose not to flag "Flimvex" as needing clarification at all (the
  old dedicated question-generation call, with no other task competing
  for its attention, apparently did) -- so the test's scripted
  clarification answer was never consumed, and the model answered from
  the unexplained term alone. This is an ingestion-collapse quality
  trade-off, not an iteration-budget one.
- `user_specific_terminology` ("What did I call the project?" ->
  "Nightingale"): abstained instead of answering. Plausibly the reduced
  reformulation budget (1 instead of 2) ran out before finding a
  phrasing that bridges "what did I call X" to the stored alias -- but
  this wasn't isolated by re-running with the old budget, so it's
  reported as a correlated observation, not a proven cause.

Both are regressions worth watching, not reverted: 28/30 (93%) on a
30-case hand-written set is within the range where one or two cases
flipping on model sampling variance alone is plausible even with *no*
code change, and the spec explicitly traded some of this quality margin
for the call-count/latency reduction on purpose. Recorded in
docs/TODO.md as the concrete starting point for a real before/after
comparison (same cases, old budget vs. new, multiple runs each) if this
turns out to matter in practice.

---

**Decision:** Cue-to-candidate matching (`app/retrieval/discriminate.py`)
gets a unified fallback: `_matches_cue()` first checks for an exact match
against the unit's structured semantic fields (unchanged, precise), and
-- new -- if that fails, checks whether the cue appears as a normalized
substring of the unit's own `retrieval_text` at all (skipped for cues
under 3 characters, to avoid short-token noise).

**Reason:** Spec-mandated: "use a unified semantic representation for
matching... not more field-specific heuristics." The structured-field
list (`people`/`organizations`/`locations`/`aliases`/`topics`/
`relations`/`definitions`) already grew once this session to fix a real
bug (an entity's name landing in `relations`/`definitions` instead of a
flat list field) -- rather than keep discovering new fields that need
adding to the list, the `retrieval_text` fallback catches anything a cue
names that the extraction step didn't separately pull into *any* field,
since `retrieval_text` already contains everything the semantic
representation knows in one place.

---

**Decision:** `is_ambiguous()` now requires the runner-up candidate's own
fused score to clear an absolute floor (`AMBIGUITY_MIN_RUNNER_UP_SCORE =
0.45`), in addition to the existing close-margin check.

**Reason:** Spec-mandated: "don't treat the fused score as a perfect
probability... consider... score margin [not in isolation]." Without
this, two candidates that are both weak, barely-above-the-relevance-bar
matches (e.g. 0.20 and 0.19) would trigger a clarifying question just
because they happen to be close to *each other* -- which is "nothing
solid was found," not "two strong contenders." The floor means a close
margin only counts as genuine ambiguity when both sides are independently
strong matches.

**Metadata overlap, considered and not built:** the spec also suggested
weighing metadata overlap between the top two candidates into the
ambiguity decision. Investigated and deliberately not added: the
existing downstream behavior (`generate_discriminating_question`
returning `None` when every candidate shares a value, at which point
`session.py` falls through to answering with both candidates' evidence
rather than narrowing) already produces the right outcome for the
"candidates are metadata-identical" case without `is_ambiguous()` itself
needing to know about it. Adding an early-exit for that case in
`is_ambiguous()` would have *changed* that outcome (triggering
`narrow_to_dominant_group()` instead, discarding the second candidate's
evidence) for no accuracy benefit found in testing -- see docs/TODO.md.

---

**Decision:** Recall-session state is never promoted into a MemoryUnit
automatically. The one explicit mechanism for turning something from a
recall session into a memory is `POST /api/recall/sessions/{id}/
save-as-memory`, which creates a brand-new `Capture` through the
*ordinary* `create_capture()` pipeline (segmentation, clarification,
enrichment, provenance -- identical to `POST /api/captures`). The
request `text` must exactly match something already in that session's
own `messages` (a user turn) or `cues` (an extracted cue) --
`app/retrieval/session.py`'s `user_supplied_texts()` -- rejected with 400
otherwise, so this can never be used to inject arbitrary or agent-
generated text (a hypothesis, a system question) as if the user had said
it.

**Reason:** Spec-mandated: never bypass the normal pipeline, and never
auto-promote. The validation against `user_supplied_texts()` is the
actual safety property -- it's not enough that the *action* is user-
initiated (a button click); the *content* being saved must independently
be traceable to something the user actually typed in this session.

**Deferred, and why:** the frontend only ever uses the one-shot
`/api/recall` flow today (`app/static/app.js`'s `initRecallPage`) --
there is no UI that displays a session's cues or message history at all
(the multi-turn `RecallSession` API is fully implemented and reachable,
just not this page's default, per an earlier phase's benchmark). Wiring
a "Save as memory" *button* into the UI would require either building
that session-history UI (a materially larger piece of work than this
feature itself) or threading a session id back out of the deliberately
anonymous one-shot endpoint (undoing the "no way to continue a one-shot
call" property decided above). Per the spec's own escape valve ("if this
would materially complicate the current UI/API, document as a small
deferred item"), the backend capability is complete and tested
(`tests/test_save_as_memory.py`); the UI affordance is deferred -- see
docs/TODO.md.

---

**Decision:** `hybrid_search()` and `hybrid_search_chunks()`
(`app/retrieval/pipeline.py`) now accept an optional `query_vector`
parameter via a new `embed_query()` helper; `automatic_search()`
(`app/retrieval/agent.py`) embeds each query string once per iteration
and passes the resulting vector into both calls instead of each one
independently calling `llm.embed()` for the same text.

**Reason:** Spec-flagged as "low risk, should be implemented" -- a
second embedding call for text that was just embedded a moment ago is
pure waste. A sentinel default (`_UNSET`, not plain `None`) distinguishes
"caller didn't pass a vector, embed it yourself" from "caller already
tried and the embedding is known unavailable" -- plain `None` couldn't
carry both meanings without either silently retrying a known-failed
embed or breaking the handful of call sites that still call either
function standalone (tests, `eval/run_eval.py`-style direct calls),
which continue to work unchanged.

---

**Decision:** The long-held PostgreSQL row lock
(`with_for_update(...)` for the full duration of an LLM call) is replaced
with a short-lived processing lease: `_claim_processing()` sets
`status="processing"` plus a fresh random `processing_token` and
`processing_started_at` in a short transaction and commits; the LLM work
(now just one embedding call, see the ingestion-collapse decision above)
runs with no transaction open at all; `_commit_if_token_matches()` does a
single `UPDATE ... WHERE id=? AND processing_token=?` and checks
`rowcount > 0`. A token mismatch means a concurrent attempt already
finished first -- the result is simply dropped, never applied on top of
whatever already landed. `retry_enrichment`/`POST /retry` only reclaims a
unit stuck in `processing` if `processing_started_at` is older than
`membase_processing_lease_stale_seconds` (300s).

**Reason:** Spec-mandated: "a short-lived processing lease/token pattern
...preserves race safety without holding a lock across expensive model
calls." A row lock held for an LLM call's duration (previously up to
several calls, now one) blocks any other transaction touching that row
-- including, in the worst case, an unrelated read -- for exactly as long
as the slowest external dependency this process doesn't control takes to
respond. The lease pattern bounds the *lock* duration to two fast local
transactions, independent of how long Ollama takes.

---

**Decision:** Image captioning moved from the synchronous ingestion path
to a fully independent background task, `finish_image_captioning()`,
scheduled by the API layer alongside (not chained after) attachment
chunking. An uploaded image is stored and its sibling text memory forms
immediately regardless of whether captioning ever succeeds; a captioning
failure leaves the image attached and uncaptioned, never blocks or fails
the capture.

**Reason:** Spec-mandated: "an image should first be stored safely; the
memory must not remain blocked because image captioning is unavailable."
This directly reverses an earlier phase's decision (captioning was made
*synchronous*, specifically so `clarify`/`enrich` could see the caption
as context and avoid an avoidable clarification question about something
a photo already resolved). Both decisions were correct for the problem
they were solving; the explicit trade-off accepted here is that the
*first* analysis pass for a capture with images now runs without their
captions as context, so a vague reference an image would have resolved
may get asked about anyway. Retrieval quality is not affected the same
way: the caption still becomes fully searchable once captioning
completes.

**Gap found and fixed while implementing this:** a caption arriving after
its sibling unit already embedded would otherwise never become
searchable at all for a *short* caption -- too short to trigger chunking
(see the lazy-chunking decision below), and the unit-level embedding that
would have included it as an excerpt already happened before the caption
existed. Fixed by having `finish_image_captioning()` reuse the same
light re-indexing path `add_attachment()` uses for a posthoc file:
rebuild each ready sibling unit's `retrieval_text`/`searchable_text` with
a trailing excerpt and re-embed, without re-running full analysis.
Extracted into a shared `_refresh_ready_units_with_attachment_excerpt()`
so both callers stay in sync.

---

**Decision:** Attachment chunking is lazy/conditional: `should_chunk()`
skips chunk-level embedding entirely when the attachment's extracted
text is under `membase_min_chars_for_chunking` (2000 chars) -- the parent
unit's own full-text search already covers a document that short.
`_index_attachment_chunks()` was made idempotent (clears any existing
chunks for the attachment before re-inserting), which is what makes
`retry_attachment_chunking()` safe to call again after a dropped
background task, without a separate pending/processing status column for
chunks.

**Reason:** Spec-mandated: "a simple deterministic size threshold is
sufficient; don't deeply process a tiny file," plus a recovery mechanism
that's "no job queue... a simple state machine or retry endpoint... or
make chunking naturally idempotent." Idempotency-via-retry was chosen
over a new status column: a chunk-having attachment and a chunk-lacking
one are already distinguishable by `AttachmentChunk` existence, so a
status column would track information already recoverable by querying
the thing it's redundant with.

**Not done:** `retry_attachment_chunking()` exists and is tested but has
no API route wired to it yet -- see docs/TODO.md.

---

## Performance and quality snapshot (post-streamlining)

Measured against this phase's final code, real `gemma4:12b` +
`nomic-embed-text` via Ollama, `scripts/run-eval.sh`'s 30-case corpus
(`eval/cases.yaml`):

- **28/30 eval cases passed** (93%), full transcript and the two
  failures' analysis in the iteration-budget decision above.
- **LLM call count per ordinary memory (no clarification, no
  attachment):** 1 generation call (`analyze_capture`) + 1 embedding
  call, down from 4 generation calls + 1 embedding call before this
  phase (`segment_capture` + `generate_questions` + 3x `enrich()`
  sub-calls + embed). A memory needing one round of clarification adds
  exactly 1 more generation call (`reanalyze_unit`) either way.
- **LLM call count per simple one-shot recall:** best case (the
  deterministic high-confidence shortcut fires) is 1 embedding call + 1
  answer-generation call, with zero `evaluate_sufficiency` calls -- down
  from a mandatory sufficiency-evaluation call every time. A query
  needing reformulation now costs at most 1 extra search round (down
  from up to 2), each round re-embedding at most once per query string
  (see the embedding-dedup decision above, down from twice).
- **Contradiction detection adds, at most, the cost it already reuses:**
  one `hybrid_search()` call per newly-finalized unit with at least one
  relation -- no new retrieval primitive, no extra LLM call for the
  comparison itself.
- Not measured: wall-clock latency under the exact same conditions
  before vs. after (no clean "before" baseline was re-run this phase --
  reverting the full call-collapse to get one would have cost more of
  this phase's budget than the measurement was worth, given the call-
  count reduction is the more direct and less noisy signal). Flagged in
  docs/TODO.md as the natural next experiment if latency numbers are
  specifically needed later.

---

**Decision:** "Multi-file posthoc attachment" -- `POST /api/captures/
{id}/attachments` now accepts multiple files in one call
(`add_attachments()` in `app/ingestion/pipeline.py`), not just one.
Validation and storage are atomic across the whole batch (one
unsupported/oversized file rolls back the entire call, matching
`create_capture()`'s own existing multi-upload handling) -- rather than
leaving a confusing partial result where some of the files from one
request got attached and one didn't. Best-effort derived processing
(captioning, chunking) stays per-file after that, same tolerance as
everywhere else. Sibling units are re-embedded exactly ONCE after the
whole batch finishes, not once per file -- N files no longer means N
redundant embedding calls. `add_attachment()` (singular) is now a thin
wrapper over this, so no existing caller changed behavior.

**Reason:** User-requested ("add the option to upload and attach
multiple files to a memory"). Create-time multi-file upload already
existed (`POST /api/captures` took `files: list[UploadFile]` from the
start); the gap was specifically the posthoc path, which only ever took
one file and had no UI at all. A UI control (file picker + "Attach
files" button) was added for this -- initially placed on the Browse
page, then moved to the Store page's own post-save result card after
feedback that it belongs where the user's attention already is right
after saving (see "Store page simplified" below).

---

**Decision:** The combined attachment text fed into generation prompts
(`analyze_capture()`'s initial call, and both `reanalyze_unit()` call
sites -- clarification resolution and correction) is now capped at
`MAX_ATTACHMENT_TEXT_FOR_ANALYSIS_CHARS` (1500 chars), via a new
`_attachment_text_for_analysis()` that truncates. The full-text search
index (`_build_searchable_text`, via the uncapped `_attachment_text_for()`)
is unaffected -- Postgres full-text search has no model-context budget
to protect.

**Reason:** A generation call's context window is shared between its
input prompt and its output -- the same 4096-token budget that caused
the "20 facts collapse to one blob" bug fixed earlier (see "Chunk long
captures before analysis" above). That fix was about OUTPUT size; an
uncapped `attachment_text` is the INPUT-side version of the identical
risk -- a large PDF, or several attachments concatenated with no limit,
crowds out the budget the model needs for its own structured-JSON
response, on every analysis call a capture's units ever make for their
whole lifetime (not just once at ingestion). Found by inspection while
answering a direct question about it, not by a reported failure --
fixed proactively since the multi-file attachment feature above makes
hitting this materially more likely (more attachments concatenated
together per capture, no cap).

---

# The "membase structurer" (post-save deep enrichment)

**Decision:** Add a second, independent background layer -- the
*structurer* -- that deepens an already-`"ready"` (searchable) unit's
representation without ever affecting whether it's searchable.
User-mandated: "this process should run in parallel to the membase
hosting, and memory recall should be able to happen to both processed
and unprocessed memories at any time."

**What it does (v1 scope):**
- Re-derives a richer semantic representation (`app/ingestion/
  structurer.py`'s `structure_unit()`) -- the same `Semantics` shape the
  fast path produces, but from one dedicated, unhurried call that isn't
  sharing a context budget with segmentation or clarification decisions.
  Explicitly allowed to be "LLM heavy" (user's words) -- unlike every
  other call in this codebase, this one has no latency pressure, since
  nothing user-facing is waiting on it.
- Finds candidate related memories via the *existing* `hybrid_search()`
  call (same reuse pattern as contradiction detection -- no new
  retrieval primitive) and asks the model which of them are genuinely
  related, storing the result as `related_unit_ids` (a flat, bounded
  list -- advisory cross-references, explicitly NOT a knowledge graph,
  consistent with the streamlining phase's "no knowledge graph" rule).
- Rebuilds `retrieval_text`/`searchable_text`/`embedding` **in place**,
  via the same lease/token commit pattern `_finalize_unit()` already
  uses (`structure_token`/`structure_started_at`, parallel to
  `processing_token`/`processing_started_at`) -- so there is no window
  where the unit is unsearchable or carries a half-written mix of old
  and new fields. `hybrid_search()` itself needs zero changes: a
  structured unit is just a `ready` unit with better data in the exact
  same columns retrieval already reads.

**Trigger -- "should start for any memory as soon as it is stored":**
scheduled as a FastAPI BackgroundTask immediately after
`finish_unit_processing` for the same unit, in the same request
(`api/captures.py`'s `create()`, and the clarification-resolution/
correction/retry paths). Starlette's `BackgroundTasks` run sequentially,
in registration order, within one request -- so scheduling
`finish_structuring` right after `finish_unit_processing` guarantees the
unit is already `"ready"` (has real `semantic`/`retrieval_text` to build
on) by the time structuring actually runs, with no polling, no queue,
and no new infrastructure needed to express that ordering.

**No new infrastructure:** still a FastAPI BackgroundTask, own DB
session, own lease -- not a worker daemon, not Celery/Redis (CLAUDE.md's
existing constraint, unrevisited). "Runs in parallel to the hosting"
is satisfied by being backgrounded and never touching the request/
response path or the search path, not by being a separate process.

**State, deliberately separate from the existing `status` gate:** a new
`structure_status` column (`unstructured` | `structuring` | `structured`
| `failed`) tracks the structurer's own progress. It is NEVER read by
`hybrid_search()` and never changes `MemoryUnit.status` -- the one and
only searchability gate remains `status == "ready"`, exactly as before
this feature existed. A unit can sit in `structure_status="unstructured"`
indefinitely (structuring never ran, dropped, or still queued) and
remain fully, normally searchable the whole time.

**Deferred, and why (not silently skipped):** structurer-driven
*re-segmentation* -- splitting an existing ready unit into more units if
the deeper pass decides the original boundary was wrong. The user's
spec explicitly scoped this to "if needed and providing meaningful
quality improvement," and on inspection it runs into the one invariant
this whole codebase treats as non-negotiable: `authoritative_content` is
never rewritten. Splitting a unit after the fact would require either
violating that (shrinking the original unit's content) or inventing a
"supersede the old unit with new siblings, retire it" lifecycle that
doesn't exist yet and has real edge cases (existing
`potential_conflicts` entries and `RecallSession.final_sources`
snapshots reference unit ids directly, by id -- a retired unit id must
stay resolvable or those go stale). Not built this pass; see
docs/TODO.md for what a safe version would need.

**Recovery:** `POST /api/memory-units/{id}/restructure` reclaims a unit
stuck in `structure_status="structuring"` past a staleness window, same
pattern as the existing `/retry` endpoint for the fast pipeline's lease.

**Visualized only on the Browse page** (per explicit instruction, and
consistent with the Store page's own simplification below): a
`structure_status` badge per unit, read at page-load time, same as the
existing `status` badges -- no polling added for this (matches the
Browse page's existing no-polling design); reloading the page shows
current state.

---

**Decision:** The Store (create) page's UI was simplified alongside the
structurer work: it no longer waits for or polls background processing
status at all, and the old "processing... / stuck after 10 minutes /
retry" UI was removed from this page entirely (`pollUnitUntilDone`,
`MAX_POLL_MS`, the `.capture-processing` block). It now shows "Saved" --
with the real derived semantic info -- as soon as every pending
clarification question (if any) is resolved.

**Reason:** `authoritative_content` and `semantic` are already set
*synchronously*, before the HTTP response for `POST /api/captures` (or
`/clarify`) even returns -- only `retrieval_text`/`embedding` (and now
structuring) are backgrounded. Waiting for those to finish before
showing anything was never actually necessary for the Store page's own
purpose (confirming what got saved); it was inherited UI caution from
when the background step did more work. Per explicit instruction, all
"is this memory's background processing done yet" visualization now
lives on the Browse page only, which is also where the structurer's own
status belongs.

**What moved there instead:** attaching additional files to a
just-saved memory. Originally built on the Browse page's per-capture
row; moved to the Store page's own post-save result card after
feedback that it belongs where the user's attention already is right
after saving, not on a separate list page they'd have to go find the
capture on again. Same backend endpoint either way
(`POST /api/captures/{id}/attachments`, see "Multi-file posthoc
attachment" above) -- purely a UI placement decision.

---

**Decision:** "Disable follow-up questions" -- a new setting, enforced
server-side, that removes the clarification round-trip from the
storage cycle entirely. `POST /api/captures` gained an optional
`disable_followup_questions` form field; `create_capture()` and
`_unit_from_analysis()` (`app/ingestion/pipeline.py`) gained an
`allow_clarification` parameter. When `False`, NO unit resulting from
that capture can ever reach `status="awaiting_clarification"` --
suppressed uniformly for both a model-proposed question AND a genuine
explicit-metadata conflict (the explicit value still silently wins
either way; only the surfaced *question* is dropped, not the authority
rule). A settings-menu checkbox (`app/templates/base.html`,
persisted client-side in `localStorage` like the existing "obligatory"
setting) controls whether the Store page sends this field.

**Reason:** User-requested, specifically to reduce how long a
submission can stay "blocked" waiting on the user. The clarification
round-trip is architecturally different from every other latency this
project has optimized this phase: analyze_capture()'s own call takes a
few bounded seconds, but a pending clarification question waits on
*human* judgment, which has no bound at all -- a user might take
minutes, or never come back to answer it, leaving the unit stuck in
`awaiting_clarification` (durably saved, but not searchable) for as
long as that takes. Disabling the setting removes that specific
unbounded dependency: every unit now goes straight to `"processing"`
(the fast-pipeline's own bounded background embed step) the moment
`analyze_capture()` returns, regardless of what it proposed.

**What was deliberately NOT done:** the `analyze_capture()` prompt and
schema are unchanged -- the model is still asked to consider
clarification need (and still produces a `clarification_question`
field), it's just unconditionally discarded by the pipeline when the
setting is on. Stripping clarification out of the prompt/schema itself
to save a few output tokens was considered and rejected: the real,
stated cost being eliminated is the unbounded human-wait, not the
model's own (already bounded, already fast) response time, so a second
schema variant would have added real complexity for a saving that
isn't the one that actually matters here.

**Superseded:** the settings-menu checkbox for this (and the "obligatory"
one mentioned above) were later removed once the Store page stopped
blocking on clarification at all (see "Clarification moved to Browse")
-- the unbounded-human-wait this setting existed to let someone opt out
of no longer happens regardless, so there was nothing left for the
toggle to usefully control from the UI. The underlying API field
(`disable_followup_questions` on `POST /api/captures`) and pipeline
parameter (`allow_clarification`) are untouched -- still real,
server-enforced capabilities, just not exposed in the web UI any more.

---

## Benchmark: fast pipeline vs. structurer wall-clock time

Measured, not estimated: 5 short, single-fact memories, each timed
through `create_capture()` (the synchronous `analyze_capture` call),
`finish_unit_processing()` (the background embed), and
`finish_structuring()` (the structurer's own pass) back to back,
against the real `gemma4:12b`/`nomic-embed-text` models via Ollama,
otherwise idle (`-np 1`, no concurrent load).

| Memory | analyze (sync) | embed (bg) | structure (bg) |
|---|---|---|---|
| "I met Anna at ETH..." | 17.5s | 1.5s | 11.4s |
| "My gym locker combination is 42." | 8.1s | 0.1s | 10.8s |
| "I visited Rome in 2024..." | 7.3s | 0.1s | 12.9s |
| "I need to renew my passport..." | 8.1s | 0.1s | 10.2s |
| "Tom gave me a book called Dune..." | 14.1s | 0.1s | 11.4s |
| **average** | **11.0s** | **0.4s** | **11.3s** |

**What this means:**

- The number the USER actually waits on is the `analyze` column alone
  (~11s average, 7-18s range) -- `embed` and `structure` are both
  backgrounded, so neither adds to request latency.
- A unit becomes `"ready"` (searchable) after analyze+embed, averaging
  **~11.4s total** -- embed itself is negligible (~0.4s average; the
  1.5s outlier on the first row is consistent with a cold-start/first-
  call artifact on the embedding model, not a representative cost).
- **Structuring costs almost exactly as much wall-clock time as the
  entire fast pipeline combined** (11.3s vs. 11.4s, a 0.99x ratio) --
  for these short, simple, single-fact memories. This is a real,
  measured finding, not the "allowed to be LLM heavy" headroom
  translating into a dramatically larger number: both paths are
  fundamentally bounded by ONE generation call each for a memory this
  size, and the structurer's richer system prompt + the related-
  memory-candidate step don't add a second call, just somewhat more
  work within the one call it does make.
- **Practical implication:** every memory now costs roughly *double*
  the model compute-time it did before structuring existed (one fast-
  pipeline pass + one structuring pass), even though none of that
  second pass is on the user's critical path. Since Ollama serializes
  requests (`-np 1`), a burst of several new memories in quick
  succession means structuring passes queue up behind each other (and
  behind any concurrent recall query) -- invisible to any single user
  action, but a real aggregate load increase worth knowing about before
  assuming structuring is "free" just because it's backgrounded.
- Not covered by this benchmark: a long/dense, multi-unit capture (the
  kind that triggers chunking in `analyze_capture()` -- see "Chunk long
  captures before analysis" above). A rough, not-controlled data point
  from earlier testing: a ~20-fact capture needed 2 chunks/analyze
  calls and took several minutes end to end for analysis alone (before
  structuring was even involved) -- scaling, as expected, with how many
  separate generation calls a large capture needs, not a fixed cost.
- For contrast, image captioning (`finish_image_captioning`, a single
  `generate()` call with no JSON schema overhead) measured ~4.6s for a
  trivial test image -- meaningfully cheaper than either full
  structured-output path, consistent with schema-constrained JSON
  generation costing more than free-text generation for a comparable
  amount of actual content.

---

## Password login with per-device tokens

**Decision:** The deployment-readiness audit (see docs/DEPLOYMENT.md)
found this app had no authentication at all, which is fine for
`127.0.0.1`-only local use but not once anything makes the port
reachable from the internet. Given a choice between a plain HTTP Basic
Auth prompt and extending to multiple devices (a future mobile app,
a second browser) for the same single user, this project implemented a
hybrid: there is still exactly one password
(`Settings.membase_auth_password`, required, no default in
production -- `flake.nix`'s shellHook sets a `dev-only-password`
fallback for local development only), but `POST /api/auth/login`
exchanges it for a freshly-generated, per-device bearer token rather
than handing back one static shared secret. Each device that logs in
gets its own `AuthToken` row (`app/models.py`) -- independently
revocable later without having to change the password for every other
device.

Mechanics (`app/auth.py`, `app/api/auth.py`):
- `AuthToken` stores only `token_hash` (SHA-256 of the raw token) --
  the raw value is returned to the client exactly once, at login, and
  never persisted, logged, or shown again.
- `require_auth`, a FastAPI dependency, reads `Authorization: Bearer
  <token>`, hashes it, looks up a matching non-revoked row, bumps
  `last_used_at`, and raises 401 on any failure (missing header, no
  match, revoked) -- the same 401 and the same generic message for
  every failure mode, so a caller can't distinguish "wrong token" from
  "revoked token" from "no token at all."
- Applied at router-inclusion time in `app/main.py`
  (`app.include_router(captures.router, dependencies=[Depends(require_auth)])`,
  same for `units.router`/`recall.router`) rather than per-route, so
  every current and future route on those three routers is covered
  automatically. `auth.router` itself stays off this list -- `/login`
  is the one endpoint whose entire job is to hand out a token before
  the caller has one; `/logout`, `/sessions`, and `DELETE
  /sessions/{id}` still require a valid token via `Depends(require_auth)`
  on each of those route functions individually (they're on the same
  router as the unauthenticated `/login`, so router-level
  `dependencies=` would have caught `/login` too).
- HTML page routes (`app/web/routes.py`) are NOT gated server-side --
  they return the same static shell regardless of auth state, since
  they contain no real data. The actual gate is client-side:
  `app/static/app.js`'s `initAuth()` shows a full-page login overlay
  whenever `localStorage` has no token, and `fetchWithTimeout()`
  attaches `Authorization: Bearer <token>` to every request and pops
  the overlay back up on any 401 response (covers both "never logged
  in" and "this token just got revoked, possibly from another device's
  session list").

**Reason:** The user explicitly preferred "type a password from any
browser" over a VPN-based approach, but also stated an intent to
expand to mobile apps and multi-device access for the same single-user
account. Plain HTTP Basic Auth (one static credential pair, browser-
native prompt) satisfies the first preference alone but has no way to
tell devices apart or revoke just one of them -- the only "log out" is
clearing browser-cached credentials, which doesn't generalize to "kick
my old phone off" once a password is shared across every client. A
single static bearer token has the same problem in a different shape.
Per-device tokens solve both at once: one password to type anywhere,
but a device-level, revocable identity once logged in -- which is also
exactly the shape a future native mobile client needs (log in once,
store its own token, keep working even if a browser session elsewhere
gets revoked).

**Not done (acceptable gaps for this app's stated scope):** no token
expiry (a device's token is valid until explicitly revoked, not until
some TTL elapses -- acceptable for a single trusted user's own
devices, revisit if that assumption ever changes); no "this is my
current device" marker in the settings session list (revoking the
session this browser is actually using is handled gracefully -- the
next authenticated call 401s and the login overlay reappears -- but
the list itself doesn't visually flag which row that is).

---

## File-based password, with server-side rate limiting on /login

**Decision:** Two follow-up changes to the password-login design
above, both requested after the initial implementation was live-
verified:

1. The one shared password moved from an env var
   (`Settings.membase_auth_password`) to a file read fresh on every
   login attempt (`Settings.membase_auth_password_file`, default
   `.data/auth_password`; see `app/auth.py` `verify_password()`). An
   env var is only set once, at process start -- changing it means
   editing whatever sets the environment and restarting the process.
   A file can be edited in place (e.g. over SSH: `vim
   .data/auth_password`) and takes effect on the very next login
   attempt, no restart. `get_settings()` is still `@lru_cache`'d, but
   that only caches the *path*; the file's contents are read fresh
   every time. A missing/unreadable file, or one that's empty, makes
   every login attempt fail closed rather than falling back to some
   default or treating an empty password as "no password required."
   `nix develop`'s shellHook seeds a dev-only default file under
   `.data/` (gitignored) so local dev keeps working out of the box;
   that file is not meant to back a real deployment.
2. `POST /api/auth/login` previously had no rate limiting at all --
   flagged as an accepted gap above, mitigated only by
   `scripts/run-prod.sh` binding `127.0.0.1`-only behind a reverse
   proxy. Two independent, in-memory limiters were added
   (`app/rate_limit.py`), checked in this order so a caller can't use
   response timing to distinguish "locked out" from "wrong password":
   - **Global flood guard:** at most 5 requests to `/login` per
     second, in total, across every caller combined (keyed by nothing
     -- it's one shared counter). Catches a burst/flood regardless of
     source; not meant to be a serious brute-force deterrent on its
     own.
   - **Per-IP brute-force guard:** at most 10 *failed* attempts per
     15 minutes, per source IP (`request.client.host`), then locked
     out (`429`, with a `Retry-After` header) until the oldest failure
     in that window ages out. A successful login immediately clears
     that IP's failure count. Only failures count here (unlike the
     global guard, which counts every request) -- a legitimate device
     logging in repeatedly should never trip this.
   Both are in-memory, per-process state -- no Redis, no DB table.
   Safe because `scripts/run-prod.sh` always runs this app with
   exactly one worker (it already has other process-local singletons
   that assume this); a restart just resets everyone's counters, which
   is a loosening, not a security hole.

**Reason:** User-requested, after reviewing the original
no-rate-limiting gap and the env-var password source live. The file-
based password directly serves the stated deployment shape --
single account, edited by whoever can SSH into the box, no redeploy
needed to rotate it. The two-tier rate limiter was an explicit design
ask (first drafted at 5 failures/15 min with no global guard, then
widened to 10 failures/15 min plus a separate 5 req/s global cap) --
splitting "don't let anyone flood this endpoint" from "don't let one
attacker brute-force the password" means the global cap can stay tight
without making life harder for one legitimate device that fat-fingers
the password a few times in a row.

**Not done:** the per-IP counter has no persistence or distributed
component -- a reverse proxy that doesn't forward the real client IP
(or that fronts many users behind one NAT'd IP) would make the per-IP
guard either useless or overly broad; out of scope for a single-user,
`127.0.0.1`-bound deployment. No alerting/logging on repeated
lockouts beyond the 429 response itself.

---

## Full-codebase audit and fix pass

**Decision:** User requested an in-depth audit of the entire codebase
for bugs, engineering quality, and structural cleanliness, then asked
to fix everything relevant and reach a state safe to deploy. Five
parallel audits covered `app/ingestion/`, `app/retrieval/`, the API/
models/LLM layer, the frontend (`app.js`/`app.css`/`base.html`), and
`tests/`/`scripts/`/migrations; a sixth covered `eval/` and
`scripts/reembed.py`. 27 findings came back from the first five, plus
one more (the eval database-isolation guard, below) from the sixth.
Nearly all were fixed in this same pass; what follows groups them by
theme. Two were deliberately deferred (see "Not done").

**Fixed -- correctness bugs with a real, live failure mode:**
- The login overlay (`app/static/app.css`) could never actually be
  hidden: `.hidden { display: none }` and `.login-overlay { display:
  flex }` had identical specificity, and source order favored
  `.login-overlay`, so every page load left the full-viewport overlay
  on screen regardless of auth state. This had already shipped and
  was never caught because the earlier verification of that feature
  was curl + `node --check` only (no real browser available that
  session). Fixed with a two-class `.login-overlay.hidden` rule, which
  reliably wins the cascade regardless of source order.
- `/api/auth/login`'s rate limiter (`app/rate_limit.py`) had a real
  TOCTOU race: `login()` is a plain `def`, so FastAPI genuinely
  thread-pools concurrent calls to it, and the check-then-record
  sequence for both the global and per-IP counters was unsynchronized.
  Fixed with `rate_limit.LOGIN_LOCK`, a `threading.Lock` held across
  the whole check-verify-record sequence in `login()` -- serializing
  login attempts, an acceptable cost given this app's single-user,
  single-worker scope.
- The login overlay's success handler never re-ran whatever page-level
  data fetch had already 401'd before login, leaving stale error text
  on screen. Fixed by reloading the page on successful login instead
  of just hiding the overlay.
- `discriminate.py`'s cue-matching substring fallback (`_matches_cue`)
  had no word-boundary awareness, so a short cue like "Ann" could match
  inside unrelated words ("annual", "banner"). Fixed with a
  `\b`-bounded regex.
- `sibling_context()` (`app/retrieval/pipeline.py`) queried a capture's
  sibling units with no `ORDER BY` before truncating to
  `MAX_CONTEXT_PER_PRIMARY`, making which siblings survive the
  truncation non-deterministic. Fixed with `.order_by(MemoryUnit.created_at)`.
- `_CANDIDATE_TERM_RE` (`app/ingestion/pipeline.py`) only matched ASCII
  letters, so related-context lookup silently found nothing for
  capitalized names with accented characters ("Zürich", "Müller",
  "François"). Fixed with a Unicode-aware character class
  (`[^\W\d_]`) plus a post-match uppercase-start check in Python
  (stdlib `re` has no portable Unicode-uppercase character class).
- `finish_image_captioning`, `finish_attachment_chunking`, and
  `_finalize_structuring` had no catch-all for unexpected (non-
  `LLMError`) exceptions, unlike `_finalize_unit`'s existing pattern --
  e.g. a missing file on disk would crash the background task silently
  with no record anywhere, and for structuring specifically, leave the
  unit's `structure_token` lease stuck forever. All three now log and,
  where there's a lease to release (`_finalize_structuring`), mark it
  `"failed"`, mirroring `_finalize_unit`.
- Two test assertions couldn't actually fail regardless of the real
  outcome: `test_backfill.py`'s `"stale / " in result.stdout` fallback
  (satisfied by `"0 stale / 1 total"` too) and
  `test_recall_session.py`'s `... is not None or True`. Both now assert
  on the exact expected string/behavior.

**Fixed -- a real data-safety gap in developer tooling (`eval/`):**
- None of the six eval/benchmark entry points (`eval/run_eval.py`,
  `run_session_eval.py`, `threshold_sweep.py`,
  `benchmark_recall_latency.py`, `run_interactive_benchmark.py`,
  `trace_interactive_benchmark.py`) verified they were actually
  connected to the isolated `membase_eval` database before running
  `TRUNCATE captures, memory_units, ...` -- that isolation was enforced
  *only* by the wrapping `scripts/*.sh` exporting `PGDATABASE`/
  `DATABASE_URL` first. Running any of these directly (e.g. `python3
  eval/run_eval.py` from inside `nix develop`, by habit or an IDE "Run"
  button) would silently truncate and overwrite real stored captures/
  memory units, with no recovery path given `.data/` has no separate
  backup anywhere in this project. Fixed with
  `eval/_helpers.py`'s `assert_eval_database(db)`, which checks `SELECT
  current_database()` on the live connection itself (not an env var
  that could diverge from it) and refuses to run otherwise. Called at
  the top of every entry point's `main()`, right after the DB session
  opens.

**Fixed -- blocking sync DB calls on the event loop:**
`require_auth` (`app/auth.py`, run on every authenticated request),
`discard_unit` (`app/ingestion/pipeline.py`), and the pure-lookup
routes with no real `await` inside (`units.get`, `captures.list_captures`/
`get`/`delete`, `recall.get_session`/`cancel`) were all `async def` with
synchronous SQLAlchemy calls and nothing to actually await -- meaning
FastAPI ran their blocking DB work directly on the event-loop thread
instead of off-loading it to the thread pool, stalling every other
concurrent request (including a second device's request -- the exact
multi-device use case the auth system was built for) for the duration.
Converted all of them to plain `def`, which FastAPI thread-pools
correctly, matching the pattern `app/api/auth.py`'s own routes already
used. `GET /api/captures` was also missing eager-loading on its
`units`/`attachments` relationships (N+1: ~1001 queries for 500
captures instead of 3) and `_format_chunk`'s `chunk.attachment.capture.units`
traversal had the same problem for up to `TOP_K_FOR_ANSWER` chunks per
answer -- both fixed with `selectinload()`.

**Fixed -- smaller cleanliness items:** removed the never-called
duplicate upload-filename-filtering comprehension in `captures.py`
(now a shared `_read_uploads()`); removed dead `ErrorResponse` schema;
added a FastAPI shutdown hook that closes `OllamaProvider`'s
`httpx.AsyncClient` pool (`aclose()` existed but nothing called it);
fixed a stale module docstring in `agent.py` that still claimed "3
search rounds, up to 2 reformulated queries" after the streamlining
pass had already changed those constants to 2/1; batched
`_refresh_ready_units_with_attachment_excerpt`'s N sequential
`llm.embed()` calls (one per ready sibling) into one call across all
siblings, consistent with the rest of this codebase's LLM-call-count
discipline; fixed `postJSON` (`app.js`) wrapping a plain-string error
`detail` in literal quote characters; made `escapeHtml` also escape
quote characters, closing a latent attribute-injection trap for any
future caller (no current call site was exploitable); made
`initAuth()`'s element-presence check cover all six elements it
touches, not just two, consistent with every other element access in
the file degrading gracefully instead of throwing; fixed
`scripts/run-eval.sh`/`run-session-eval.sh`/`sweep-threshold.sh`/
`benchmark-recall-latency.sh`/`run-interactive-benchmark.sh` resolving
`backend/`/`eval/` relative to cwd instead of `$MEMBASE_ROOT` (the
exact footgun CLAUDE.md #7 documents, already fixed in
`run-dev.sh`/`run-prod.sh` but not propagated to these); removed 138
now-redundant `@pytest.mark.asyncio` decorators across the test suite
(`pyproject.toml` already sets `asyncio_mode = "auto"`, making the
marker a no-op) and the resulting 13 unused `import pytest` lines;
fixed `scripts/dev-db.sh` interpolating `$PGDATABASE` unescaped into a
SQL string (now doubles embedded single quotes, standard SQL-literal
escaping -- not exploitable today since the value comes from the
flake's own trusted env, but worth doing right; psql's `:'var'`
substitution looked like the more obvious fix but does not apply to
`-c` command strings, confirmed by testing it directly).

**Not done, deliberately deferred:**
- The remaining async routes that genuinely `await` real LLM work
  (`create`, `clarify`, `retry`, `correct`, etc.) still interleave
  blocking sync SQLAlchemy calls with those awaits. Fully correct
  async behavior would mean either migrating to SQLAlchemy's async
  engine or wrapping every blocking DB call in
  `run_in_threadpool`/`anyio.to_thread` -- both substantial, risky
  refactors touching most of `app/ingestion/` and `app/retrieval/`.
  Given this app's single-worker, single-user, localhost-Postgres
  scope (sub-millisecond query latency in practice), the cost/risk of
  that refactor isn't justified by this pass's "fix and ship safely"
  goal; the highest-value instance (`require_auth`, on every request)
  and every trivially-fixable pure-sync route were already converted
  above. Revisit if this app ever needs to serve more than one
  concurrent user meaningfully.
- Two low-severity duplication findings were reviewed and left as-is
  rather than risk introducing a subtle bug under time pressure:
  the ~6-line "embed, set fields, log-and-skip on LLMError" pattern
  repeated across 4 call sites in `app/ingestion/pipeline.py`, and
  `structurer.py` hand-redefining the same semantic-fields JSON schema
  `analysis.py` already defines instead of deriving from it. Both are
  real but cosmetic; revisit opportunistically next time either file
  is touched for an unrelated reason.

---

## Split deployment: backend local behind nginx + Cloudflare Tunnel, frontend on GitHub Pages

**Decision:** The deployment shape changed from "one process serving both
the API and the HTML/JS/CSS on one origin" to two independently deployed
pieces: the backend keeps running locally (as before -- `scripts/run-prod.sh`,
still bound `127.0.0.1`-only), and the frontend is published to GitHub
Pages as static files, a different origin. This section covers the pieces
that changed to make that split work: how the backend becomes reachable at
all despite CGNAT, and the auth/reverse-proxy architecture in front of it.
See "Publishing the frontend to GitHub Pages" below for the actual
static-file extraction/publish workflow.

### Cloudflare Tunnel for CGNAT-safe public reachability

**Reason:** The machine running the backend is behind CGNAT (Carrier-Grade
NAT) -- it has no unique public IP of its own, so conventional router
port-forwarding cannot work: there is no "public IP" under local control to
open a hole in, regardless of any router configuration. The machine is
already on a Tailscale tailnet, but Tailscale Funnel (the feature that
would expose a tailnet service to the public internet without port
forwarding) only serves on a Tailscale-assigned `*.ts.net` hostname, not an
arbitrary custom domain -- and a real custom domain (`vxliu.com`, already
owned) was a hard requirement.

**Mechanics:** `cloudflared`, a daemon running on the backend machine,
makes an *outbound* connection to Cloudflare's edge network -- no inbound
port needed on this machine at all, so CGNAT is irrelevant. Cloudflare's
edge terminates TLS for `membase-api.vxliu.com` (using their own cert,
auto-issued and auto-renewed -- no certbot/ACME management on this end)
and forwards traffic through that outbound tunnel to whatever
`cloudflared`'s ingress config points it at locally -- here, nginx on
`127.0.0.1:8080`, not directly to the FastAPI backend (see below for why).
Prerequisite: `vxliu.com`'s authoritative DNS must be hosted on Cloudflare
(`cloudflared tunnel route dns` provisions the actual DNS record pointing
the hostname at the tunnel). Config: `deploy/cloudflared-config.yml`.

**The hostname is `membase-api.vxliu.com`, a sibling of `membase.vxliu.com`
(both one level under the zone apex), not a child of it** -- went through
two iterations before landing here, each caught by actually testing the
live connection rather than assuming the config was right:

1. First attempt used the bare `membase.vxliu.com` for the backend too --
   discovered as a real conflict immediately: that hostname was already
   claimed as GitHub Pages' custom domain for the frontend (a `CNAME` file
   at the repo root, added via GitHub's own UI before this was noticed),
   and a single hostname's DNS can't point at both GitHub Pages'
   infrastructure and a Cloudflare Tunnel at the same time.
2. Second attempt used `api.membase.vxliu.com` (the backend as a child of
   the frontend's hostname) -- DNS resolved fine (`cloudflared tunnel route
   dns` succeeded), but every connection failed with a TLS handshake error.
   Confirmed by inspecting the actual certificate Cloudflare's edge serves
   for the zone: free Universal SSL only covers `vxliu.com` and
   `*.vxliu.com` -- one wildcard level, nothing deeper. `api.membase.vxliu.com`
   is two labels below the apex (`api.` *and* `membase.`), so no cert on
   Cloudflare's edge ever matched that SNI.
3. `membase-api.vxliu.com` -- one label below the apex, a sibling of
   `membase.vxliu.com` rather than a child -- is covered by the same
   `*.vxliu.com` wildcard the frontend already uses, and resolves the
   original GitHub Pages conflict identically to the `api.` attempt (still
   a distinct hostname from the frontend's). No DNS-zone changes needed
   beyond the one new record `cloudflared tunnel route dns` creates.

**Not done:** no failover/redundancy if the single `cloudflared` process or
its one tunnel connection drops -- acceptable for a single-user app; revisit
if uptime ever becomes a real requirement.

### Publishing the frontend to GitHub Pages

**Decision:** `scripts/build_frontend.py` pre-renders the three Jinja
page templates (`backend/app/templates/{create,recall,captures}.html`)
with plain Jinja2 and an empty context into static `dist/index.html`,
`dist/recall/index.html`, `dist/captures/index.html`, copies
`backend/app/static/` verbatim, and adds a `.nojekyll` marker plus the
repo-root `CNAME`. `.github/workflows/pages.yml` runs that script on
every push to `main` touching the frontend's own inputs and publishes
`dist/` via `actions/upload-pages-artifact` + `actions/deploy-pages` --
nothing built is ever committed to `main`.

**Reason:** The templates rendered with an empty context at every call
site, back when `app/web/routes.py` still existed and served them
itself -- verified by grep at the time, not assumed: no `{{ }}`
interpolation anywhere outside `{% block %}`/`{% extends %}`. That made
a one-time, build-time render exactly as correct as the backend's own
live Jinja2Templates rendering was, with no server. `app/web/routes.py`
itself was removed shortly after (see "API-only backend: stop serving
the frontend's own HTML/static files") once nothing depended on the
backend serving these pages live any more -- the templates under
`backend/app/templates/` remain, now purely as this build script's
source material, the single place the HTML structure is defined.

**Not done:** no visual/screenshot regression check between what the
backend would render and what the static build produces -- acceptable
given the render is deterministic and the templates have zero dynamic
context to begin with.

**Layering, not replacing, nginx with Cloudflare's edge features:**
Cloudflare can do several of nginx's jobs at the edge too (security
headers, HSTS, rate limiting), raising the question of whether nginx
is redundant now. Decision: keep nginx as the one local chokepoint,
add the Cloudflare-side equivalents as an *additional* layer in front
of it, not a replacement -- see docs/DEPLOYMENT.md "Layering
Cloudflare's edge features on top of nginx" for the concrete dashboard
steps. Reason: the backend machine is also reachable over Tailscale;
an edge-only approach (Cloudflare Access instead of nginx's Basic
Auth, edge-only headers/rate-limiting with no local copy) would only
protect requests that go through the public `membase-api.vxliu.com`
hostname, leaving a direct Tailscale hit to the machine completely
unauthenticated, since the app itself has no auth code of its own.
nginx sits in front of the app regardless of which network path a
request takes to reach the machine, so it stays authoritative; the
Cloudflare-edge copies are redundant on purpose, catching abuse before
it ever consumes tunnel bandwidth, not instead of nginx's own.

### Retire app-level auth for nginx Basic Auth

**Decision:** The password-login + per-device bearer-token system
(`app/auth.py`, `app/api/auth.py`, `app/rate_limit.py`, the `AuthToken`
table, the login overlay's token/session-list UI) was removed entirely.
Authentication is now HTTP Basic Auth, enforced by nginx in front of the
backend (`deploy/nginx-membase.conf`'s `auth_basic`/`auth_basic_user_file`),
not by the application. The frontend still shows a custom login-style
overlay (`app/static/app.js`'s `initAuth()`), but it no longer talks to any
`/api/auth/*` endpoint -- there isn't one anymore. It just collects a
username + password once, stores them in `localStorage`, and attaches
`Authorization: Basic <base64(user:pass)>` to every API request itself
(not via the browser's native Basic Auth prompt/cache, which would mean
a jarring OS-native dialog instead of this app's own overlay). nginx is
the only thing that ever actually validates the credentials. (The
backend's own address stopped being collected here too, shortly after --
see "Hardcode the backend URL instead of asking for it at login".)

**Reason:** User-requested, after an explicit discussion of the tradeoff:
proxy-level Basic Auth is the conventional baseline for a single-user
self-hosted app (most such apps never build their own session/token layer
at all), and the custom app-level system -- built specifically for
per-device revocation ahead of an anticipated mobile client -- was judged
not worth keeping once a reverse proxy was being introduced for other
reasons (TLS termination via Cloudflare, security headers, rate limiting).
Consolidating "is this request allowed at all" into one place (nginx) is
simpler than splitting it across two layers.

**Trade-off, accepted knowingly:** there is now exactly one shared
credential pair, with no per-device identity and no way to revoke access
for one specific device without changing the password for every device at
once. The old system's whole reason for existing -- "kick my phone off
without re-typing the password everywhere else" -- is gone. Acceptable
given this app's stated single-user scope; revisit (likely by reintroducing
an app-level or proxy-level per-device mechanism, e.g. Cloudflare Access
with per-device service tokens) if multi-device revocation becomes a real
need again.

**What moved where, concretely:**
- **nginx** (`deploy/nginx-membase.conf`): HTTP Basic Auth, security
  response headers (`X-Content-Type-Options`, `X-Frame-Options`,
  `Referrer-Policy`, `Strict-Transport-Security`), a general per-IP rate
  limit across every route (`limit_req`, replacing the retired app-level
  `/login`-specific limiter), and the request body-size cap
  (`client_max_body_size`, matching `membase_max_upload_mb`). Real client
  IP recovered from `CF-Connecting-IP` (set by `cloudflared`, trusted here
  because that connection is always local, never attacker-reachable
  directly).
- **The FastAPI app** (`app/main.py`): CORS only (`CORSMiddleware`,
  allow-listing `app/config.py`'s `frontend_origin` -- the published
  GitHub Pages origin). Deliberately NOT moved to nginx: CORS is naturally
  expressed as middleware already built into the framework, and keeping it
  here means it stays in sync with whatever config says the frontend's
  origin is, without a second place to update. `allow_credentials` is
  correctly `False` -- the frontend's own JS attaches the Basic Auth header
  explicitly; nothing here relies on browser-managed cookie credentials.
- **nginx's `auth_basic` and CORS preflight**: a plain `if ($request_method
  = OPTIONS) { set $membase_auth "off"; }` disables Basic Auth just for
  preflight requests, which never carry credentials by design -- without
  this, every preflight would 401 before the real, credentialed request
  behind it ever got a chance, breaking all cross-origin calls from the
  GitHub Pages frontend. Standard, documented nginx idiom for this exact
  conflict (`auth_basic` accepts a variable; the literal value `"off"`
  disables it).

**Not done:** no per-device anything (see trade-off above); nginx's rate
limit is IP-based and request-count-based only, not aware of what counts
as a "failed login attempt" the way the retired app-level limiter was
(nginx has no visibility into response semantics without more complex
config); the htpasswd file's credentials are managed entirely manually
(generate with `openssl passwd -apr1`, no rotation tooling).

---

## Two live bugs found deploying the real frontend (DOM ordering, CORS on proxy-level rejections)

Both only surfaced once the actual deployed frontend was exercised in a
real browser for the first time -- everything up to this point had only
been checked via `curl`/`node --check` (explicitly noted as a known gap
in earlier entries), which can't catch either of these.

### `#login-overlay` positioned after the script that needs it

**The bug:** `app/templates/base.html` had `<div id="login-overlay">`
positioned *after* `<main>{% block content %}{% endblock %}</main>`.
Every page's content block includes `<script src="/static/app.js">`
as a classic (non-deferred) script, which executes synchronously the
moment the HTML parser reaches it -- before the parser has read
`#login-overlay`, further down, into the DOM at all. `initAuth()` runs
unconditionally at script load time and does
`document.getElementById("login-overlay")` first; finding `null` (by
design, matching every other element lookup in that file, each of
which bails out gracefully rather than throwing on a missing element),
it returned immediately. No error, no overlay, no event listeners ever
attached -- indistinguishable from "there's no authentication at all"
from the outside, which is exactly how it was reported.

**Fix:** moved `#login-overlay` to appear directly after `</header>`,
before `<main>` -- zero JS changes, the overlay's own `position: fixed`
styling means its DOM position never affected how it looks, only
whether `initAuth()` could find it in time.

### Move CORS to nginx: an app-level `CORSMiddleware` can't cover a proxy-level rejection

**The bug:** CORS lived in the FastAPI app's `CORSMiddleware`
(`app/main.py`), allow-listing `Settings.frontend_origin`. That only
covers responses the app itself produces. nginx's own `auth_basic` can
reject a request (401) *before it ever reaches the app* -- and a
401 nginx generates entirely on its own carries no CORS headers at
all, since nothing added them. A browser blocks such a response from
ever being read by the page's JS (a CORS failure, not a readable 401),
so `fetchWithTimeout`'s `resp.status === 401` branch -- the one that
clears stored credentials and re-shows the login overlay -- never ran;
the failure surfaced as an opaque, generic error instead ("failed to
save"). The same gap applies to the rate limiter's 429.

**Fix:** moved CORS entirely into nginx (`deploy/nginx-membase.conf`):
`Access-Control-Allow-*` headers added with `always` at the `location /`
level, so they're present on every response that location can produce
-- auth_basic's 401, the rate limiter's 429, and proxied responses from
the app alike, not just the success path. The preflight `OPTIONS`
request is now answered by nginx directly (`return 204`, after the CORS
headers are already set) instead of being forwarded to the app with
Basic Auth conditionally disabled for it -- simpler, and one fewer
thing that depends on the app behaving correctly for an unauthenticated
request type. `Settings.frontend_origin` and the app's `CORSMiddleware`
were removed entirely; nginx is now the single source of truth for
CORS, with no risk of the two layers emitting conflicting or duplicate
`Access-Control-Allow-Origin` values (browsers reject a response with
more than one).

**Reason this wasn't caught earlier:** every prior CORS check (including
in this same deployment session) tested the *success* path -- a
preflight from the right origin, a request with correct credentials.
Nothing exercised what a cross-origin client sees on a rejection nginx
generates itself, which is precisely the case that matters most for the
frontend's intended retry-on-401 UX.

**Not done:** no automated test reproduces either bug (both are
browser-DOM-timing and cross-origin-error-response behaviors that the
existing `pytest` suite, which never runs a real browser or a real
nginx, structurally can't exercise) -- caught by manually loading the
deployed site in an actual browser instead.

---

## Hardcode the backend URL instead of asking for it at login

**Decision:** `app/static/app.js` now has a fixed `BACKEND_URL` constant
(`https://membase-api.vxliu.com`) instead of reading it from a field the
login overlay collected and stored in `localStorage`. The overlay now
only asks for username + password.

**Reason:** User-requested, once the backend's hostname had actually
settled (after two iterations to get past the GitHub Pages hostname
conflict and the Universal SSL wildcard-depth limit -- see "Cloudflare
Tunnel for CGNAT-safe public reachability"). This frontend is only ever
published for this one backend; asking for its address at every login
was friction with nothing to gain once that address stopped changing.
Matches the "fixed at deploy time" option considered (and deliberately
not chosen) back when the backend's reachability was still unsettled --
revisit if that ever becomes true again.

**Not done:** no build-time templating/env-var injection for this
constant -- it's a literal string in the committed source, so pointing
a fork of this frontend at a different backend means editing
`app/static/app.js` directly, not just setting a build variable.
Acceptable given there's exactly one deployment of this frontend.

## API-only backend: stop serving the frontend's own HTML/static files

**Decision:** Removed `app/web/routes.py` (the four page routes --
`/`, `/recall`, `/captures`, `/docs`) and the `/static` `StaticFiles`
mount from `app/main.py` entirely. The backend now serves `/api/*`
only; everything else 404s.

**Reason:** User-reported: visiting the backend's own domain
(`membase-api.vxliu.com`) rendered the same HTML pages the frontend
does, because nginx proxied every path -- not just `/api/*` -- straight
to this app, which still had its own, now-entirely-redundant copy of
the page routes left over from before the GitHub Pages split. An API
domain rendering a website is confusing and not what "it should only
be an API" means. `backend/app/templates/` and `backend/app/static/`
still exist in the repo -- `scripts/build_frontend.py` reads them
directly off disk as source files for the GitHub Pages build (see
"Publishing the frontend to GitHub Pages") -- but this running process
never serves them over HTTP any more.

**Not done:** nginx's own config wasn't narrowed to `location /api/`
specifically -- it still proxies every path to the backend, which now
just means those paths 404 instead of silently serving a website. Left
as-is since a 404 is already the correct, unconfusing behavior for an
API-only domain; narrowing the nginx location would be pure belt-and-
suspenders with no behavior change.

---

## Clarification moved to Browse

**Decision:** The Store page no longer shows any clarification UI at
all. `runCapture()` (`app/static/app.js`) goes straight from "Saving..."
to a result card the moment `POST /api/captures` returns -- if any
resulting unit has `pending_clarification` set, the result card just
notes that and points to the Browse tab; it never blocks, never shows
the question, never offers to answer or discard inline. Resolving a
pending clarification (`POST /api/memory-units/{id}/clarify` or
`.../discard`) now happens exclusively on the Browse tab, via the
per-unit form added there (see "Full-codebase audit and fix pass" --
this UI already existed before this decision; what changed is Store no
longer has a competing copy of it). The Browse tab's capture list also
now sorts captures with at least one pending unit first (a stable sort,
so each group keeps the backend's own newest-first order), and gained
a single "Skip all remaining follow-up questions" button that discards
every currently-pending unit across the whole list in one action
(sequential `POST .../discard` calls, one per unit, confirmed once
up front -- not parallel, since these share the same backend as
everything else and there's no reason to race them).

**Reason:** User-requested, consolidating clarification into one place
instead of two. The Store page's in-flight clarification flow and the
Browse tab's later-resolution flow had become two independent
implementations of the same thing (same endpoints, materially the same
UI, different code) -- confusing to maintain, and splitting "where do I
answer a follow-up question" across two pages depending on *when* you
happen to do it. Browse is also simply the better fit: it's the page
that already visualizes a unit's status and background-processing
progress, so a pending clarification sitting there alongside is one
more thing to check in the place that's already "checking on things",
not an interruption on the page whose only job is writing something
down quickly.

**What this removes:** both settings-menu toggles ("obligatory
questions", "disable follow-up questions") -- see the two superseded
notes on their original decisions above for why each stopped making
sense once nothing on Store blocks any more. `buildCaptureCard()`'s
clarification markup, `runCapture()`'s `pendingQueue`/
`askNextClarification()`/clarification-form-handling, and
`isObligatory()`/`isFollowupDisabled()` and their `localStorage` keys
are all gone from `app.js`.

**Not done:** no visual distinction in the Browse list beyond sort
order for *why* something is sorted first (no separate heading/divider
between "pending" and "everything else") -- the sort alone was judged
sufficient; revisit if the list ever gets long enough that it isn't.
