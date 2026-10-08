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
- `membase_max_chunks_per_attachment` (default 300): bounds how many
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
`membase_max_upload_mb` (25MB) already bounds worst-case payload size
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
