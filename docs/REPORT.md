# Engineering report: the artificial memory engine

This supersedes the previous report (restore the multi-call pipeline / instant
save / image structured data / attachment access), which is still accurate
about the ingestion pipeline and is summarised where relevant below.

Scope: turn a working augmented-memory MVP into a system that can retrieve a
memory the query could never have matched directly — by building an explicit
associative structure over what ingestion already extracts, making retrieval
adaptive over that structure, grouping memories into episodes, exposing the
evidence behind every result, and measuring whether any of it actually helps.

Verified against the real local model throughout. Fast suite: **392 tests
passing** (up from 172), plus 7 real-`gemma4:12b` integration tests. All seven
UI sections verified in a real browser against a live backend with real seeded
data.

---

## 1. What was built

### Associative memory graph

`MemoryUnit.related_unit_ids` was explicitly "NOT a knowledge graph" — a flat
advisory id list with no types, weights, provenance, or traversal. It is
unchanged, and now additionally projected into a real one.

Four typed tables: `memory_entities` (the canonical things memories are about),
`unit_entity_links` (which memory mentions what, in which semantic field, and
whether the user said it), `entity_relations` (subject-predicate-object triples,
each asserted by one specific memory), `unit_associations` (weighted edges
between memories). Four tables rather than one polymorphic edge table so every
edge has real foreign keys and real indexes.

Entities are the hub nodes, and they are the whole mechanism: two memories
sharing no wording at all are one hop apart if they name the same lake.

Three properties were treated as requirements rather than nice-to-haves:

- **Zero additional LLM calls.** Every edge derives from fields ingestion
  already produces. Entity resolution is a normalisation function, not a model
  call. Ingestion already costs ~6 generation calls per capture on an 11 GB GPU
  that fits exactly one generation model; spending more was never available.
- **A projection, not primary data.** `project_unit()` is idempotent
  (delete-then-rebuild), so it runs from four different pipeline commit points
  with no change tracking, and a bug in it is fixable with
  `scripts/rebuild_graph.py` rather than a re-ingest. At ~20-25s of model time
  per memory, that distinction is the difference between a fixable bug and a
  permanent one.
- **`origin` is a column, not a heuristic**: `explicit` (the user said it) >
  `extracted` (a model read it out of their words) > `inferred` (the system
  computed it). This is the storage side's existing authority hierarchy carried
  into the graph.

The API (`/api/graph/*`) is read-only. The brief says not to turn this into a
manually maintained knowledge graph; having no write endpoint makes that
structural rather than a matter of discipline.

### Adaptive associative recall

`app/retrieval/associative.py` runs as a *stage* after the existing
`automatic_search`, which keeps its exact contract. It takes the ranked seeds,
walks the graph, collects structured evidence, and re-ranks on relevance +
contextual consistency + evidence quality.

Expansion is adaptive: `should_expand` declines to walk the graph when direct
search already produced a dominant, evidentially-complete winner — the same
three-signal test as the existing sufficiency short-circuit. A clear query
costs exactly what it did before this feature existed, and the trail records
that exploration was skipped deliberately.

**An inferred lead can never become an answer.** Three independent mechanisms,
deliberately redundant: confidence caps at `weak`; `rerank` clamps the score
below the moderate-direct-match threshold so no accumulated evidence lets it
outrank a real match; and the answer prompt labels the block "Associated
memory" with an explicit instruction not to answer from it alone.

Clarifying questions are chosen by **expected information gain**, discounted by
how answerable each attribute actually is (see §4 for why the second half is
not a fudge factor).

### Episodic memory

Episodes are derived deterministically — no LLM call, including for titles —
by three rules in descending confidence: same capture; compatible date *and*
shared anchor entity; two shared anchors when undated.

Requiring both time and entity overlap is what stops an episode being either a
calendar or a bag of every memory mentioning a recurring person. `topic`
entities are excluded from anchors entirely: "hiking" is shared by every hike.

Uncertainty is represented rather than smoothed away. Dates are nullable and
carry a precision (`day`/`month`/`season`/`year`/`unknown`) and a
`boundary_confidence` (`asserted`/`inferred`/`uncertain`). An episode whose
members carry no resolvable date has NULL dates and says so — it is never
back-filled from `created_at`, because "I don't know when this was" and "this
happened the day it was typed" are different claims and a user cannot tell them
apart afterwards.

### Multimodal

`semantic.visual_concepts` is now populated (it was read by the graph but
written by nothing), through the existing retrieval-aids enrichment call — no
new LLM call. An image of a lake and mountains yields `mountain range`,
`body of water`, `snow-capped peaks`, which become `object` entities and
therefore associative paths.

**Audio works.** Transcription runs through `gemma4:12b` via the existing
`images` field on Ollama's `/api/chat`; no second model, which the 11 GB VRAM
budget would not have allowed. Four empirical findings are recorded in
`docs/DECISIONS.md`; the one worth repeating here is that a separate `audio`
field on the message is silently discarded — HTTP 200, zero media tokens, a
confident hallucinated transcript — so the obvious-looking "fix" to the working
path would be a quiet correctness disaster.

### Inspectable results

Both the ranking and the explanation are computed from one list of typed
evidence items, so there is no display-only explanation that can drift from
what actually drove the ranking. Each item carries one of four provenance
levels: `original` (the user's words) / `extracted` / `inferred` / `hypothesis`
(a search guess — displayed and auditable, excluded from scoring).

Confidence is a **category**, never a percentage: `strong` / `moderate` /
`weak` / `insufficient`, defined by an explicit rule table. Nothing in this
system produces a calibrated probability, so a number would be fabricated
precision — the same class of error as inventing a date.

`RecallSession` gained `strategy_trail` (one entry per round: strategies run,
direct vs. expanded counts, what the user's cues ruled out, the question asked
with its approximate information gain, the budget in force), `final_evidence`,
`final_confidence`, and `abstention_reason`.

### The interface

Seven sections (Overview, Recall, Explorer, Timeline, Library, Evaluation,
Store), rebuilt from three pages sharing one 755-line IIFE into a shared core
plus per-section scripts and stylesheets, on a real design system.

The Explorer's force-directed graph is written from scratch against `<canvas>`
— velocity-Verlet with repulsion, springs, centring gravity and cooling. No
library: the repo is dependency-free and the whole frontend build is
`pip install jinja2`, so vendoring d3 to draw a few hundred nodes is not a
trade worth making. It stops its animation loop once settled (measured: 264
steps / 122 frames / 1999 ms for 68 nodes and 179 edges, `running=false`
afterwards) and honours `prefers-reduced-motion`.

---

## 2. Measured results

`scripts/run-strategy-benchmark.sh` compares four strategies over the **same**
44 synthetic memories (51 ready units) and 40 queries across 8 categories.
Full definitions and caveats in `docs/EVALUATION.md`.

| strategy | recall@1 | recall@5 | correctness | false retrieval ↓ | abstention ↓ | questions | LLM calls | wall s |
|---|---|---|---|---|---|---|---|---|
| `vector` | 0.829 | 0.886 | 0.771 | **0.40** | 0.171 | 0 | 1.95 | 8.8 |
| `semantic_rerank` | 0.800 | 0.857 | **0.857** | 0.60 | **0.143** | 0 | 3.27 | 11.7 |
| `associative` | 0.829 | **0.943** | **0.857** | 0.60 | **0.143** | 0 | 3.30 | 13.7 |
| `adaptive` | **0.857** | 0.857 | 0.714 | 0.80 | 0.229 | 0.68 | 6.75 | 20.8 |

44 memories → 57 ready units; 528 entities, 580 entity links, 247 association
edges; `degraded_units: 0`. **Expansion verified live this time**: 30 of the 38
queries that reached it expanded, mean 5.1 added candidates, max 8. An earlier
run of this table was made with the expansion defect described in §4 and is not
comparable — it also used a differently-seeded corpus — so it has been
discarded rather than reported alongside.

**Associative retrieval has the best recall@5, 0.943 against 0.886 for
conventional vector similarity and 0.857 for hybrid search with LLM
reformulation** — and it gets there without giving anything up: identical
recall@1 to vector, identical correctness and abstention to
`semantic_rerank`, for 0.03 more LLM calls and two more seconds. That is the
shape of result the feature was built for.

It is also a **small** margin, and it has to be read as one. 0.943 vs 0.886 on
35 answerable queries is 33 against 31 — two queries. Both are queries where
expansion contributed 8 candidates, and one (`Q12`,
*"What do I know about Hoshi Nakamura's family?"*) is tagged
`indirect_association`, which is exactly the category multi-hop exists for. So
the direction is right and the mechanism is visible in the per-query data. But
with `repeat: 1` against a non-deterministic local model, two queries is inside
the range a re-run could move on its own. **Suggestive, not established** —
`--repeat 3` is the next measurement, and it is cheap now that the corpus is
cached.

**Clarification is a net negative on this corpus, and that is the surprise.**
`adaptive` takes the best recall@1 (0.857) and then posts the *worst*
correctness (0.714, below even plain vector search) and the *worst* false
retrieval (0.80), for double the LLM calls and double the latency. Asking the
user a question put the right memory on top more often and produced a worse
answer from it. The pairwise win-sets say the same thing from the other
direction: `associative` beat `adaptive` on 8 queries while `adaptive` beat
`associative` on 2.

**The uncomfortable number.** False retrieval — the share of deliberately
unanswerable queries a strategy answers anyway — is bad for everything here
(0.40 to 0.80) and worst for the most capable strategy. Being better at
finding things and being better at convincing yourself you have found
something are not separable by any mechanism currently in this system. Any
reading of the recall column that ignores this one is wrong, which is why the
Evaluation page refuses to compute a combined score. Note the granularity: 5
unanswerable queries means this metric moves in steps of 0.2, so 0.40 vs 0.60
is two queries.

**Caveats that matter.** Single run (`repeat: 1`) — on a 40-query set with a
non-deterministic local model, one or two flips are plausible with no code
change at all, so treat the ordering as a hypothesis rather than a measurement.
Only 5 unanswerable queries, so false-retrieval rate has 0.2 granularity.
Correctness is a substring proxy, not a judge. The corpus is synthetic and
authored alongside the system being measured. The simulated user answers from
post-ingestion semantic fields and is therefore an upper bound on how helpful a
real person would be.

---

## 3. Database migrations and API changes

Three migrations, all additive:

| revision | what |
|---|---|
| `134238d01cd1` | `memory_entities`, `unit_entity_links`, `entity_relations`, `unit_associations` + indexes |
| `0c4003935d6f` | `recall_sessions.strategy_trail / final_evidence / final_confidence / abstention_reason` |
| `e5315b2fcda5` | `episodes`, `episode_memberships` |

`alembic upgrade head` from `backend/`. Note that a production database must be
migrated before running the new code against it; the live backend here has its
own database (`membase_prod`), separate from the one `pytest` truncates.

**No existing endpoint changed shape.** Eight new read-only routes
(`/api/graph`, `/api/graph/stats`, `/api/graph/entities`,
`/api/graph/entities/{id}`, `/api/graph/units/{id}/associations`,
`/api/episodes`, `/api/episodes/{id}`, `/api/episodes/for-unit/{id}`), and
additive fields on `RecallSessionOut` and `RecallResponse`. Accepted upload
types extended with audio. Every existing client keeps working untouched.

---

## 4. What running the real model found

The most useful decision in this phase was to drive the actual model over a
real corpus early and often, rather than relying on a fake-LLM suite. Fifteen
defects surfaced that way; a fake-model test suite could not have found any of
them, because every one originates in what a real model actually emits.

The four worth reading:

**The noise filter silently disabled the entire feature.** The
document-frequency ceiling that stops a ubiquitous entity connecting everything
to everything is a *fraction*. With three stored memories, an entity in two of
them sits at 0.67, over the 0.5 ceiling — so the single most informative entity
in the store produced no association at all, and would not have until the
corpus grew large. Compounded by plain IDF being exactly zero when `df == N`,
which on a small store is the normal case rather than an edge case. Both fixed:
the ceiling now also requires an absolute corpus size and mention count, and
specificity is smoothed so a genuinely shared entity never collapses to no
signal.

**Information gain alone picks bad questions.** Maximising it chose *"Was this
about Nature?"* — 0.98 of 2.32 bits, a near-perfect split — over asking about
Markus or Lake Bled. It is close to optimal by the metric and useless to a
person: the user never wrote the word "Nature", so neither answer carries the
information the entropy calculation credits it with. Questions are now ranked
by gain × answerability, which is a *better* estimate of expected gain rather
than a departure from it.

**The same signal was counted three times.** A confirmed cue was rewarded once
by `apply_positive_cue_boost` upstream, again as a cue-match bonus, and a third
time as a shared-entity bonus — pinning a memory about a conference talk to the
score ceiling, at `strong` confidence, next to the photograph the user was
actually looking for, because both mentioned the same person. Confidence now
also judges on the pre-boost score: a user corroborating a detail does not make
a memory a better match for the question.

**The date parser invented a date, in exactly the case it exists to prevent.**
The model produced an `event_date` of *"Not specified in the primary memory
(though related memories suggest a timeframe between 2021 and 2023)"*, and the
bare-year fallback read "2021" out of it and placed the memory on the timeline.
A string that disclaims a date now short-circuits the parser.

Also found and fixed: a memory's whole alias list was folded into *every*
entity it mentioned (so the location "Switzerland" acquired the alias "avs
number", corrupting resolution, since entities are looked up by alias); the
model names the store's owner "Narrator"/"User (Owner of the AHV number)" where
the text said "I", which would have made the owner the highest-degree node in
the graph; one real thing fragmented into several nodes by role; image *medium
and style* descriptors ("minimalist illustration") became `object` entities and
grouped memories by how their pictures were drawn; re-projection never
refreshed or pruned an entity a memory stopped mentioning, so a thing mentioned
by nothing advertised three mentions; deleting a capture left a phantom episode
on the timeline; and a constraint violation during projection poisoned the
caller's transaction and marked a perfectly-ingested capture as failed —
projection now runs inside a SAVEPOINT, because the graph is an index and a bug
in it must never cost a memory.

---

## 5. Remaining limitations

Full list in `docs/TODO.md`. The ones that would change how you read the
results above:

- **Entity resolution is exact-match plus conservative alias folding.** "ETH"
  merges into "ETH Zurich"; "ETH Zurich" and "Swiss Federal Institute of
  Technology" stay two nodes, as do a name and its misspelling. Deliberate —
  wrongly merging two entities is unrecoverable while failing to merge one only
  costs a missed association — but it is a real ceiling on recall.
- **Every graph and episode threshold is an engineering default**, chosen by
  reasoning plus one real-model corpus and not swept. The benchmark is now the
  harness that *could* sweep them; on a 44-memory synthetic corpus a sweep would
  mostly overfit it.
- **Association weights go stale** as the corpus grows, until either endpoint is
  re-projected or `scripts/rebuild_graph.py` runs.
- **Episode clustering is greedy and order-dependent.** Re-running projection
  converges; until then the grouping depends on ingestion order.
- **Visual concepts are only as good as the caption**, and caption quality on
  real photographs still has not been evaluated — the demonstration dataset
  uses generated illustrations, which exercises the path but says nothing about
  photographs.
- **Audio transcription accuracy is not established** — only synthesised speech
  was available to test with. The model demonstrably hears the audio but
  rewrites clauses it cannot make out rather than marking them uncertain.
- **No way to correct a wrong association or episode assignment** except by
  correcting the underlying memory, because both APIs are deliberately
  read-only. Whether a user-visible "these are not related" signal should exist,
  and how it would survive a re-projection, is an open design question.

---

## 6. Running and demonstrating it

```sh
nix develop
scripts/model-doctor.sh
scripts/dev-db.sh start
(cd backend && alembic upgrade head)
scripts/run-dev.sh                      # API on 127.0.0.1:8000
python3 scripts/seed_demo.py            # synthetic demo dataset, several minutes
python3 scripts/build_frontend.py
python3 -m http.server --directory dist 8099
```

Then in the browser console, once: `localStorage.setItem("membase.backendUrl",
"http://127.0.0.1:8000")` and reload. (The dev server sends no CORS headers —
nginx owns CORS in production — so cross-origin needs a same-origin `/api/*`
proxy in front of `dist/`. See `docs/USAGE.md`.)

**The demonstration.** Open `/app/recall/` and ask:

> I am trying to remember a photograph from a trip a few years ago. I think we
> were near a lake, possibly in Switzerland or Austria. There were mountains in
> the background, and I remember taking it with a friend.

The store contains three lake trips that each match that description partially
and differ in one attribute. The system does not guess. It shows the five
candidates it is choosing between, each with its evidence and an honest `weak`
confidence, and asks one concrete question chosen for how much it would
actually resolve — measured, and shown, in the retrieval trail.

Answer it the way a real person would, including correcting your own wrong
guess:

> Actually no, not Switzerland. I am fairly sure it was with Markus.

Cue extraction pulls three cues from that one sentence — the denial of the
pending question, the denial of Switzerland, and the new detail "Markus" — rules
out the memories the denials exclude, and raises the Lake Bled photograph to
`strong`. From the result you can open its memory network (nine one-hop
associations, each with a deterministic explanation of *why*), its episode, and
the photograph itself.

**The demonstration does not always take that path, and the reason is worth
understanding rather than tuning away.** The query hedges — "I think",
"possibly in Switzerland or Austria" — but the system treats those terms with
the same weight as a stated fact. The Lake Thun memory is in Switzerland and
literally says "photographed the mountains", so on some corpus states it
dominates clearly enough that there is no ambiguity left to resolve, and the
session answers in round one with Lake Thun rather than asking.

That answer is *defensible* retrieval: it is the best match for the words the
user actually used, the user's own recollection of the country is what is
wrong, and the system reports `moderate` rather than `strong` and surfaces the
contradiction it found by association — "You may be confusing the location
because you have noted that Lake Bled in Slovenia looks very similar to the
Alps around Interlaken." It is not a confident falsehood.

But it is not what a memory system for *imperfect* recall should do with a
hedged clue. "Possibly in Switzerland" is evidence that the user does not know
where it was, and the system reads it as evidence that it was in Switzerland.
Detecting hedged query terms and discounting them is a real design gap, logged
in docs/TODO.md; it was not attempted here because it changes what the system
treats as a constraint, which deserves its own evaluation rather than being
slipped in to make a demonstration land more reliably.

`/app/explorer/` shows the same structure as a graph; `/app/timeline/` shows it
chronologically, with undated memories labelled rather than placed somewhere
invented; `/app/evaluation/` shows whether any of it actually beats
conventional retrieval.
