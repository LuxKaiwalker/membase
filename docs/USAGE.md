# Usage

Practical run/operate notes for this codebase — what the top-level
README describes conceptually, this is the "how to actually run it"
companion: requirements, starting the dev stack, configuring the
local LLM, the capture/clarify/recall request flow, and the test/eval
suite. See the top-level `README.md` for what this project is and
why; `docs/ARCHITECTURE.md` for the full design.

Write memories in plain language, optionally attach a file, and later
ask a question in different words than you originally used — the
system retrieves the relevant memory and answers from it, citing
exactly which memory it used.

This is explicitly **not** a notes app or a general RAG chatbot: it answers
only from what you've actually stored, and says "I don't have a stored
memory about this" rather than guessing. See `docs/ARCHITECTURE.md` for the
full design and `docs/REPORT.md` for what's implemented vs. not.

## Requirements

- NixOS (or any Linux with Nix flakes enabled) — `flake.nix` provides
  everything else (Python, Postgres+pgvector, Ollama CLI).
- A running [Ollama](https://ollama.com) service reachable at
  `http://127.0.0.1:11434`. On NixOS this is typically a system service
  (`systemctl status ollama`); elsewhere, `ollama serve`.

## Running it

Run these from the repo root — `scripts/*.sh` are root-relative paths, not
relative to `backend/`:

```sh
nix develop                 # enters the dev shell; prints next steps
scripts/model-doctor.sh     # checks Ollama, pulls missing models, verifies generation+embeddings
scripts/dev-db.sh start     # inits/starts a project-local Postgres
(cd backend && alembic upgrade head)   # first time only / after schema changes
scripts/run-dev.sh          # starts uvicorn with --reload
```

The backend is API-only (no HTML page routes, no `/static` mount --
everything outside `/api/*` 404s, see `docs/DECISIONS.md` "API-only
backend: stop serving the frontend's own HTML/static files"), so
`127.0.0.1:8000` itself has no web UI to open. Hit the API directly, e.g.:

```sh
curl -X POST 127.0.0.1:8000/api/captures -F text="Dentist appointment next Tuesday at 3pm"
curl -X POST 127.0.0.1:8000/api/recall -d '{"query": "when is my dentist appointment"}' -H 'Content-Type: application/json'
```

The actual web UI (`backend/app/templates/` + `backend/app/static/`) is
published separately to GitHub Pages (see `docs/DEPLOYMENT.md`
"Publishing the frontend to GitHub Pages") and its JS has this project's
production backend URL hardcoded at the source level (`BACKEND_URL` in
`app.js`, see `docs/DECISIONS.md` "Hardcode the backend URL instead of
asking for it at login") -- it does not point at a local dev server out
of the box. To exercise the UI against your local backend instead,
temporarily point `BACKEND_URL` at `http://127.0.0.1:8000`, run
`python3 scripts/build_frontend.py`, then serve `dist/` with a plain
static file server (e.g. `python3 -m http.server --directory dist`) --
the rendered pages reference `/static/...` as an absolute path, so they
need to be served from a web root, not opened directly as `file://`.

To stop Postgres: `scripts/dev-db.sh stop`.

## Configuring the local LLM

All LLM access goes through `app.llm.LLMProvider` (see
`backend/app/llm/base.py`); the only implementation today is
`OllamaProvider`. Configure via env vars (see `.env.example`, all exported
automatically by `nix develop`'s shellHook):

| Var | Default | Meaning |
|---|---|---|
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollama endpoint |
| `MEMBASE_GENERATION_MODEL` | `gemma4:12b` | segmentation, enrichment, agent decisions, answers, image captioning — one multimodal model for everything, see `docs/DECISIONS.md` |
| `MEMBASE_EMBED_MODEL` | `nomic-embed-text` | embeddings (768 dims) |

Changing `MEMBASE_EMBED_MODEL` to a model with a different dimension needs
a schema migration (`embedding` is a fixed-size `pgvector(768)` column) —
`scripts/model-doctor.sh` and `scripts/reembed.py` both refuse to proceed
rather than silently corrupt things if the configured dimension doesn't
match the schema. After changing the embedding model (same dimension) or
recovering from an outage that left some rows unembedded, run
`python3 scripts/reembed.py` (or `--dry-run` first) to backfill.

## Creating a memory

POST (or use the web form at `/`) some text and optional files to
`/api/captures`. One submission (a "capture") may become **one or more
independently retrievable memory units** — the system decides this via
segmentation, not the user (see `docs/ARCHITECTURE.md`).

A unit may surface a clarification question if something is genuinely
unclear (an unfamiliar term, an unresolvable relative date) — a fast,
narrow check that runs *before* the heavier semantic-extraction step, so
you see the question without waiting on it (see `docs/ARCHITECTURE.md`).
Clarification is strictly sequential — one question at a time, never a
pre-generated batch — and bounded to 3 rounds total if an answer reveals
something new worth asking, so the system can't turn into an
interrogation. **A unit with an open question is durably saved but not
yet searchable** — it only becomes visible to recall once answered (via
`POST /api/memory-units/{id}/clarify`) or explicitly discarded (`POST
.../discard`). Answering is optional — a blank answer just skips the
question, enforced only as a client-side preference (the settings menu's
"obligatory" toggle), never by the API. A real answer becomes part of
that unit's permanent semantic representation (aliases, definitions),
not a transient chat turn.

Once no question is left pending, the heavier semantic-extraction step
(enrichment + embedding) runs as a background task rather than blocking
the save request — the unit sits briefly in `status="processing"` and
the web UI polls until it's `"ready"` (see `docs/ARCHITECTURE.md`), so
submitting a second memory never waits on the first one's enrichment.
Once ready, a background "structurer" pass deepens the unit's semantic
representation further over time, without ever blocking anything on it
(see `docs/ARCHITECTURE.md`'s "The membase structurer" section).

The original text and files are never modified by any of this. A
committed memory can be corrected later in natural language (`POST
/api/memory-units/{id}/correct`) or deleted (`DELETE /api/captures/{id}`,
also removes its files) — there's no automatic forgetting/expiry.

## How retrieval works

`POST /api/recall` (used by the web form at `/recall`) runs a bounded
retrieval agent (`backend/app/retrieval/agent.py:run_agent`): hybrid
search (vector + full-text, over each unit's original text and derived
semantic representation, *and* over chunked attachment text), up to 3
rounds of "is this enough, or should I try different phrasing" before
answering. The LLM is instructed to answer only from retrieved evidence
and to mark the answer "grounded" only if it actually addresses the
question; if not, or if nothing is found at all, you get the fixed
string "I don't have a stored memory about this." and no sources — never
a fabricated guess.

There's also a multi-round **interactive** mode
(`POST /api/recall/sessions`, `backend/app/retrieval/session.py`) that
can ask a clarifying question when several stored memories are
plausible, instead of guessing, and accumulate your replies across
rounds. It's fully implemented, tested, and reachable via the API — it's
just not what the `/recall` page uses by default right now. A benchmark
(`eval/run_interactive_benchmark.py`) found it underperforming the
simpler one-shot agent overall on a deliberately ambiguous test corpus;
see `docs/DECISIONS.md` for the result and the root cause, and
`docs/ARCHITECTURE.md`'s "Iterative recall" section for the full
mechanics.

## Tests & evaluation

```sh
pytest -q                      # fast unit tests (fakes) + real-LLM integration tests
pytest -q -m "not integration" # skip the ones that need Ollama running
scripts/run-eval.sh            # 30-case one-shot hand-written eval set
scripts/run-session-eval.sh    # iterative-recall eval set (multi-round, ambiguity, correction)
scripts/run-interactive-benchmark.sh  # one-shot vs. LLM-assisted vs. interactive A/B/C comparison
scripts/sweep-threshold.sh     # retrieval-threshold calibration experiment
scripts/benchmark-recall-latency.sh   # per-round latency, broken down by LLM call kind
python3 scripts/reembed.py --dry-run   # report stale/missing embeddings
```

## Known limitations

See `docs/REPORT.md` for the (phase-dated) full list and `docs/TODO.md`
for the current one; the short version today: uploaded images are
captioned by the generation model itself rather than a dedicated
OCR/vision model (not independently quality-evaluated yet, see
`docs/TODO.md`), there's no chunking-aware re-embedding backfill
granularity beyond whole-row staleness, and the relevance threshold for
vector search is calibrated against a 30-case eval set rather than a
large production corpus (see `docs/DECISIONS.md` for the sweep that
picked it).
