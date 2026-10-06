# membase

A self-hosted augmented personal memory system (MVP). Write memories in
plain language, optionally attach a file, and later ask a question in
different words than you originally used — the system retrieves the
relevant memory and answers from it, citing exactly which memory it used.

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

Open http://127.0.0.1:8000/ to store a memory, and
http://127.0.0.1:8000/recall to query it. The server binds to localhost
only, by design (see `docs/DECISIONS.md`).

To stop Postgres: `scripts/dev-db.sh stop`.

## Configuring the local LLM

All LLM access goes through `app.llm.LLMProvider` (see
`backend/app/llm/base.py`); the only implementation today is
`OllamaProvider`. Configure via env vars (see `.env.example`, all exported
automatically by `nix develop`'s shellHook):

| Var | Default | Meaning |
|---|---|---|
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollama endpoint |
| `MEMBASE_GENERATION_MODEL` | `qwen2.5:14b-instruct-q4_K_M` | segmentation, enrichment, agent decisions, answers — one model for everything, see `docs/DECISIONS.md` |
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

A unit may surface at most one bounded clarification question if
something is genuinely unclear (an unfamiliar term, an unresolvable
relative date). **A unit with an open question is durably saved but not
yet searchable** — it only becomes visible to recall once answered (via
`POST /api/memory-units/{id}/clarify`) or explicitly discarded (`POST
.../discard`). The answer becomes part of that unit's permanent semantic
representation (aliases, definitions), not a transient chat turn.

The original text and files are never modified by any of this. A
committed memory can be corrected later in natural language (`POST
/api/memory-units/{id}/correct`) or deleted (`DELETE /api/captures/{id}`,
also removes its files) — there's no automatic forgetting/expiry.

## How retrieval works

`POST /api/recall` (or the web form at `/recall`) runs a small bounded
retrieval agent (`backend/app/retrieval/agent.py`): it searches (hybrid
vector + full-text over each unit's original text and derived semantic
representation, *and* over chunked attachment text), decides if that's
enough to answer, and if not, proposes up to 2 alternative search
phrasings (bounded to 3 rounds total) before answering. The LLM is
instructed to answer only from retrieved evidence and to mark the answer
"grounded" only if it actually addresses the question; if not, the API
returns the fixed string "I don't have a stored memory about this." and
no sources — never a fabricated guess.

## Tests & evaluation

```sh
pytest -q                      # fast unit tests (fakes) + real-LLM integration tests
pytest -q -m "not integration" # skip the ones that need Ollama running
scripts/run-eval.sh            # 30-case hand-written eval set
scripts/sweep-threshold.sh     # retrieval-threshold calibration experiment
python3 scripts/reembed.py --dry-run   # report stale/missing embeddings
```

## Known limitations

See `docs/REPORT.md` for the full list; the short version: no image
understanding, no chunking-aware re-embedding backfill granularity beyond
whole-row staleness, and the relevance threshold for vector search is
calibrated against a 30-case eval set rather than a large production
corpus (see `docs/DECISIONS.md` for the sweep that picked it).
