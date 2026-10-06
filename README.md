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
scripts/model-doctor.sh      # checks Ollama, pulls missing models, verifies generation+embeddings
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
`scripts/model-doctor.sh` will flag a mismatch rather than silently
breaking. There's no automatic re-embedding/backfill tool for existing
memories after a model change yet (see `docs/TODO.md`).

## Creating a memory

POST (or use the web form at `/`) some text and optional files to
`/api/captures`. One submission (a "capture") may become **one or more
independently retrievable memory units** — the system decides this via
segmentation, not the user (see `docs/ARCHITECTURE.md#memory-representation`).
Each unit gets a derived semantic representation (meaning, aliases,
definitions, relationships) and may surface at most one bounded
clarification question if something is genuinely unclear; the answer
becomes part of that unit's permanent semantic representation, not a
transient chat turn. The original text and files are never modified by
any of this.

## How retrieval works

`POST /api/recall` (or the web form at `/recall`) runs a small bounded
retrieval agent (`backend/app/retrieval/agent.py`): it searches (hybrid
vector + full-text over each unit's original text *and* derived semantic
representation), decides if that's enough to answer, and if not,
proposes up to 2 alternative search phrasings (bounded to 3 rounds total)
before answering. The LLM is instructed to answer only from retrieved
evidence and to mark the answer "grounded" only if it actually addresses
the question; if not, the API returns the fixed string "I don't have a
stored memory about this." and no sources — never a fabricated guess.

## Tests & evaluation

```sh
pytest -q                 # fast unit tests (fakes) + real-LLM integration tests
pytest -q -m "not integration"   # skip the ones that need Ollama running
scripts/run-eval.sh       # hand-written eval set (segmentation/clarification-alias/exact/paraphrase/indirect/relation/distractor/no-answer/...)
```

## Known limitations

See `docs/REPORT.md` for the full list; the short version: no image
understanding, no backfill when embeddings are missing/stale, the
relevance threshold for vector search is a hand-picked value from a
handful of manual trials rather than a tuned one, and clarification
enriches a memory but doesn't gate its availability (both documented as
deliberate MVP scope decisions in `docs/DECISIONS.md`).
