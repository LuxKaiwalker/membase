# membase

A self-hosted augmented personal memory system (MVP). Store memories in
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
  `http://127.0.0.1:11434` with a generation model pulled (default:
  `qwen2.5:7b-instruct`). On NixOS this is typically a system service
  (`systemctl status ollama`); elsewhere, `ollama serve`.

## Running it

```sh
nix develop                 # enters the dev shell; prints next steps
scripts/pull-models.sh      # pulls the embedding model (nomic-embed-text)
scripts/dev-db.sh start     # inits/starts a project-local Postgres
(cd backend && alembic upgrade head)   # first time only / after schema changes
uvicorn app.main:app --reload --app-dir backend --host 127.0.0.1 --port 8000
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
| `MEMBASE_GENERATION_MODEL` | `qwen2.5:7b-instruct` | text generation + structured extraction |
| `MEMBASE_EMBED_MODEL` | `nomic-embed-text` | embeddings (768 dims) |

Changing `MEMBASE_EMBED_MODEL`/dims requires re-embedding existing memories
— there's no migration/backfill tool for this in the MVP (see
`docs/TODO.md`).

## Creating a memory

POST (or use the web form at `/`) a caption and optional files to
`/api/memories`. The system extracts people/location/topics/date and a
summary in the background; if something materially ambiguous comes up
(e.g. "yesterday" relative to no clear date) it may ask one clarifying
question via `pending_clarification`, answered through
`POST /api/memories/{id}/clarify`. The original caption and files are
never modified by any of this — see `docs/ARCHITECTURE.md#data-integrity`.

## How retrieval works

`POST /api/recall` (or the web form at `/recall`) hybridizes:
- vector search over embeddings (pgvector, HNSW/cosine) — catches
  paraphrased/indirect cues,
- Postgres full-text search over caption + attachment text + derived
  metadata (GIN) — catches literal keyword overlap,

fuses the two with a weighted-score rerank, and only passes memories that
clear a relevance bar to the LLM for answer generation. The LLM is
instructed to answer only from that evidence and to mark the answer
"grounded" only if it actually addresses the question; if not, the API
returns the fixed string "I don't have a stored memory about this." and no
sources — never a fabricated guess. See `docs/ARCHITECTURE.md` for the full
pipeline and `docs/TODO.md` for the retrieval threshold's known limits.

## Tests & evaluation

```sh
pytest -q                 # fast unit tests (fakes) + real-LLM integration tests
pytest -q -m "not integration"   # skip the ones that need Ollama running
scripts/run-eval.sh       # small hand-written eval set (exact/paraphrase/indirect/multi-concept/distractor/no-answer)
```

## Known limitations

See `docs/REPORT.md` for the full list; the short version: no image
understanding (images are stored but not captioned/OCR'd), no backfill
when embedding fails or the embedding model changes, clarification
questions are occasionally spurious or mistargeted (small local model),
and the relevance threshold for vector search is a hand-picked value from
a handful of manual trials, not a tuned one.
