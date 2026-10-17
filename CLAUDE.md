# membase — working notes for Claude

Augmented personal memory MVP. **Status: working MVP, see
`docs/REPORT.md`.** Full product spec lives in the task that created this
repo; durable project docs are under `docs/` (`ARCHITECTURE.md`,
`DECISIONS.md`, `TODO.md`, `REPORT.md`). This file is operational notes
for working in *this* environment — keep it short, prune anything that
becomes obvious from the code or from `docs/`.

## Environment facts

- NixOS 26.05, flakes enabled, x86_64-linux, 16 cores / 32GB RAM, NVIDIA
  GPU with 11GB VRAM.
- Ollama runs as a **system service** (`systemctl status ollama`), already
  listening on `127.0.0.1:11434`.
- The app uses exactly two models, both via Ollama, chosen after testing
  (not the spec's suggested default) -- see docs/DECISIONS.md:
  - `gemma4:12b` for *every* generation/structured-output call
    (segmentation, semantic enrichment, agent decisions, answers) --
    migrated from `qwen2.5:14b-instruct-q4_K_M`, see docs/DECISIONS.md.
    Natively multimodal (text/image/audio, no separate vision encoder),
    which is what `app/ingestion/pipeline.py`'s `_describe_image` uses to
    make uploaded images searchable.
  - `nomic-embed-text` (768 dims) for embeddings -- unchanged by the
    gemma4 migration, embeddings are a separate concern from generation.
  - `OllamaProvider` talks to Ollama's `/api/chat` endpoint, not
    `/api/generate` -- required for gemma4, whose grammar-constrained
    (JSON schema) output via `/api/generate` was measured to silently
    truncate array-typed fields; `/api/chat` doesn't have this problem.
    See docs/DECISIONS.md "Gemma 4 migration" before ever touching
    `app/llm/ollama_provider.py`'s endpoint choice.
  - **Don't mix in a second generation model.** This GPU has 11GB VRAM;
    measured (`nvidia-smi`, not just the published quantized size):
    `gemma4:12b` alone uses ~8.7GB, `nomic-embed-text` ~0.25GB, leaving
    only ~1GB of headroom -- less than qwen's setup had. There is no
    realistic room for a second generation model here; adding one would
    risk the same eviction/reload thrashing previously confirmed
    empirically with qwen's 7b/14b combination (~7-14s per call, see git
    history). `scripts/model-doctor.sh` checks the two models actually
    configured are present and working; it does not pull extras.
- No docker/podman available. No system-wide Postgres. Everything the app
  needs beyond Ollama is provided by `flake.nix`.
- All Python dependencies come from `nixpkgs.python313Packages` via
  `python3.withPackages` in `flake.nix` — there is no `requirements.txt`,
  `venv`, or `pip install` in this repo.

## Dev workflow

All of this runs from the **repo root** -- `scripts/*.sh` are root-relative
paths, and `scripts/run-dev.sh` specifically exists because a bare
`uvicorn ... --app-dir backend` silently breaks (`ModuleNotFoundError`) if
your cwd is `backend/` instead of root when you run it.

```
nix develop                 # enters the shell, prints next steps
scripts/model-doctor.sh     # Ollama reachable? models present? generation/embed actually work?
scripts/dev-db.sh start     # init/start project-local Postgres (port 5544)
(cd backend && alembic upgrade head)   # first run / after a schema change
scripts/run-dev.sh          # uvicorn --reload, --app-dir resolved from $MEMBASE_ROOT
pytest -q                   # fast fakes + real-LLM integration tests
scripts/run-eval.sh         # 30-case one-shot eval set, isolated DB (membase_eval)
scripts/run-session-eval.sh # iterative-recall eval set (multi-round/ambiguity), same isolated DB
scripts/sweep-threshold.sh  # retrieval-threshold calibration experiment, same isolated DB
python3 scripts/reembed.py --dry-run   # report stale/missing embeddings; drop --dry-run to fix
scripts/dev-db.sh stop
```

Env vars needed by the app (`DATABASE_URL`, `PGHOST_DIR`, `PGPORT`, `OLLAMA_HOST`,
`UPLOADS_DIR`) are exported by the flake's `shellHook` — don't hardcode them
elsewhere.

## NixOS footguns encountered (and how this repo avoids them)

1. **Flakes only see git-tracked files.** `nix develop`/`nix flake check` on
   an untracked `flake.nix` fails with "Path ... is not tracked by Git".
   Just-staged (not committed) is enough.
2. **pip-installed binary wheels generally don't run on NixOS.** Avoided
   entirely by sourcing every Python dependency from nixpkgs instead of
   PyPI.
3. **`initdb` needs a locale archive that NixOS doesn't provide by
   default.** Fixed via `--locale=C.UTF-8` in `scripts/dev-db.sh` plus
   `LOCALE_ARCHIVE` exported in the flake's `shellHook`.
4. **No assumption of a system Postgres service.** This project runs its
   own project-local instance (`scripts/dev-db.sh`), isolated under `.data/`.
5. **Don't start a second Ollama;** one already runs system-wide. Don't
   mix generation models either -- see "Environment facts" above, this
   one isn't NixOS-specific but bit us the same way (unexpected 7-14s
   stalls from model eviction/reload).
6. **The nix store is read-only.** All writable app state lives under
   `.data/` in the repo, never under a store path.
7. **`MEMBASE_ROOT` must be resolved via `git rev-parse --show-toplevel`,
   not `$PWD`.** `$PWD` is captured once at shell-entry; if `nix develop`
   is run from a subdirectory (e.g. `backend/`), everything derived from
   a `$PWD`-based root (Postgres data dir, uploads dir, `--app-dir`)
   silently points at the wrong place for the rest of that shell's life,
   including after a later `cd` to the right place. Hit this for real;
   see git history. `git rev-parse --show-toplevel` is correct regardless
   of invocation directory.
8. The web server binds `127.0.0.1` only, never `0.0.0.0`.
9. **`pytest`'s `db` fixture `TRUNCATE`s the whole dev database before
   every single test** (see `backend/tests/conftest.py`) -- there is no
   separate test database. Running two `pytest` invocations concurrently
   (or `pytest` alongside manual testing against a running dev server)
   against this same Postgres instance causes real, confusing failures
   (a capture inserted by one process vanishing mid-transaction because
   an unrelated test in the other process truncated it) that look like
   application bugs but aren't -- hit this for real while validating the
   gemma4 migration, see docs/DECISIONS.md. Run one DB-touching thing at
   a time; `scripts/run-eval.sh`/`run-session-eval.sh`/
   `sweep-threshold.sh` avoid this by using an isolated `membase_eval`
   database instead, but the default `pytest` run does not.
10. **The live production backend now runs against its own `membase_prod`
    database, separate from the `membase` dev database `pytest` targets**
    (`scripts/run-prod.sh`, same isolation pattern as #9's `membase_eval`
    -- all three are just database names on the same project-local
    Postgres cluster, same `PGHOST_DIR`/`PGPORT`). This exists because
    footgun #9 used to apply to the *production* process too: before this
    split, `pytest`'s `TRUNCATE` ran against the exact same database the
    live backend was reading/writing in production, and really did wipe
    every stored memory for real on this machine (see git history, the
    "separate production database" commit) while this backend was
    actively serving traffic. Never point `scripts/run-prod.sh` (or
    whatever ends up running the live backend) at the plain `membase`
    database again.

## Key architecture terms (see docs/ARCHITECTURE.md for the full picture)

- **Capture**: one user submission (`captures` table), authoritative
  `raw_text`, never rewritten.
- **MemoryUnit**: one independently-retrievable piece of memory
  (`memory_units` table), `capture_id` is the "origin group". A capture
  may produce one unit (coherent episode) or several (unrelated facts) --
  the LLM decides, see `app/ingestion/segmentation.py`.
- **Authority hierarchy**: `authoritative_content` (user-typed) >
  `clarifications` (user-answered, append-only) > `semantic` (LLM-derived,
  regenerable) > `retrieval_text`/`searchable_text`/`embedding` (fully
  derived indexes).
- **Status gate**: `draft` -> `awaiting_clarification` (open question;
  NOT searchable -- excluded from `hybrid_search()` by the `status='ready'`
  SQL filter) -> `ready` (searchable) -> `failed` (unexpected exception
  during derived processing, authoritative content untouched, retryable)
  / `discarded` (explicit user cancel of an awaiting unit).
- **Clarification is semantic enrichment, not conversation**: an answer
  gets folded into the unit's semantic representation (aliases,
  definitions) and triggers re-embedding, not just appended to a chat log.
  Same mechanism powers user-initiated correction
  (`POST /memory-units/{id}/correct`), which doesn't consume the bounded
  system-clarification round budget.
- **AttachmentChunk**: large documents are chunked (page-based for PDFs,
  fixed-window for txt/md) and each chunk independently full-text + vector
  indexed, so a fact buried in a long document is findable rather than
  only ever showing up as "first 500 chars of the attachment."
- **Agent** (`app/retrieval/agent.py`): bounded deterministic state
  machine (search -> evaluate -> reformulate, max 3 rounds), not a
  general tool-calling framework. Searches both memory units and
  attachment chunks every round. Grounding is enforced in exactly one
  place in that file. `automatic_search()` is this loop extracted for
  reuse by RecallSession below; `run_agent()` is the one-shot wrapper
  (legacy `POST /api/recall`, eval harness).
- **RecallSession** (`app/retrieval/session.py`): recall as a persisted,
  multi-round process, not one request/response. Accumulates user cues
  (`certain`/`likely`/`uncertain` × `positive`/`negative`) and agent
  search hypotheses across rounds; asks one concrete discriminating
  question (`app/retrieval/discriminate.py`) when several stored
  memories are still plausible, instead of guessing. Cues/hypotheses
  live only on the session row -- never written to MemoryUnit/Capture.
  See docs/ARCHITECTURE.md "Iterative recall" for the full state
  machine.
