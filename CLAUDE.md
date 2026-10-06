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
  - `qwen2.5:14b-instruct-q4_K_M` for *every* generation/structured-output
    call (segmentation, semantic enrichment, agent decisions, answers).
    `qwen2.5:7b-instruct` visibly under-performed on segmentation and
    clarification judgment in manual testing (over-split coherent
    episodes; never recognized an unfamiliar term as worth asking about).
  - `nomic-embed-text` (768 dims) for embeddings.
  - **Don't mix in a third generation model.** This GPU has 11GB VRAM;
    14b-q4 (~9GB) + nomic-embed (~0.3GB) coexist without being swapped out
    of VRAM between calls. Adding a second *generation* model (e.g. also
    using 7b somewhere) means Ollama evicts/reloads between calls,
    costing ~7-14s *per call* -- confirmed empirically, see git history
    for the measurement. `scripts/model-doctor.sh` checks the two models
    actually configured are present and working; it does not pull extras.
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
scripts/run-eval.sh         # 30-case hand-written eval set, isolated DB (membase_eval)
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
  place in that file.
