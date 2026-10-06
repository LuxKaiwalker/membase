# membase — working notes for Claude

Augmented personal memory MVP. **Status: working MVP, see
`docs/REPORT.md`.** Full product spec lives in the task that created this
repo; durable project docs are under `docs/` (`ARCHITECTURE.md`,
`DECISIONS.md`, `TODO.md`, `REPORT.md`). This file is operational notes
for working in *this* environment — keep it short, prune anything that
becomes obvious from the code or from `docs/`.

## Environment facts

- NixOS 26.05, flakes enabled, x86_64-linux, 16 cores / 32GB RAM, NVIDIA GPU present.
- Ollama runs as a **system service** (`systemctl status ollama`), already
  listening on `127.0.0.1:11434`. Models already present: `qwen2.5:7b-instruct`
  (default generation model), `qwen2.5-coder:1.5b`, `qwen2.5:14b-instruct-q4_K_M`.
  Embedding model `nomic-embed-text` (768 dims) was pulled via
  `scripts/pull-models.sh` for this project.
- No docker/podman available. No system-wide Postgres. Everything the app
  needs beyond Ollama is provided by `flake.nix`.
- All Python dependencies come from `nixpkgs.python313Packages` via
  `python3.withPackages` in `flake.nix` — there is no `requirements.txt`,
  `venv`, or `pip install` in this repo. See footgun #2 below for why.

## Dev workflow

```
nix develop                 # enters the shell, prints next steps
scripts/dev-db.sh start     # init/start project-local Postgres (port 5544)
scripts/pull-models.sh      # idempotent: ensures nomic-embed-text is pulled
(cd backend && alembic upgrade head)   # first run / after a schema change
uvicorn app.main:app --reload --app-dir backend --host 127.0.0.1 --port 8000
pytest -q                   # fast fakes + real-LLM integration tests
scripts/run-eval.sh         # hand-written eval set, isolated DB (membase_eval)
scripts/dev-db.sh stop
```

Env vars needed by the app (`DATABASE_URL`, `PGHOST_DIR`, `PGPORT`, `OLLAMA_HOST`,
`UPLOADS_DIR`) are exported by the flake's `shellHook` — don't hardcode them
elsewhere.

## NixOS footguns encountered (and how this repo avoids them)

1. **Flakes only see git-tracked files.** `nix develop`/`nix flake check` on
   an untracked `flake.nix` fails with "Path ... is not tracked by Git".
   Just-staged (not committed) is enough. Keep this in mind after creating
   any new file the flake references.
2. **pip-installed binary wheels generally don't run on NixOS** (no
   `/lib64/ld-linux`, libs not on the loader's search path) — classic trap
   for anything with compiled extensions (psycopg2, pillow, pydantic-core,
   etc.). Avoided entirely here by sourcing every Python dependency from
   nixpkgs instead of PyPI. If a future dependency isn't packaged in
   nixpkgs, don't reach for `pip install` inside the nix env — that's a
   decision to flag, not to silently work around (options: nixpkgs
   overlay/override, `nix-ld`, or vendoring).
3. **`initdb` needs a locale archive that NixOS doesn't provide by
   default** — fails/warns without one. Fixed by using `--locale=C.UTF-8`
   in `scripts/dev-db.sh` and exporting `LOCALE_ARCHIVE` to
   `pkgs.glibcLocales` in the flake's `shellHook`.
4. **No assumption of a system Postgres service.** This project runs its
   own project-local instance (`scripts/dev-db.sh`) with `PGDATA` and the
   unix socket dir both under `.data/` (gitignored), on port 5544 — isolated
   from anything else on the machine, no root/systemd needed.
5. **Don't start a second Ollama.** One is already running system-wide on
   `:11434`. `ollama serve` again would fail to bind the port. The app just
   talks to the existing instance via `OLLAMA_HOST`.
6. **The nix store is read-only.** All writable app state (Postgres data,
   uploaded files) lives under `.data/` in the repo, never under a store
   path.
7. The web server binds `127.0.0.1` only, never `0.0.0.0` — required by the
   project's privacy goals, and avoids surprises if the NixOS firewall
   config ever changes.

Design decisions, architecture, and open questions all now live in
`docs/` and aren't duplicated here — start with `docs/REPORT.md` for the
current state, `docs/ARCHITECTURE.md` for how it fits together, and
`docs/DECISIONS.md` / `docs/TODO.md` for why/what's unresolved.
