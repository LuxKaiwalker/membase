#!/usr/bin/env bash
# Runs the dev server. Uses $MEMBASE_ROOT (exported by the flake's
# shellHook) for an absolute --app-dir, so this works regardless of
# which directory you're in when you invoke it -- unlike a bare
# `uvicorn app.main:app --app-dir backend`, which only resolves
# correctly if your cwd happens to be the repo root.
set -euo pipefail

: "${MEMBASE_ROOT:?run inside 'nix develop'}"

exec uvicorn app.main:app --reload --app-dir "$MEMBASE_ROOT/backend" --host 127.0.0.1 --port 8000
