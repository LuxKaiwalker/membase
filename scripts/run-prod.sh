#!/usr/bin/env bash
# Runs the server for actual deployment -- see docs/DEPLOYMENT.md.
#
# Differs from run-dev.sh in exactly two ways, both deliberate:
#   - no --reload: a file-watcher + extra process has no purpose once
#     you're not actively editing code, and --reload is measurably
#     slower per-request.
#   - a single worker, explicitly: this app keeps a few process-local
#     singletons (the LLM provider, the SQLAlchemy engine/sessionmaker,
#     both @lru_cache'd -- see app/llm/__init__.py, app/db.py) that were
#     never designed or tested for multiple worker processes sharing
#     one FastAPI app. Given this app's own stated scope (single user,
#     local-first, one Ollama instance that itself only serves one
#     request at a time), there is no throughput reason to want more
#     than one worker anyway.
#
# Still binds 127.0.0.1 only, same as run-dev.sh -- putting this
# directly on a public interface is NOT what this script is for. If
# this needs to be reachable from the internet, put a reverse proxy
# (nginx, Caddy, Tailscale Funnel, etc.) in front of it, terminate TLS
# there, and forward to this process -- see docs/DEPLOYMENT.md for why,
# and for what authentication to add in front of it first.
#
# Uses its OWN database ('membase_prod'), separate from the 'membase'
# dev database -- both live on the same project-local Postgres cluster
# (same PGHOST_DIR/PGPORT from the flake's shellHook), just a different
# database name, same isolation pattern run-eval.sh already used for
# 'membase_eval'. This exists because `pytest`'s `db` fixture TRUNCATEs
# captures/memory_units/attachments/attachment_chunks/recall_sessions
# before every single test (see backend/tests/conftest.py) -- before
# this script existed, running the test suite while this process was
# live, serving real traffic against the SAME 'membase' database,
# silently wiped every stored memory. Hit this for real; see git
# history. Running `pytest` (or `scripts/run-eval.sh`, etc.) can now
# never touch what this script's data lives in, regardless of what
# else is running on this machine at the same time.
set -euo pipefail

: "${MEMBASE_ROOT:?run inside 'nix develop'}"
: "${PGUSER:?run inside 'nix develop'}"
: "${PGHOST_DIR:?run inside 'nix develop'}"
: "${PGPORT:?run inside 'nix develop'}"

export PGDATABASE=membase_prod
export DATABASE_URL="postgresql+psycopg://$PGUSER@/$PGDATABASE?host=$PGHOST_DIR&port=$PGPORT"

"$MEMBASE_ROOT/scripts/dev-db.sh" start
(cd "$MEMBASE_ROOT/backend" && alembic upgrade head)

exec uvicorn app.main:app --app-dir "$MEMBASE_ROOT/backend" --host 127.0.0.1 --port 8000 --workers 1 --log-level info
