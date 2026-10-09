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
set -euo pipefail

: "${MEMBASE_ROOT:?run inside 'nix develop'}"

exec uvicorn app.main:app --app-dir "$MEMBASE_ROOT/backend" --host 127.0.0.1 --port 8000 --workers 1 --log-level info
