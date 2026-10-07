#!/usr/bin/env bash
# Runs eval/run_session_eval.py (iterative-recall cases) against the same
# dedicated 'membase_eval' database as scripts/run-eval.sh.
set -euo pipefail

: "${MEMBASE_ROOT:?run inside 'nix develop'}"

export PGDATABASE=membase_eval
export DATABASE_URL="postgresql+psycopg://$PGUSER@/$PGDATABASE?host=$PGHOST_DIR&port=$PGPORT"

scripts/dev-db.sh start
(cd backend && alembic upgrade head)
python3 eval/run_session_eval.py
