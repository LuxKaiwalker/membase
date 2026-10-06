#!/usr/bin/env bash
# Runs eval/threshold_sweep.py against the dedicated 'membase_eval'
# database (same isolation as scripts/run-eval.sh).
set -euo pipefail

: "${MEMBASE_ROOT:?run inside 'nix develop'}"

export PGDATABASE=membase_eval
export DATABASE_URL="postgresql+psycopg://$PGUSER@/$PGDATABASE?host=$PGHOST_DIR&port=$PGPORT"

scripts/dev-db.sh start
(cd backend && alembic upgrade head)
python3 eval/threshold_sweep.py
