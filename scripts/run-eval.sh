#!/usr/bin/env bash
# Runs eval/run_eval.py against a dedicated 'membase_eval' database so it
# never touches real stored memories in the main dev database.
set -euo pipefail

: "${MEMBASE_ROOT:?run inside 'nix develop'}"

export PGDATABASE=membase_eval
export DATABASE_URL="postgresql+psycopg://$PGUSER@/$PGDATABASE?host=$PGHOST_DIR&port=$PGPORT"

scripts/dev-db.sh start
(cd backend && alembic upgrade head)
python3 eval/run_eval.py
