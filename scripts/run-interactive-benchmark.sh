#!/usr/bin/env bash
# Runs eval/run_interactive_benchmark.py (A/B/C mode comparison on the
# synthetic ambiguous-world corpus) against the same dedicated
# 'membase_eval' database as the other eval scripts.
set -euo pipefail

: "${MEMBASE_ROOT:?run inside 'nix develop'}"

export PGDATABASE=membase_eval
export DATABASE_URL="postgresql+psycopg://$PGUSER@/$PGDATABASE?host=$PGHOST_DIR&port=$PGPORT"

scripts/dev-db.sh start
(cd backend && alembic upgrade head)
python3 eval/run_interactive_benchmark.py
