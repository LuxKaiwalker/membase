#!/usr/bin/env bash
# Runs eval/threshold_sweep.py against the dedicated 'membase_eval'
# database (same isolation as scripts/run-eval.sh).
set -euo pipefail

: "${MEMBASE_ROOT:?run inside 'nix develop'}"

export PGDATABASE=membase_eval
export DATABASE_URL="postgresql+psycopg://$PGUSER@/$PGDATABASE?host=$PGHOST_DIR&port=$PGPORT"

"$MEMBASE_ROOT/scripts/dev-db.sh" start
(cd "$MEMBASE_ROOT/backend" && alembic upgrade head)
python3 "$MEMBASE_ROOT/eval/threshold_sweep.py"
