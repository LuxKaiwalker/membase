#!/usr/bin/env bash
# Runs eval/strategy_benchmark.py against a dedicated 'membase_eval' database
# so it never touches real stored memories in the main dev database.
#
# Arguments are passed straight through, e.g.:
#   scripts/run-strategy-benchmark.sh --self-check
#   scripts/run-strategy-benchmark.sh --limit 8
#   scripts/run-strategy-benchmark.sh --repeat 3
#   scripts/run-strategy-benchmark.sh --reseed
set -euo pipefail

: "${MEMBASE_ROOT:?run inside 'nix develop'}"

export PGDATABASE=membase_eval
export DATABASE_URL="postgresql+psycopg://$PGUSER@/$PGDATABASE?host=$PGHOST_DIR&port=$PGPORT"

"$MEMBASE_ROOT/scripts/dev-db.sh" start
(cd "$MEMBASE_ROOT/backend" && alembic upgrade head)
python3 "$MEMBASE_ROOT/eval/strategy_benchmark.py" "$@"
