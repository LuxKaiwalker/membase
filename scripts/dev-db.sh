#!/usr/bin/env bash
# Init/start/stop a project-local Postgres instance for development.
# Deliberately NOT a system service: keeps this project isolated from
# any NixOS-managed Postgres and avoids touching system config.
set -euo pipefail

: "${PGDATA:?run inside 'nix develop'}"
: "${PGHOST_DIR:?run inside 'nix develop'}"
: "${PGPORT:?run inside 'nix develop'}"
: "${PGUSER:?run inside 'nix develop'}"
: "${PGDATABASE:?run inside 'nix develop'}"

cmd="${1:-start}"

init() {
  if [ ! -s "$PGDATA/PG_VERSION" ]; then
    echo "Initializing Postgres data dir at $PGDATA"
    initdb --locale=C.UTF-8 --encoding=UTF8 -U "$PGUSER" -D "$PGDATA" >/dev/null
  fi
}

start() {
  init
  mkdir -p "$PGHOST_DIR"
  if pg_ctl -D "$PGDATA" status >/dev/null 2>&1; then
    echo "Postgres already running."
  else
    pg_ctl -D "$PGDATA" -l "$PGDATA/log.txt" -o "-p $PGPORT -k $PGHOST_DIR -h ''" start
  fi
  # Idempotent: create db + extensions if missing.
  psql -h "$PGHOST_DIR" -p "$PGPORT" -U "$PGUSER" -d postgres -tc \
    "SELECT 1 FROM pg_database WHERE datname = '$PGDATABASE'" | grep -q 1 || \
    createdb -h "$PGHOST_DIR" -p "$PGPORT" -U "$PGUSER" "$PGDATABASE"
  psql -h "$PGHOST_DIR" -p "$PGPORT" -U "$PGUSER" -d "$PGDATABASE" \
    -c "CREATE EXTENSION IF NOT EXISTS vector;" >/dev/null
  echo "Postgres ready on port $PGPORT, db '$PGDATABASE' (pgvector enabled)."
}

stop() {
  pg_ctl -D "$PGDATA" stop -m fast 2>/dev/null || echo "Postgres not running."
}

case "$cmd" in
  start) start ;;
  stop) stop ;;
  restart) stop; start ;;
  *) echo "usage: $0 {start|stop|restart}" >&2; exit 1 ;;
esac
