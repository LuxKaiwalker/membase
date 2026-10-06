#!/usr/bin/env bash
# Pulls the models membase needs from the already-running system Ollama
# service. Generation models are assumed to already be present (see
# README); this script only ensures the embedding model exists.
set -euo pipefail

EMBED_MODEL="${MEMBASE_EMBED_MODEL:-nomic-embed-text}"

echo "Checking Ollama at ${OLLAMA_HOST:-http://127.0.0.1:11434} ..."
curl -sf "${OLLAMA_HOST:-http://127.0.0.1:11434}/api/tags" >/dev/null || {
  echo "Ollama is not reachable. On NixOS it usually runs as a system service:" >&2
  echo "  systemctl status ollama" >&2
  exit 1
}

echo "Pulling embedding model: $EMBED_MODEL"
ollama pull "$EMBED_MODEL"
