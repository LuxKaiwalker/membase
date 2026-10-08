#!/usr/bin/env bash
# Diagnoses the local Ollama setup membase depends on: reachability,
# whether the configured models are present (pulling if not), and that
# generation + embeddings actually work end-to-end. Safe to rerun any
# time -- every step is idempotent and purely read-only against Ollama
# except pulling a genuinely missing model.
set -uo pipefail

: "${MEMBASE_ROOT:?run inside 'nix develop'}"
HOST="${OLLAMA_HOST:-http://127.0.0.1:11434}"
FAILED=0

pass() { echo "  OK   $1"; }
fail() { echo "  FAIL $1"; FAILED=1; }

echo "membase model doctor"
echo "====================="

echo "1. Ollama reachable at $HOST ?"
tags_json=$(curl -sf "$HOST/api/tags") || { fail "cannot reach $HOST -- is the Ollama service running? (systemctl status ollama)"; exit 1; }
pass "reachable"

read -r GEN_MODEL EMBED_MODEL EMBED_DIM <<< "$(cd "$MEMBASE_ROOT/backend" && python3 -c '
from app.config import get_settings
s = get_settings()
print(s.membase_generation_model, s.membase_embed_model, s.membase_embed_dim)
')"

echo "2. Configured models (from app config)"
echo "   generation: $GEN_MODEL"
echo "   embedding:  $EMBED_MODEL (expected dim $EMBED_DIM)"

installed=$(echo "$tags_json" | python3 -c "
import json, sys
print(' '.join(m['name'] for m in json.load(sys.stdin)['models']))
")

echo "3. Models present locally?"
for model in "$GEN_MODEL" "$EMBED_MODEL"; do
  # `ollama list` suffixes untagged models with ":latest"; match either form.
  if echo "$installed" | tr ' ' '\n' | sed 's/:latest$//' | grep -qx "${model%:latest}"; then
    pass "$model already pulled"
  else
    echo "  ... pulling $model (not found locally)"
    if ollama pull "$model"; then
      pass "$model pulled"
    else
      fail "could not pull $model"
    fi
  fi
done

echo "4. Generation works?"
# /api/chat, not /api/generate -- that's what OllamaProvider actually
# calls (see app/llm/ollama_provider.py; /api/generate's flat
# system+prompt fields were found to silently truncate array-typed
# JSON-schema-constrained output for gemma4, see docs/DECISIONS.md), so
# this smoke test exercises the same endpoint the app depends on rather
# than a different one that happens to also exist.
resp=$(curl -sf "$HOST/api/chat" -d "{\"model\":\"$GEN_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Say OK.\"}],\"stream\":false}")
if echo "$resp" | python3 -c "import json,sys; d=json.load(sys.stdin); exit(0 if d.get('message', {}).get('content') else 1)" 2>/dev/null; then
  pass "generation returned a response"
else
  fail "generation call failed or returned no 'message.content' field"
fi

echo "5. Embeddings work and match configured dimension?"
resp=$(curl -sf "$HOST/api/embed" -d "{\"model\":\"$EMBED_MODEL\",\"input\":\"hello\"}")
actual_dim=$(echo "$resp" | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
    print(len(d['embeddings'][0]))
except Exception:
    print(-1)
" 2>/dev/null)
actual_dim="${actual_dim:--1}"
if [ "$actual_dim" = "$EMBED_DIM" ]; then
  pass "embedding dim $actual_dim matches configured $EMBED_DIM"
elif [ "$actual_dim" = "-1" ]; then
  fail "embedding call failed"
else
  fail "embedding dim $actual_dim does NOT match configured $EMBED_DIM -- changing MEMBASE_EMBED_MODEL to a model with a different dimension requires a schema migration (the 'embedding' column is a fixed-size pgvector(N)), not just a config change. See docs/DECISIONS.md."
fi

echo "====================="
if [ "$FAILED" = "0" ]; then
  echo "All checks passed."
  exit 0
else
  echo "Some checks failed -- see FAIL lines above."
  exit 1
fi
