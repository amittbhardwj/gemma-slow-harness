#!/usr/bin/env bash
set -euo pipefail

MODEL="${GEMMA_MODEL:-gemma4:12b}"
PROFILE="${GEMMA_PROFILE:-gemma4-fast}"
BASE_URL="${GEMMA_BASE_URL:-http://localhost:11434}"
HARNESS="${GEMMA_HARNESS_BIN:-gemma-harness}"
OLLAMA_BIN="${OLLAMA_BIN:-ollama}"
STARTED_OLLAMA_PID=""

if ! command -v "$OLLAMA_BIN" >/dev/null 2>&1; then
  if [ -x /Applications/Ollama.app/Contents/Resources/ollama ]; then
    OLLAMA_BIN=/Applications/Ollama.app/Contents/Resources/ollama
  else
    echo "ollama not found. Install Ollama or set OLLAMA_BIN=/path/to/ollama." >&2
    exit 127
  fi
fi

cleanup() {
  if [ -n "$STARTED_OLLAMA_PID" ]; then
    kill "$STARTED_OLLAMA_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

if ! curl -fsS "$BASE_URL/api/tags" >/dev/null 2>&1; then
  echo "Starting Ollama on $BASE_URL..."
  OLLAMA_FLASH_ATTENTION="${OLLAMA_FLASH_ATTENTION:-1}" \
  OLLAMA_KV_CACHE_TYPE="${OLLAMA_KV_CACHE_TYPE:-q8_0}" \
  OLLAMA_CONTEXT_LENGTH="${OLLAMA_CONTEXT_LENGTH:-4096}" \
    "$OLLAMA_BIN" serve >/tmp/gemma-harness-ollama-smoke.log 2>&1 &
  STARTED_OLLAMA_PID=$!
  for _ in $(seq 1 30); do
    curl -fsS "$BASE_URL/api/tags" >/dev/null 2>&1 && break
    sleep 1
  done
fi

curl -fsS "$BASE_URL/api/tags" >/dev/null

if ! "$OLLAMA_BIN" show "$MODEL" >/dev/null 2>&1; then
  echo "Pulling $MODEL..."
  "$OLLAMA_BIN" pull "$MODEL"
fi

echo "Ollama model:"
"$OLLAMA_BIN" show "$MODEL" | sed -n '1,24p'

echo
echo "Harness doctor:"
env \
  GEMMA_PROFILE="$PROFILE" \
  GEMMA_PROVIDER=ollama \
  GEMMA_MODEL="$MODEL" \
  GEMMA_BASE_URL="$BASE_URL" \
  GEMMA_MAX_TOKENS="${GEMMA_MAX_TOKENS:-256}" \
  "$HARNESS" doctor

echo
echo "Harness ask:"
env \
  GEMMA_PROFILE="$PROFILE" \
  GEMMA_PROVIDER=ollama \
  GEMMA_MODEL="$MODEL" \
  GEMMA_BASE_URL="$BASE_URL" \
  GEMMA_MAX_TOKENS="${GEMMA_MAX_TOKENS:-128}" \
  "$HARNESS" ask "Reply in exactly five words: Gemma smoke test passed."
