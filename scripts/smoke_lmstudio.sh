#!/usr/bin/env bash
set -euo pipefail

LMS="${LMS_BIN:-$HOME/.lmstudio/bin/lms}"
MODEL_KEY="${LMSTUDIO_MODEL_KEY:-}"
MODEL_ID="${LMSTUDIO_MODEL_ID:-lmstudio-smoke}"
PORT="${LMSTUDIO_PORT:-1234}"
BASE_URL="${GEMMA_BASE_URL:-http://127.0.0.1:$PORT/v1}"
HARNESS="${GEMMA_HARNESS_BIN:-gemma-harness}"
STARTED_SERVER=0

if [ ! -x "$LMS" ]; then
  echo "LM Studio CLI not found at $LMS. Install LM Studio or set LMS_BIN." >&2
  exit 127
fi

if [ -z "$MODEL_KEY" ]; then
  MODEL_KEY="$("$LMS" ls --llm --json | python3 -c 'import json,sys; data=json.load(sys.stdin); print(data[0]["modelKey"] if data else "")')"
fi

if [ -z "$MODEL_KEY" ]; then
  echo "No LM Studio LLM models are installed. Download a model in LM Studio, or set LMSTUDIO_MODEL_KEY." >&2
  exit 2
fi

SERVER_STATUS="$("$LMS" server status 2>&1 || true)"
if echo "$SERVER_STATUS" | grep -qi "not running"; then
  echo "Starting LM Studio server on port $PORT..."
  "$LMS" server start --port "$PORT" --bind 127.0.0.1 >/tmp/gemma-harness-lmstudio-server.log 2>&1
  STARTED_SERVER=1
fi

if ! "$LMS" ps | grep -q "$MODEL_ID"; then
  echo "Loading LM Studio model $MODEL_KEY as $MODEL_ID..."
  "$LMS" load "$MODEL_KEY" --identifier "$MODEL_ID" --context-length "${LMSTUDIO_CONTEXT_LENGTH:-4096}" --ttl "${LMSTUDIO_TTL:-120}" -y
fi

echo "LM Studio loaded models:"
"$LMS" ps

echo
echo "Harness doctor:"
env \
  GEMMA_PROVIDER=lmstudio \
  GEMMA_BASE_URL="$BASE_URL" \
  GEMMA_MODEL="$MODEL_ID" \
  GEMMA_API_KEY="${GEMMA_API_KEY:-local}" \
  GEMMA_MAX_TOKENS="${GEMMA_MAX_TOKENS:-128}" \
  GEMMA_NUM_CTX="${GEMMA_NUM_CTX:-4096}" \
  GEMMA_CANDIDATES="${GEMMA_CANDIDATES:-1}" \
  GEMMA_DEBATE_ROUNDS="${GEMMA_DEBATE_ROUNDS:-0}" \
  "$HARNESS" doctor

echo
echo "Harness ask:"
env \
  GEMMA_PROVIDER=lmstudio \
  GEMMA_BASE_URL="$BASE_URL" \
  GEMMA_MODEL="$MODEL_ID" \
  GEMMA_API_KEY="${GEMMA_API_KEY:-local}" \
  GEMMA_MAX_TOKENS="${GEMMA_MAX_TOKENS:-128}" \
  GEMMA_NUM_CTX="${GEMMA_NUM_CTX:-4096}" \
  GEMMA_CANDIDATES="${GEMMA_CANDIDATES:-1}" \
  GEMMA_DEBATE_ROUNDS="${GEMMA_DEBATE_ROUNDS:-0}" \
  "$HARNESS" ask "Reply in exactly five words: LM Studio smoke test passed."

if [ "$STARTED_SERVER" = "1" ]; then
  echo
  echo "LM Studio server was started for this smoke test and is still running."
fi
