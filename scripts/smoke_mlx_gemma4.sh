#!/usr/bin/env bash
set -euo pipefail

MODEL="${MLX_MODEL:-mlx-community/gemma-4-12B-it-4bit}"
MODEL_ID="${MLX_MODEL_ID:-$MODEL}"
PORT="${MLX_PORT:-8080}"
BASE_URL="${GEMMA_BASE_URL:-http://127.0.0.1:$PORT/v1}"
HARNESS="${GEMMA_HARNESS_BIN:-gemma-harness}"
SERVER_PID=""

if ! command -v mlx_lm.server >/dev/null 2>&1; then
  echo "mlx_lm.server not found. Install MLX-LM first: pip install mlx-lm" >&2
  exit 127
fi

cleanup() {
  if [ -n "$SERVER_PID" ]; then
    kill "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

if ! curl -fsS "$BASE_URL/models" >/dev/null 2>&1; then
  echo "Starting MLX-LM server for $MODEL on port $PORT..."
  mlx_lm.server \
    --model "$MODEL" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --max-tokens "${MLX_MAX_TOKENS:-256}" \
    --chat-template-args '{"enable_thinking":false}' \
    >/tmp/gemma-harness-mlx-smoke.log 2>&1 &
  SERVER_PID=$!
  for _ in $(seq 1 120); do
    curl -fsS "$BASE_URL/models" >/dev/null 2>&1 && break
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "MLX-LM server exited before becoming ready. Log:" >&2
      sed -n '1,160p' /tmp/gemma-harness-mlx-smoke.log >&2 || true
      exit 1
    fi
    sleep 1
  done
fi

curl -fsS "$BASE_URL/models" >/dev/null

echo "MLX-LM models endpoint:"
curl -fsS "$BASE_URL/models" | python3 -m json.tool | sed -n '1,80p'

echo
echo "Harness doctor:"
env \
  GEMMA_PROVIDER=mlx \
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
  GEMMA_PROVIDER=mlx \
  GEMMA_BASE_URL="$BASE_URL" \
  GEMMA_MODEL="$MODEL_ID" \
  GEMMA_API_KEY="${GEMMA_API_KEY:-local}" \
  GEMMA_MAX_TOKENS="${GEMMA_MAX_TOKENS:-128}" \
  GEMMA_NUM_CTX="${GEMMA_NUM_CTX:-4096}" \
  GEMMA_CANDIDATES="${GEMMA_CANDIDATES:-1}" \
  GEMMA_DEBATE_ROUNDS="${GEMMA_DEBATE_ROUNDS:-0}" \
  "$HARNESS" ask "Reply in exactly five words: MLX smoke test passed."
