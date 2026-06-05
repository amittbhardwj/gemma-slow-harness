#!/usr/bin/env bash
set -euo pipefail

# Lowest-memory fallback. q4 KV may reduce quality; use only under memory pressure.
export GEMMA_NUM_CTX=4096
export GEMMA_MAX_TOKENS=768
export GEMMA_CANDIDATES=2
export GEMMA_DEBATE_ROUNDS=1
export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_KV_CACHE_TYPE=q4_0
export OLLAMA_CONTEXT_LENGTH=4096

ollama serve
