#!/usr/bin/env bash
set -euo pipefail

# Best daily profile for Mac mini M4 base 16 GB.
export GEMMA_NUM_CTX=4096
export GEMMA_MAX_TOKENS=1024
export GEMMA_CANDIDATES=3
export GEMMA_DEBATE_ROUNDS=1
export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_KV_CACHE_TYPE=q8_0
export OLLAMA_CONTEXT_LENGTH=4096

ollama serve
