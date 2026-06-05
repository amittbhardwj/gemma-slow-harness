#!/usr/bin/env bash
set -euo pipefail

# Heavier reasoning profile. Close Chrome/VS Code/Docker first on 16 GB Macs.
export GEMMA_NUM_CTX=8192
export GEMMA_MAX_TOKENS=1536
export GEMMA_CANDIDATES=4
export GEMMA_DEBATE_ROUNDS=2
export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_KV_CACHE_TYPE=q8_0
export OLLAMA_CONTEXT_LENGTH=8192

ollama serve
