#!/usr/bin/env bash
set -euo pipefail

if ! command -v brew >/dev/null 2>&1; then
  echo "Homebrew not found. Install Homebrew first: https://brew.sh"
  exit 1
fi

if ! command -v ollama >/dev/null 2>&1; then
  brew install --cask ollama
fi

echo "Start Ollama manually if it is not running: ollama serve"
echo "Pulling Gemma 4 12B..."
ollama pull gemma4:12b

echo "Done. Run: gemma-harness doctor"
