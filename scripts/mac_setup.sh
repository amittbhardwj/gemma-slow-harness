#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e .

echo "Installed gemma-slow-harness."
echo "Next: start Ollama and run: gemma-harness doctor"
