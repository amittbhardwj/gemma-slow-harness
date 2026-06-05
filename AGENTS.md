# AGENTS.md

Instructions for local AI coding agents working in this repository.

## Project
- Mac-first Python CLI harness that turns small local 4-bit models into safer repo-aware coding assistants.
- The project should stay focused on practical local developer workflows: runtime profiles, repo context, safe tools, and measurable slow-compute improvements.
- Avoid drifting into a generic agent framework. The niche is reliable local coding work on consumer Macs.
- Keep the harness dependency-light and offline-testable.
- Prefer stdlib, SQLite, pathlib, subprocess, and pytest over heavy services or frameworks.

## Commands
- Install: `python3 -m venv .venv && source .venv/bin/activate && pip install -e . pytest`
- Test: `python3 -m pytest -q`
- CLI smoke: `gemma-harness --help`

## Coding Rules
- Keep changes minimal, focused, and easy to inspect.
- Do not require live Ollama/Gemma for unit tests.
- Do not add Docker, vector DB services, browser automation, or large dependencies without explicit need.
- Keep model/tool actions conservative: write, shell, patch apply, and macOS control must stay gated or dry-run by default.

## Verification
- Run the smallest relevant test first.
- Run the full offline pytest suite before handing back changes.
- For CLI changes, run the affected `gemma-harness ... --help` smoke command.
