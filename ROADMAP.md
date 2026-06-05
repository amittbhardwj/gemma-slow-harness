# Roadmap

Gemma Slow Harness should stay focused on one promise: make small local 4-bit models useful for real repository work on consumer Macs.

## Near term

- Improve repo-context injection for `ask` and `deep` so the model sees the right files by default.
- Add runtime smoke coverage for Ollama, LM Studio, and MLX without requiring those services in unit tests.
- Track run timings and model metadata so slow-compute improvements are measurable.
- Keep Mac memory profiles conservative and easy to reproduce.

## Medium term

- Add a compact benchmark suite comparing one-shot answers with multi-pass harness answers on repo tasks.
- Make the tool loop better at proposing patches while keeping write/shell access gated.
- Improve saved skills so repeated workflows become lightweight local playbooks, not opaque agent state.
- Add clearer failure reports for unsupported model/runtime combinations.

## Non-goals

- Do not become a broad agent framework.
- Do not require cloud APIs, vector databases, Docker, or large service dependencies.
- Do not optimize for leaderboard benchmarks over practical local developer workflows.
- Do not make file writes, shell commands, or desktop control implicit.
