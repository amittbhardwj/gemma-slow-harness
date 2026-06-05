# Gemma Slow Harness

A lightweight **local agent harness** designed to make **Gemma 4 12B 4-bit** more useful on a **Mac mini M4 base model with 16 GB unified memory**.

The goal is not to make a 12B model magically equal a frontier model. The goal is to use **slow test-time compute**:

- plan first
- generate multiple candidate answers
- critique them
- refine them
- judge the final answer
- optionally retrieve local workspace context with RAG
- optionally use local tools for files and Python execution

This makes a small local model much stronger on practical tasks like coding, ML scripts, document Q&A, project planning, and cost-modeling workflows.

---

## Recommended Mac mini M4 16 GB settings

Use the built-in memory profiles instead of manually guessing context/KV settings.

### Gemma 4 fast profile

For quick Gemma 4 12B 4-bit smoke tests and low-latency local runs:

```bash
gemma-harness --profile gemma4-fast doctor
gemma-harness --profile gemma4-fast ask "Summarize this repository in one sentence."
```

Equivalent key defaults:

```bash
export GEMMA_PROFILE=gemma4-fast
export GEMMA_MODEL=gemma4:12b
export GEMMA_NUM_CTX=4096
export GEMMA_MAX_TOKENS=512
export GEMMA_CANDIDATES=1
export GEMMA_DEBATE_ROUNDS=0
export GEMMA_OLLAMA_THINK=false
```

Explicit environment variables still override the profile, so you can do:

```bash
GEMMA_PROFILE=gemma4-fast GEMMA_MAX_TOKENS=128 gemma-harness doctor
```

### Safe daily profile

```bash
gemma-harness memory profile safe --exports
```

Equivalent settings:

```bash
export GEMMA_PROVIDER=ollama
export GEMMA_MODEL=gemma4:12b
export GEMMA_BASE_URL=http://localhost:11434
export GEMMA_NUM_CTX=4096
export GEMMA_MAX_TOKENS=1024
export GEMMA_CANDIDATES=3
export GEMMA_DEBATE_ROUNDS=1
export GEMMA_MIN_SCORE=80

# Must be set before `ollama serve` starts:
export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_KV_CACHE_TYPE=q8_0
export OLLAMA_CONTEXT_LENGTH=4096
```

### Strong profile

Use only after closing memory-heavy apps:

```bash
gemma-harness memory profile strong --exports
```

### Emergency low-memory profile

Use this if Activity Monitor shows high memory pressure or swap growth:

```bash
gemma-harness memory profile emergency --exports
```

Avoid jumping to 16K/32K context on a 16 GB machine. The model weights fit, but long context/KV cache consumes memory quickly.

---

## Install

### 1. Install Ollama on macOS

```bash
brew install --cask ollama
```

Start Ollama:

```bash
ollama serve
```

In another terminal:

```bash
ollama pull gemma4:12b
```

### 2. Install this harness

From this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e .
```

### 3. Check connection

```bash
gemma-harness doctor
```

Expected result: it should print `OK` or a similar short response from the local model.

---

## Memory, KV cache, and cold storage policy

This harness **does not try to store active KV cache on SSD**. Active KV cache is hot attention state and should stay in RAM. On a 16 GB Apple Silicon Mac, SSD-backed active KV usually causes severe slowdown or swap thrashing.

Instead, the harness uses:

- **q8/q4 KV-cache policy** through Ollama environment variables
- **strict 4K/8K context profiles**
- **automatic active-context compaction** before model calls
- **cold storage** for intelligence: SQLite RAG, learning runs, skills, and summaries

Useful commands:

```bash
# List profiles
gemma-harness memory profiles

# Show safe/strong/emergency exports
gemma-harness memory profile safe --exports
gemma-harness memory profile strong --exports
gemma-harness memory profile emergency --exports

# Estimate KV cache and total memory footprint
gemma-harness memory estimate
gemma-harness memory estimate --ctx 8192 --kv q8_0
gemma-harness memory estimate --ctx 8192 --kv q4_0

# Show current Ollama/harness memory env
gemma-harness memory env

# macOS memory snapshot
gemma-harness memory mac
```

To start Ollama with the safe profile:

```bash
bash scripts/ollama_memory_safe.sh
```

Or use the generated exports manually:

```bash
gemma-harness memory profile safe --exports
# copy the exports, then restart ollama serve
```

Important: `OLLAMA_FLASH_ATTENTION`, `OLLAMA_KV_CACHE_TYPE`, and `OLLAMA_CONTEXT_LENGTH` must be set **before** `ollama serve` starts. Restart Ollama after changing them.

---

## Usage

### Single-pass response

```bash
gemma-harness ask "Explain Random Forest for casting cost prediction."
```

### Slow multi-pass response

```bash
gemma-harness deep "Create a machine learning plan to predict casting cost from weight, material, vendor, OD, ID, wall thickness, and standard cost." --show-trace
```

### Use a prompt file

```bash
gemma-harness deep --file examples/casting_cost_prompt.md --show-trace
```

---

## RAG over your local project

Index a workspace:

```bash
export GEMMA_WORKSPACE=/path/to/your/project
gemma-harness index
```

Search indexed context:

```bash
gemma-harness search "Flask create_app route error"
```

Ask with RAG enabled:

```bash
gemma-harness deep "Find why my Flask app import is failing and propose a fix." --show-trace
```

The RAG implementation uses SQLite FTS5 instead of embeddings. This keeps memory low and avoids loading another model on a 16 GB Mac.


---

## Self-learning loop and saved skills

This harness now has persistent retrieval memory in:

```bash
.gemma_harness/learning.sqlite3
```

This is **not model fine-tuning**. It is also not active KV-cache offload. It is **cold memory** stored on SSD and retrieved into a small active context. This is safer and lighter for a 16 GB Mac:

1. Every `deep` run can save the task, plan, final answer, judge score, critiques, and distilled lessons.
2. Future `deep` runs retrieve similar lessons and inject them as context.
3. Repeated workflows can be promoted to named **skills**.
4. Skills are retrieved automatically when the task looks similar, or forced manually with `--skill`.

### Self-learning settings

```bash
export GEMMA_LEARNING_ENABLED=true
export GEMMA_LEARNING_DB=.gemma_harness/learning.sqlite3
export GEMMA_MEMORY_TOP_K=5
export GEMMA_SKILL_TOP_K=3
export GEMMA_AUTO_LEARN_MIN_SCORE=80
```

Disable learning for one run:

```bash
gemma-harness deep "your task" --no-learn
```

Inspect learning memory:

```bash
gemma-harness learn stats
gemma-harness learn list --limit 10
gemma-harness learn search "flask import error"
gemma-harness learn show 1
```

### Save common workflows as skills

Create a skill manually:

```bash
gemma-harness skill save casting-cost-ml \
  --description "ML workflow for casting cost prediction" \
  --tags "ml,casting,cost" \
  --trigger "Use for casting cost prediction from Excel/tabular data" \
  --workflow-file examples/skills/casting_cost_ml_workflow.md
```

List/search/show skills:

```bash
gemma-harness skill list
gemma-harness skill search "casting cost"
gemma-harness skill show casting-cost-ml
```

Force a skill into a run:

```bash
gemma-harness deep "Build a model for my castings Excel file" --skill casting-cost-ml --show-trace
```

Promote a good run into a reusable skill:

```bash
gemma-harness deep "Create a robust casting cost ML pipeline" \
  --save-skill casting-cost-ml \
  --skill-tags "ml,casting,cost" \
  --show-trace
```

Or promote an older saved run:

```bash
gemma-harness skill from-run 3 casting-cost-ml --tags "ml,casting,cost"
```

### How the loop works

```text
New task
  ↓
Retrieve matching skills
  ↓
Retrieve similar past runs and lessons
  ↓
Plan → candidates → critics → refiners → judge
  ↓
Repair if score is low
  ↓
Extract reusable lessons
  ↓
Save run to learning.sqlite3
  ↓
Optionally promote the run to a skill
```

Good skills to create first:

- `casting-cost-ml`
- `flask-debugging`
- `vue-vitest-debugging`
- `python-package-imports`
- `ocr-drawing-extraction`
- `github-readme-generator`

---

## OpenAI-compatible servers

For LM Studio, llama-server, LiteRT-LM, vLLM, SGLang, or any OpenAI-compatible local server:

```bash
export GEMMA_PROVIDER=openai
export GEMMA_BASE_URL=http://localhost:1234/v1
export GEMMA_MODEL=gemma-4-12b-it
export GEMMA_API_KEY=local
```

Then:

```bash
gemma-harness doctor
```

### LM Studio smoke test

The LM Studio script starts the local server if needed, loads the first installed LLM unless `LMSTUDIO_MODEL_KEY` is set, and checks the harness through LM Studio's OpenAI-compatible API:

```bash
bash scripts/smoke_lmstudio.sh
```

To target a specific downloaded model:

```bash
LMSTUDIO_MODEL_KEY=your-model-key bash scripts/smoke_lmstudio.sh
```

### MLX-LM smoke test

If MLX-LM is installed, this starts an OpenAI-compatible MLX server for the cached or downloadable MLX Gemma 4 12B 4-bit model:

```bash
bash scripts/smoke_mlx_gemma4.sh
```

Defaults:

```bash
export MLX_MODEL=mlx-community/gemma-4-12B-it-4bit
export MLX_PORT=8080
```

### Ollama Gemma 4 smoke test

For a repeatable end-to-end Ollama check:

```bash
bash scripts/smoke_ollama_gemma4.sh
```

The script starts Ollama if needed, pulls `gemma4:12b` if missing, runs `doctor`, and then runs a short `ask`.

---

## Safety defaults

By default:

- file reading is allowed inside `GEMMA_WORKSPACE`
- file search is allowed inside `GEMMA_WORKSPACE`
- Python snippets can run inside `GEMMA_WORKSPACE`
- file writing is disabled
- shell execution is disabled

Enable writes only when you trust the task:

```bash
export GEMMA_ALLOW_WRITE=true
```

Enable shell only when you really need it:

```bash
export GEMMA_ALLOW_SHELL=true
```

---

## Design

```text
User task
  ↓
Planner
  ↓
RAG context retrieval
  ↓
Candidate answer 1..N
  ↓
Critic loop
  ↓
Refiner loop
  ↓
Synthesis
  ↓
Judge score
  ↓
Repair pass if score is low
  ↓
Final answer
```

---

## Why this helps Gemma 4 12B

A small model loses quality when it has to do everything in one shot. This harness improves output quality by giving it structure:

- multiple attempts reduce random failure
- critics catch hallucinations and weak assumptions
- judge pass creates a quality gate
- local RAG gives project-specific knowledge
- Python execution verifies calculations and ML logic
- conservative Mac settings prevent memory pressure

---

## Suggested modes

### Fast mode

```bash
export GEMMA_CANDIDATES=2
export GEMMA_DEBATE_ROUNDS=1
gemma-harness deep "your task"
```

### Strong mode

```bash
export GEMMA_CANDIDATES=6
export GEMMA_DEBATE_ROUNDS=3
export GEMMA_MAX_TOKENS=2048
gemma-harness deep "your task" --show-trace
```

### Mac mini safe mode

```bash
export GEMMA_NUM_CTX=8192
export GEMMA_MAX_TOKENS=1024
export GEMMA_CANDIDATES=3
export GEMMA_DEBATE_ROUNDS=1
```

---

## Notes for casting-cost ML work

Use prompts that force the model to verify rather than guess:

```text
You are building a model to predict casting cost.
Columns: part number, weight, vendor, material, factory standard cost, OD, ID, wall thickness.
Create a Python pipeline that tests Linear Regression, Ridge, Random Forest, ExtraTrees, XGBoost/CatBoost if available.
Use train/test split and cross-validation.
Report R2, MAE, RMSE.
Check leakage.
Explain feature importance.
```

Put your dataset description, data dictionary, and prior mistakes into markdown files, index the folder, and ask with `gemma-harness deep`.

---

## Project structure

```text
harness/
  agent.py      # multi-pass planner/candidate/critic/refiner/judge loop
  cli.py        # command-line interface
  config.py     # environment-based config
  llm.py        # Ollama and OpenAI-compatible clients
  prompts.py    # role prompts
  rag.py        # lightweight SQLite FTS5 RAG
  tools.py      # local tool registry
  utils.py      # file/path helpers
examples/
  casting_cost_prompt.md
  repo_debug_prompt.md
scripts/
  mac_setup.sh
  ollama_gemma4_setup.sh
```

---

## macOS desktop automation

The harness now includes an optional macOS automation layer. This lets it use macOS-native tools to observe and control the desktop:

- open or activate apps
- take screenshots with `screencapture`
- click screen coordinates through Accessibility UI scripting
- type short text
- send hotkeys such as `command,space`
- run explicit AppleScript snippets
- ask the model to propose a small JSON action plan with `gemma-harness act`

This is disabled by default because it can control your mouse/keyboard and can expose screen content.

### Permissions

Show the setup checklist:

```bash
gemma-harness mac permissions
```

You will usually need:

1. System Settings → Privacy & Security → Accessibility
2. System Settings → Privacy & Security → Screen Recording
3. Restart your terminal after enabling permissions

### Enable automation

```bash
export GEMMA_ALLOW_MAC_CONTROL=true
export GEMMA_ALLOW_MAC_SCREENSHOT=true
```

### Direct macOS actions

By default, click/type/hotkey/open/activate are dry-run unless you pass `--yes`.

```bash
# Observation
gemma-harness mac screenshot .gemma_harness/desktop.png

# Dry-run by default
gemma-harness mac open Safari

# Actually open Safari
gemma-harness mac open Safari --yes

# Spotlight hotkey
gemma-harness mac hotkey command,space --yes

# Click coordinates
gemma-harness mac click 500 400 --yes

# Type short text into the active app
gemma-harness mac type "hello from Gemma" --yes
```

### Model-planned actions

`act` asks the local model to produce a short JSON action plan. It does not execute by default.

```bash
gemma-harness act "Open Notes and create a note titled Casting ML ideas"
```

To execute the proposed plan:

```bash
gemma-harness act "Open Notes and create a note titled Casting ML ideas" --execute --yes
```

Safety notes:

- Prefer app opening, hotkeys, and AppleScript over blind coordinate clicking.
- Never use desktop automation to type passwords, payment information, government IDs, or other secrets.
- Coordinate clicking is brittle across monitor layouts, app windows, and scaling settings.
- Screenshots may contain private information; keep them inside `.gemma_harness/` or another private workspace path.

---

## Comparison with Codex-style harnesses

This harness is local-first and optimized for a small Mac running Gemma 4 12B. Codex-style tools are usually stronger for software engineering because they have a mature interactive terminal workflow, repository editing loop, command execution, patch review, and cloud/local integration.

What this harness has:

- local model support through Ollama/OpenAI-compatible servers
- multi-candidate slow reasoning
- local RAG over a workspace
- persistent learning/skills memory
- memory/KV policy profiles for 16 GB Macs
- optional macOS desktop automation
- conservative local tools

What is still missing compared with a mature Codex-style coding harness:

- robust native patch application and diff review UI
- automatic test-fix-test development loop with checkpoints
- Git-aware branch/PR workflow
- built-in sandboxing/container isolation
- reliable structured tool-call loop for every agent step
- IDE integration
- multimodal screen understanding beyond raw screenshots
- strong cloud execution option for heavy tasks
- first-class AGENTS.md / repo instruction handling
- permission model as polished as mature coding agents


---

## Codex-like developer-agent layer

This version adds a local developer workflow layer on top of the slow Gemma harness.
It is still lighter than Codex, but it now has the missing basics:

- `AGENTS.md` / repo instruction loading
- git status and diff inspection
- patch checkpoints before risky changes
- unified-diff validation and application
- structured JSON tool-call loop
- basic test-fix-test coding loop

### AGENTS.md support

Create a starter repo instruction file:

```bash
gemma-harness repo init-agents
```

Show loaded instructions:

```bash
gemma-harness repo instructions
```

The harness automatically injects AGENTS-style instructions into `deep`, `loop`, and `code` runs. It looks for:

```text
AGENTS.md
GEMMA.md
CLAUDE.md
.gemma_harness/INSTRUCTIONS.md
nested AGENTS.md files in shallow monorepo folders
```

### Git helpers

```bash
# Show branch and short status
gemma-harness repo status

# Show full diff
gemma-harness repo diff

# Compact diff stat
gemma-harness repo diff-stat

# Save a safe patch checkpoint into .gemma_harness/checkpoints/
gemma-harness repo checkpoint --label before-refactor

# Optional real git commit checkpoint
gemma-harness repo checkpoint --commit --message "Gemma checkpoint before refactor"
```

The default checkpoint is a patch file, not a commit. This is safer on your own machine because it does not alter history.

### Patch review and application

Check a model-generated patch:

```bash
gemma-harness patch check --file proposed.patch
```

Apply only after review:

```bash
gemma-harness patch apply --file proposed.patch --yes
```

Without `--yes`, `patch apply` only performs a dry-run check.

### Structured tool-call loop

`loop` gives the model a bounded JSON tool-call loop for repo work:

```bash
gemma-harness loop "Inspect this repo and tell me why the Flask import is failing" --show-trace
```

Use a smaller allowlist when needed:

```bash
gemma-harness loop "Find failing tests" \
  --allowed-tools read_file,list_files,search_files,run_python \
  --show-trace
```

Writes and shell commands are still blocked unless explicitly enabled:

```bash
export GEMMA_ALLOW_WRITE=true
export GEMMA_ALLOW_SHELL=true
```

### Test-fix-test coding loop

Dry-run / propose patch only:

```bash
gemma-harness code "Fix the failing Flask import test" --test "pytest -q"
```

Apply patches and rerun tests:

```bash
export GEMMA_ALLOW_WRITE=true
gemma-harness code "Fix the failing Flask import test" --test "pytest -q" --apply
```

The code loop does this:

```text
save patch checkpoint
run test command
ask model for unified diff only
validate patch with git apply --check
apply only if --apply is set
rerun tests
repeat up to --attempts
```

For your Mac mini 16 GB, keep attempts low:

```bash
gemma-harness code "Fix the Vue unit test failure" --test "npm test -- --run" --attempts 2
```

### Current comparison vs Codex after this update

| Capability | This Gemma harness | Codex-style mature harness |
|---|---:|---:|
| Local Gemma/Ollama support | Yes | Usually not the focus |
| AGENTS.md-style repo instructions | Yes | Yes |
| Git status/diff awareness | Yes | Yes |
| Patch/diff validation | Yes | Yes, usually smoother |
| Patch checkpointing | Yes | Yes |
| Test-fix-test loop | Basic | Stronger |
| Structured tool calls | Basic JSON loop | Stronger/native |
| macOS desktop control | Yes, opt-in | Usually not core |
| IDE integration | No | Often yes |
| PR/GitHub workflow | No | Often yes |
| Sandbox/container isolation | No | Often yes |
| Approval UX | CLI flags | More polished |

Remaining gaps:

- no polished interactive diff UI
- no automatic branch/PR creation
- no container sandbox
- no IDE plugin
- no robust visual UI grounding beyond screenshots
- no native function-calling protocol; uses JSON prompts for local-model compatibility
