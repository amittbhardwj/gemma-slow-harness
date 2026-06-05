from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .agent import SlowHarnessAgent
from .config import PROFILE_DEFAULTS, HarnessConfig
from .learning import LearningStore
from .llm import LLMError, LocalLLMClient
from .memory_policy import (
    PROFILES,
    current_ollama_env,
    estimate_kv_cache_mb,
    estimate_total_memory_gb,
    format_profile,
    format_profiles,
    get_profile,
    input_context_budget_tokens,
    mac_memory_snapshot,
    shell_exports,
)
from .rag import RagStore
from . import macos_control as mac
from .code_agent import CodeRepairAgent, format_code_run
from .git_ops import create_git_commit_checkpoint, create_patch_checkpoint, diff, diff_stat, is_git_repo, status
from .patching import apply_patch, check_patch, extract_unified_diff
from .repo_context import create_agents_template, load_repo_instructions
from .tool_loop import ToolLoopAgent
from .tools import ToolRegistry
import json


def _read_prompt(args: argparse.Namespace) -> str:
    if getattr(args, "file", None):
        return Path(args.file).read_text(encoding="utf-8", errors="replace")
    if getattr(args, "prompt", None):
        return args.prompt
    if not sys.stdin.isatty():
        return sys.stdin.read()
    raise SystemExit("Provide a prompt argument, --file prompt.md, or pipe text on stdin.")


def _read_text_arg(value: str | None, file_value: str | None, *, field_name: str, required: bool = False) -> str:
    if file_value:
        return Path(file_value).read_text(encoding="utf-8", errors="replace")
    if value:
        return value
    if required:
        raise SystemExit(f"Provide --{field_name} or --{field_name}-file.")
    return ""


def _parse_tags(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [t.strip() for t in raw.split(",") if t.strip()]


def _parse_json_array(raw: str) -> list:
    text = raw.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1 :]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end == -1 or end <= start:
            raise
        data = json.loads(text[start : end + 1])
    if not isinstance(data, list):
        raise ValueError("Action plan must be a JSON array.")
    return data


def cmd_doctor(args: argparse.Namespace) -> int:
    cfg = HarnessConfig.from_env()
    print("Config:")
    print(cfg.summary())
    print("\nTesting model server...")
    try:
        resp = LocalLLMClient(cfg).chat(
            [
                {"role": "system", "content": "You are a health check."},
                {"role": "user", "content": "Reply with exactly: OK"},
            ],
            temperature=0.0,
            max_tokens=128,
        )
        print(f"Model response: {resp.content!r} ({resp.elapsed_sec:.2f}s)")
        if cfg.learning_enabled:
            store = LearningStore(cfg)
            print(f"Learning DB: {store.db_path}")
            print(f"Learning stats: {store.stats()}")
        return 0
    except LLMError as exc:
        print(f"ERROR: {exc}")
        print("\nFor Ollama try:")
        print("  ollama serve")
        print("  ollama pull gemma4:12b")
        print("  export GEMMA_MODEL=gemma4:12b")
        return 1


def cmd_ask(args: argparse.Namespace) -> int:
    cfg = HarnessConfig.from_env()
    prompt = _read_prompt(args)
    client = LocalLLMClient(cfg)
    resp = client.chat(
        [
            {"role": "system", "content": "You are a precise local assistant. Give a useful direct answer."},
            {"role": "user", "content": prompt},
        ]
    )
    print(resp.content)
    return 0


def cmd_deep(args: argparse.Namespace) -> int:
    cfg = HarnessConfig.from_env()
    prompt = _read_prompt(args)
    agent = SlowHarnessAgent(cfg)
    run = agent.run(
        prompt,
        use_rag=not args.no_rag,
        use_learning=not args.no_learn,
        use_skills=not args.no_skills,
        forced_skill=args.skill,
        candidates=args.candidates,
        debate_rounds=args.rounds,
        save_learning=not args.no_learn,
    )
    if args.show_trace:
        print("\n# PLAN\n")
        print(run.plan)
        print("\n# USED SKILLS\n")
        print(", ".join(run.used_skills or []) or "None")
        print("\n# JUDGE\n")
        print(run.judge)
        if run.lessons:
            print("\n# LEARNED LESSONS\n")
            print(run.lessons)
        print("\n# FINAL\n")
    print(run.answer)
    if run.score is not None:
        print(f"\n[Judge score: {run.score}]", file=sys.stderr)
    if run.run_id is not None:
        print(f"[Saved learning run: {run.run_id}]", file=sys.stderr)
    if args.save_skill:
        if run.run_id is None:
            print("Cannot save skill because this run was not saved. Remove --no-learn.", file=sys.stderr)
            return 2
        skill_id = agent.learning.promote_run_to_skill(
            run_id=run.run_id,
            name=args.save_skill,
            description=args.skill_description,
            tags=_parse_tags(args.skill_tags),
        )
        print(f"[Saved skill: {args.save_skill} id={skill_id}]", file=sys.stderr)
    return 0


def cmd_index(args: argparse.Namespace) -> int:
    cfg = HarnessConfig.from_env()
    store = RagStore(cfg)
    root = Path(args.path).expanduser().resolve() if args.path else cfg.workspace
    count = store.index_workspace(root)
    print(f"Indexed {count} chunks into {store.db_path}")
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    cfg = HarnessConfig.from_env()
    store = RagStore(cfg)
    hits = store.search(args.query, top_k=args.top_k)
    for h in hits:
        print(f"\n--- {h.path} chunk={h.chunk_id} score={h.score:.4f} ---")
        print(h.text[: args.max_chars])
    return 0


def _print_run(run) -> None:
    status = "success" if run.success else "needs caution"
    print(f"\n--- run {run.id} | score={run.score} | {status} | {run.created_at} ---")
    print(f"Task: {run.task[:500]}")
    if run.lessons:
        print("\nLessons:")
        print(run.lessons[:1600])


def _print_skill(skill) -> None:
    print(f"\n--- skill {skill.id}: {skill.name} | uses={skill.uses} | updated={skill.updated_at} ---")
    print(f"Description: {skill.description}")
    print(f"Tags: {', '.join(skill.tags) if skill.tags else 'none'}")
    print(f"Use when: {skill.trigger}")
    print("\nWorkflow:")
    print(skill.workflow)
    if skill.prompt_template:
        print("\nPrompt template:")
        print(skill.prompt_template)
    if skill.verification:
        print("\nVerification:")
        print(skill.verification)


def cmd_learn(args: argparse.Namespace) -> int:
    cfg = HarnessConfig.from_env()
    store = LearningStore(cfg)
    if args.learn_cmd == "stats":
        print(store.stats())
        print(f"DB: {store.db_path}")
        return 0
    if args.learn_cmd == "list":
        for run in store.list_runs(limit=args.limit, successful_only=args.successful):
            _print_run(run)
        return 0
    if args.learn_cmd == "search":
        for run in store.search_runs(args.query, top_k=args.limit, successful_only=args.successful):
            _print_run(run)
        return 0
    if args.learn_cmd == "show":
        run = store.get_run(args.run_id)
        if run is None:
            print(f"Run not found: {args.run_id}", file=sys.stderr)
            return 1
        _print_run(run)
        print("\nAnswer:")
        print(run.answer)
        print("\nJudge:")
        print(run.judge)
        return 0
    raise SystemExit(f"Unknown learn command: {args.learn_cmd}")



def cmd_memory(args: argparse.Namespace) -> int:
    cfg = HarnessConfig.from_env()
    if args.memory_cmd == "profiles":
        print(format_profiles(PROFILES.values()))
        return 0
    if args.memory_cmd == "profile":
        try:
            profile = get_profile(args.name)
        except KeyError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(format_profile(profile))
        if args.exports:
            print("\nShell exports:")
            print(shell_exports(profile))
        if args.launch_ollama:
            print("\nStart Ollama with this profile:")
            print(shell_exports(profile))
            print("ollama serve")
        return 0
    if args.memory_cmd == "estimate":
        kv_type = args.kv or cfg.kv_cache_type
        ctx = args.ctx or cfg.num_ctx
        kv_mb = estimate_kv_cache_mb(ctx, kv_cache_type=kv_type)
        total, notes = estimate_total_memory_gb(cfg, kv_cache_type=kv_type, num_ctx=ctx)
        print(f"Config: {cfg.summary()}")
        print(f"\nInput context budget: ~{input_context_budget_tokens(cfg)} tokens")
        print(f"Estimated active KV cache: ~{kv_mb} MB using {kv_type}")
        print(f"Estimated total local run footprint: ~{total:.1f} GB")
        print("\nEstimate components:")
        for note in notes:
            print(f"  - {note}")
        print("\nThis is an approximation; verify with Activity Monitor and `ollama ps`.")
        return 0
    if args.memory_cmd == "env":
        print("Current Ollama memory-related env vars:")
        for k, v in current_ollama_env().items():
            print(f"  {k}={v if v is not None else '<unset>'}")
        print("\nHarness config:")
        print(cfg.summary())
        return 0
    if args.memory_cmd == "mac":
        print(mac_memory_snapshot())
        return 0
    raise SystemExit(f"Unknown memory command: {args.memory_cmd}")



def _print_tool_result(result) -> int:
    print(result.output)
    return 0 if result.ok else 1


def cmd_mac(args: argparse.Namespace) -> int:
    cfg = HarnessConfig.from_env()
    tools = ToolRegistry(cfg)
    dry_run = not getattr(args, "yes", False)

    if args.mac_cmd == "permissions":
        print(mac.permissions_help())
        return 0
    if args.mac_cmd == "screenshot":
        # Screenshot is not dry-run by default because it is observational.
        return _print_tool_result(tools.call("mac_screenshot", path=args.path, dry_run=args.dry_run))
    if args.mac_cmd == "open":
        return _print_tool_result(tools.call("mac_open_app", app_name=args.app, dry_run=dry_run))
    if args.mac_cmd == "activate":
        return _print_tool_result(tools.call("mac_activate_app", app_name=args.app, dry_run=dry_run))
    if args.mac_cmd == "click":
        return _print_tool_result(tools.call("mac_click", x=args.x, y=args.y, dry_run=dry_run))
    if args.mac_cmd == "type":
        return _print_tool_result(tools.call("mac_type", text=args.text, dry_run=dry_run))
    if args.mac_cmd == "hotkey":
        return _print_tool_result(tools.call("mac_hotkey", keys=args.keys, dry_run=dry_run))
    if args.mac_cmd == "applescript":
        script = _read_text_arg(args.script, args.file, field_name="script", required=True)
        return _print_tool_result(tools.call("mac_applescript", script=script, dry_run=dry_run))
    raise SystemExit(f"Unknown mac command: {args.mac_cmd}")


def cmd_act(args: argparse.Namespace) -> int:
    """Generate a small desktop action plan, optionally execute it.

    This is intentionally conservative. The model must emit a JSON array of tool
    calls; by default the plan is printed and not executed.
    """
    cfg = HarnessConfig.from_env()
    task = _read_prompt(args)
    tools = ToolRegistry(cfg)
    client = LocalLLMClient(cfg)
    allowed = [
        "mac_screenshot", "mac_open_app", "mac_activate_app", "mac_click",
        "mac_type", "mac_hotkey", "mac_applescript", "run_shell", "read_file",
        "list_files", "search_files", "run_python",
    ]
    instruction = f"""Create a conservative desktop automation plan as JSON only.
Return a JSON array. Each item must have this shape:
{{"tool": "tool_name", "args": {{...}}, "reason": "brief reason"}}

Allowed tools: {', '.join(allowed)}
Available manifest:
{tools.tool_manifest()}

Rules:
- Prefer open_app/activate_app, keyboard shortcuts, and AppleScript over blind coordinate clicking.
- Use screenshots only for observation.
- Use coordinate click only when necessary.
- Never type passwords, payment details, government IDs, or sensitive secrets.
- Keep the plan to at most {args.steps} actions.
- No markdown. JSON only.
"""
    resp = client.chat(
        [
            {"role": "system", "content": "You are a cautious macOS desktop automation planner."},
            {"role": "user", "content": instruction + "\nUSER TASK:\n" + task},
        ],
        temperature=0.1,
        max_tokens=1200,
    )
    raw = resp.content.strip()
    print("# Proposed action plan")
    print(raw)
    if not args.execute:
        print("\nDry run only. Re-run with --execute and enable GEMMA_ALLOW_MAC_CONTROL/GEMMA_ALLOW_MAC_SCREENSHOT as needed.")
        return 0
    try:
        actions = _parse_json_array(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        print(f"Could not parse JSON action plan: {exc}", file=sys.stderr)
        return 2
    for idx, action in enumerate(actions[: args.steps], start=1):
        if not isinstance(action, dict):
            print(f"Skipping invalid action {idx}: {action!r}")
            continue
        name = action.get("tool")
        kwargs = action.get("args") or {}
        if name not in allowed:
            print(f"Skipping disallowed tool {name!r}")
            continue
        if name.startswith("mac_") and name not in {"mac_screenshot"}:
            kwargs.setdefault("dry_run", not args.yes)
        result = tools.call(name, **kwargs)
        print(f"\n# Action {idx}: {name}")
        print(result.as_context())
        if not result.ok and args.stop_on_error:
            return 1
    return 0


def cmd_repo(args: argparse.Namespace) -> int:
    cfg = HarnessConfig.from_env()
    if args.repo_cmd == "instructions":
        instructions = load_repo_instructions(cfg.workspace)
        text = instructions.as_context(cfg.workspace)
        print(text or "No AGENTS.md/GEMMA.md/CLAUDE.md instructions found.")
        return 0
    if args.repo_cmd == "init-agents":
        try:
            path = create_agents_template(cfg.workspace, force=args.force)
        except FileExistsError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(f"Created {path}")
        return 0
    if args.repo_cmd == "status":
        print(status(cfg.workspace))
        return 0
    if args.repo_cmd == "diff":
        print(diff(cfg.workspace, staged=args.staged, max_chars=args.max_chars))
        return 0
    if args.repo_cmd == "diff-stat":
        print(diff_stat(cfg.workspace))
        return 0
    if args.repo_cmd == "checkpoint":
        if not is_git_repo(cfg.workspace):
            print("Not a git repository.", file=sys.stderr)
            return 1
        if args.commit:
            result = create_git_commit_checkpoint(cfg.workspace, message=args.message)
            print(result.output or "Commit checkpoint created.")
            return 0 if result.ok else 1
        try:
            path = create_patch_checkpoint(cfg.workspace, label=args.label or args.message)
        except Exception as exc:  # noqa: BLE001
            print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        print(f"Saved checkpoint patch: {path}")
        return 0
    raise SystemExit(f"Unknown repo command: {args.repo_cmd}")


def cmd_patch(args: argparse.Namespace) -> int:
    cfg = HarnessConfig.from_env()
    raw = _read_text_arg(args.patch, args.file, field_name="patch", required=True)
    try:
        patch_text = extract_unified_diff(raw)
    except Exception as exc:  # noqa: BLE001
        print(f"Could not extract unified diff: {exc}", file=sys.stderr)
        return 2
    if args.patch_cmd == "check":
        result = check_patch(cfg.workspace, patch_text)
        print(result.output)
        return 0 if result.ok else 1
    if args.patch_cmd == "apply":
        if not args.yes:
            print("Dry run only. Add --yes to apply this patch.")
            result = check_patch(cfg.workspace, patch_text)
            print(result.output)
            return 0 if result.ok else 1
        result = apply_patch(cfg.workspace, patch_text)
        print(result.output)
        return 0 if result.ok else 1
    raise SystemExit(f"Unknown patch command: {args.patch_cmd}")


def cmd_loop(args: argparse.Namespace) -> int:
    cfg = HarnessConfig.from_env()
    task = _read_prompt(args)
    allowed = _parse_tags(args.allowed_tools) if args.allowed_tools else None
    agent = ToolLoopAgent(cfg)
    run = agent.run(task, max_steps=args.steps, allowed_tools=allowed)
    if args.show_trace:
        for idx, step in enumerate(run.steps, start=1):
            print(f"\n# Step {idx}")
            print(f"Thought: {step.thought}")
            for result in step.results:
                print(result.as_context())
    print("\n# Final\n")
    print(run.final)
    return 0


def cmd_code(args: argparse.Namespace) -> int:
    cfg = HarnessConfig.from_env()
    task = _read_prompt(args)
    agent = CodeRepairAgent(cfg)
    run = agent.run(
        task,
        test_command=args.test,
        max_attempts=args.attempts,
        apply=args.apply,
        checkpoint=not args.no_checkpoint,
        timeout=args.timeout,
    )
    print(format_code_run(run, show_patch=not args.no_patch))
    return 0


def cmd_skill(args: argparse.Namespace) -> int:
    cfg = HarnessConfig.from_env()
    store = LearningStore(cfg)
    if args.skill_cmd == "list":
        for skill in store.list_skills(limit=args.limit):
            _print_skill(skill)
        return 0
    if args.skill_cmd == "search":
        for skill in store.search_skills(args.query, top_k=args.limit):
            _print_skill(skill)
        return 0
    if args.skill_cmd == "show":
        skill = store.get_skill(args.name_or_id)
        if skill is None:
            print(f"Skill not found: {args.name_or_id}", file=sys.stderr)
            return 1
        _print_skill(skill)
        return 0
    if args.skill_cmd == "delete":
        ok = store.delete_skill(args.name_or_id)
        print("Deleted." if ok else "Skill not found.")
        return 0 if ok else 1
    if args.skill_cmd == "save":
        workflow = _read_text_arg(args.workflow, args.workflow_file, field_name="workflow", required=True)
        prompt_template = _read_text_arg(args.prompt_template, args.prompt_file, field_name="prompt", required=False)
        verification = _read_text_arg(args.verification, args.verification_file, field_name="verification", required=False)
        skill_id = store.upsert_skill(
            name=args.name,
            description=args.description or "Reusable local workflow.",
            tags=_parse_tags(args.tags),
            trigger=args.trigger or args.description or args.name,
            workflow=workflow,
            prompt_template=prompt_template,
            verification=verification,
        )
        print(f"Saved skill {args.name!r} as id={skill_id}")
        return 0
    if args.skill_cmd == "from-run":
        skill_id = store.promote_run_to_skill(
            run_id=args.run_id,
            name=args.name,
            description=args.description,
            tags=_parse_tags(args.tags),
        )
        print(f"Promoted run {args.run_id} to skill {args.name!r} id={skill_id}")
        return 0
    raise SystemExit(f"Unknown skill command: {args.skill_cmd}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="gemma-harness", description="Slow multi-pass harness for Gemma 4 12B on local Macs.")
    p.add_argument(
        "--profile",
        choices=sorted(PROFILE_DEFAULTS),
        help="Apply a built-in runtime profile. Env vars still override individual settings.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    doctor = sub.add_parser("doctor", help="Check local model connectivity.")
    doctor.set_defaults(func=cmd_doctor)

    ask = sub.add_parser("ask", help="Single-pass direct answer.")
    ask.add_argument("prompt", nargs="?", help="Prompt text.")
    ask.add_argument("--file", help="Read prompt from a file.")
    ask.set_defaults(func=cmd_ask)

    deep = sub.add_parser("deep", help="Slow multi-candidate/critic/judge run with optional self-learning.")
    deep.add_argument("prompt", nargs="?", help="Prompt text.")
    deep.add_argument("--file", help="Read prompt from a file.")
    deep.add_argument("--candidates", type=int, default=None, help="Override GEMMA_CANDIDATES.")
    deep.add_argument("--rounds", type=int, default=None, help="Override GEMMA_DEBATE_ROUNDS.")
    deep.add_argument("--no-rag", action="store_true", help="Disable indexed workspace retrieval.")
    deep.add_argument("--no-learn", action="store_true", help="Disable retrieving and saving learning memory for this run.")
    deep.add_argument("--no-skills", action="store_true", help="Disable saved-skill retrieval for this run.")
    deep.add_argument("--skill", help="Force a named/id skill into context for this run.")
    deep.add_argument("--save-skill", help="After the run, promote it to a reusable skill with this name.")
    deep.add_argument("--skill-description", help="Description to use with --save-skill.")
    deep.add_argument("--skill-tags", help="Comma-separated tags for --save-skill.")
    deep.add_argument("--show-trace", action="store_true", help="Show plan, used skills, judge output, and lessons.")
    deep.set_defaults(func=cmd_deep)

    index = sub.add_parser("index", help="Index workspace for lexical RAG.")
    index.add_argument("path", nargs="?", help="Path to index; defaults to GEMMA_WORKSPACE/current directory.")
    index.set_defaults(func=cmd_index)

    search = sub.add_parser("search", help="Search indexed workspace.")
    search.add_argument("query")
    search.add_argument("--top-k", type=int, default=6)
    search.add_argument("--max-chars", type=int, default=1200)
    search.set_defaults(func=cmd_search)

    learn = sub.add_parser("learn", help="Inspect self-learning memory.")
    learn_sub = learn.add_subparsers(dest="learn_cmd", required=True)
    learn_stats = learn_sub.add_parser("stats", help="Show learning DB stats.")
    learn_stats.set_defaults(func=cmd_learn)
    learn_list = learn_sub.add_parser("list", help="List saved learning runs.")
    learn_list.add_argument("--limit", type=int, default=20)
    learn_list.add_argument("--successful", action="store_true")
    learn_list.set_defaults(func=cmd_learn)
    learn_search = learn_sub.add_parser("search", help="Search saved learning runs.")
    learn_search.add_argument("query")
    learn_search.add_argument("--limit", type=int, default=10)
    learn_search.add_argument("--successful", action="store_true")
    learn_search.set_defaults(func=cmd_learn)
    learn_show = learn_sub.add_parser("show", help="Show one saved learning run.")
    learn_show.add_argument("run_id", type=int)
    learn_show.set_defaults(func=cmd_learn)

    memory = sub.add_parser("memory", help="Inspect/apply RAM, context, and KV-cache policy profiles.")
    memory_sub = memory.add_subparsers(dest="memory_cmd", required=True)
    memory_profiles = memory_sub.add_parser("profiles", help="List available memory/KV profiles.")
    memory_profiles.set_defaults(func=cmd_memory)
    memory_profile = memory_sub.add_parser("profile", help="Show one memory/KV profile.")
    memory_profile.add_argument("name", choices=list(PROFILES.keys()))
    memory_profile.add_argument("--exports", action="store_true", help="Print shell exports for this profile.")
    memory_profile.add_argument("--launch-ollama", action="store_true", help="Print exports plus `ollama serve`.")
    memory_profile.set_defaults(func=cmd_memory)
    memory_estimate = memory_sub.add_parser("estimate", help="Estimate KV cache and total memory footprint.")
    memory_estimate.add_argument("--ctx", type=int, help="Context length to estimate; defaults to GEMMA_NUM_CTX.")
    memory_estimate.add_argument("--kv", choices=["f16", "q8_0", "q4_0"], help="KV cache type to estimate.")
    memory_estimate.set_defaults(func=cmd_memory)
    memory_env = memory_sub.add_parser("env", help="Show current Ollama/harness memory settings.")
    memory_env.set_defaults(func=cmd_memory)
    memory_mac = memory_sub.add_parser("mac", help="Best-effort macOS memory snapshot.")
    memory_mac.set_defaults(func=cmd_memory)

    mac_p = sub.add_parser("mac", help="Direct macOS automation helpers. Requires explicit env permissions.")
    mac_sub = mac_p.add_subparsers(dest="mac_cmd", required=True)
    mac_perm = mac_sub.add_parser("permissions", help="Show required macOS Accessibility/Screen Recording permissions.")
    mac_perm.set_defaults(func=cmd_mac)
    mac_shot = mac_sub.add_parser("screenshot", help="Take a screenshot into the workspace.")
    mac_shot.add_argument("path", nargs="?", default=".gemma_harness/screenshot.png")
    mac_shot.add_argument("--dry-run", action="store_true")
    mac_shot.set_defaults(func=cmd_mac)
    mac_open = mac_sub.add_parser("open", help="Open an app by name, e.g. Safari.")
    mac_open.add_argument("app")
    mac_open.add_argument("--yes", action="store_true", help="Actually execute. Default is dry-run.")
    mac_open.set_defaults(func=cmd_mac)
    mac_activate = mac_sub.add_parser("activate", help="Activate an app by name.")
    mac_activate.add_argument("app")
    mac_activate.add_argument("--yes", action="store_true", help="Actually execute. Default is dry-run.")
    mac_activate.set_defaults(func=cmd_mac)
    mac_click = mac_sub.add_parser("click", help="Click screen coordinates.")
    mac_click.add_argument("x", type=int)
    mac_click.add_argument("y", type=int)
    mac_click.add_argument("--yes", action="store_true", help="Actually execute. Default is dry-run.")
    mac_click.set_defaults(func=cmd_mac)
    mac_type = mac_sub.add_parser("type", help="Type short text into the active app.")
    mac_type.add_argument("text")
    mac_type.add_argument("--yes", action="store_true", help="Actually execute. Default is dry-run.")
    mac_type.set_defaults(func=cmd_mac)
    mac_hotkey = mac_sub.add_parser("hotkey", help="Press a hotkey, e.g. command,space or command,c.")
    mac_hotkey.add_argument("keys")
    mac_hotkey.add_argument("--yes", action="store_true", help="Actually execute. Default is dry-run.")
    mac_hotkey.set_defaults(func=cmd_mac)
    mac_as = mac_sub.add_parser("applescript", help="Run AppleScript.")
    mac_as.add_argument("--script", help="AppleScript text.")
    mac_as.add_argument("--file", help="Read AppleScript from file.")
    mac_as.add_argument("--yes", action="store_true", help="Actually execute. Default is dry-run.")
    mac_as.set_defaults(func=cmd_mac)

    act = sub.add_parser("act", help="Ask the model for a small macOS/tool action plan; dry-run by default.")
    act.add_argument("prompt", nargs="?", help="Goal to accomplish.")
    act.add_argument("--file", help="Read goal from a file.")
    act.add_argument("--steps", type=int, default=5, help="Maximum actions to propose/execute.")
    act.add_argument("--execute", action="store_true", help="Execute the proposed JSON action plan.")
    act.add_argument("--yes", action="store_true", help="Actually perform macOS click/type/hotkey. Without this, mac actions dry-run.")
    act.add_argument("--stop-on-error", action="store_true")
    act.set_defaults(func=cmd_act)


    repo = sub.add_parser("repo", help="Codex-like repository helpers: AGENTS.md, git status, diffs, checkpoints.")
    repo_sub = repo.add_subparsers(dest="repo_cmd", required=True)
    repo_inst = repo_sub.add_parser("instructions", help="Show AGENTS.md/GEMMA.md/CLAUDE.md repository guidance.")
    repo_inst.set_defaults(func=cmd_repo)
    repo_init = repo_sub.add_parser("init-agents", help="Create an AGENTS.md template in the workspace.")
    repo_init.add_argument("--force", action="store_true")
    repo_init.set_defaults(func=cmd_repo)
    repo_status = repo_sub.add_parser("status", help="Show git branch and short status.")
    repo_status.set_defaults(func=cmd_repo)
    repo_diff = repo_sub.add_parser("diff", help="Show current git diff.")
    repo_diff.add_argument("--staged", action="store_true")
    repo_diff.add_argument("--max-chars", type=int, default=120_000)
    repo_diff.set_defaults(func=cmd_repo)
    repo_stat = repo_sub.add_parser("diff-stat", help="Show compact git diff stat.")
    repo_stat.set_defaults(func=cmd_repo)
    repo_checkpoint = repo_sub.add_parser("checkpoint", help="Save a safe patch checkpoint; optionally commit instead.")
    repo_checkpoint.add_argument("--label", default="manual")
    repo_checkpoint.add_argument("--message", default="Gemma harness checkpoint")
    repo_checkpoint.add_argument("--commit", action="store_true", help="Create a real git commit checkpoint instead of a patch file.")
    repo_checkpoint.set_defaults(func=cmd_repo)

    patch_p = sub.add_parser("patch", help="Check or apply model-generated unified diffs.")
    patch_sub = patch_p.add_subparsers(dest="patch_cmd", required=True)
    patch_check_p = patch_sub.add_parser("check", help="Validate a unified diff with git apply --check.")
    patch_check_p.add_argument("--patch", help="Patch text.")
    patch_check_p.add_argument("--file", help="Read patch from file.")
    patch_check_p.set_defaults(func=cmd_patch)
    patch_apply_p = patch_sub.add_parser("apply", help="Apply a unified diff. Dry-run unless --yes is passed.")
    patch_apply_p.add_argument("--patch", help="Patch text.")
    patch_apply_p.add_argument("--file", help="Read patch from file.")
    patch_apply_p.add_argument("--yes", action="store_true")
    patch_apply_p.set_defaults(func=cmd_patch)

    loop = sub.add_parser("loop", help="Structured JSON tool-call loop for local repo tasks.")
    loop.add_argument("prompt", nargs="?", help="Goal to accomplish.")
    loop.add_argument("--file", help="Read goal from a file.")
    loop.add_argument("--steps", type=int, default=6)
    loop.add_argument("--allowed-tools", help="Comma-separated tool allowlist; default is safe coding tools.")
    loop.add_argument("--show-trace", action="store_true")
    loop.set_defaults(func=cmd_loop)

    code = sub.add_parser("code", help="Codex-like test-fix-test loop using unified patches.")
    code.add_argument("prompt", nargs="?", help="Coding task/failure to repair.")
    code.add_argument("--file", help="Read task from a file.")
    code.add_argument("--test", required=True, help="Test/verification command, e.g. 'pytest -q'.")
    code.add_argument("--attempts", type=int, default=3)
    code.add_argument("--apply", action="store_true", help="Apply proposed patches and rerun tests. Without this, patch is proposed only.")
    code.add_argument("--no-checkpoint", action="store_true", help="Do not save a patch checkpoint before attempting changes.")
    code.add_argument("--timeout", type=int, default=120)
    code.add_argument("--no-patch", action="store_true", help="Hide patch body in output.")
    code.set_defaults(func=cmd_code)

    skill = sub.add_parser("skill", help="Save, inspect, and reuse common workflows.")
    skill_sub = skill.add_subparsers(dest="skill_cmd", required=True)
    skill_list = skill_sub.add_parser("list", help="List saved skills.")
    skill_list.add_argument("--limit", type=int, default=50)
    skill_list.set_defaults(func=cmd_skill)
    skill_search = skill_sub.add_parser("search", help="Search saved skills.")
    skill_search.add_argument("query")
    skill_search.add_argument("--limit", type=int, default=10)
    skill_search.set_defaults(func=cmd_skill)
    skill_show = skill_sub.add_parser("show", help="Show one skill.")
    skill_show.add_argument("name_or_id")
    skill_show.set_defaults(func=cmd_skill)
    skill_delete = skill_sub.add_parser("delete", help="Delete one skill.")
    skill_delete.add_argument("name_or_id")
    skill_delete.set_defaults(func=cmd_skill)
    skill_save = skill_sub.add_parser("save", help="Create or update a reusable workflow skill.")
    skill_save.add_argument("name")
    skill_save.add_argument("--description")
    skill_save.add_argument("--tags", help="Comma-separated tags.")
    skill_save.add_argument("--trigger", help="When this skill should be used.")
    skill_save.add_argument("--workflow", help="Workflow text.")
    skill_save.add_argument("--workflow-file", help="Read workflow text from file.")
    skill_save.add_argument("--prompt-template", help="Prompt template text.")
    skill_save.add_argument("--prompt-file", help="Read prompt template from file.")
    skill_save.add_argument("--verification", help="Verification checklist.")
    skill_save.add_argument("--verification-file", help="Read verification checklist from file.")
    skill_save.set_defaults(func=cmd_skill)
    skill_from_run = skill_sub.add_parser("from-run", help="Promote a saved run into a reusable skill.")
    skill_from_run.add_argument("run_id", type=int)
    skill_from_run.add_argument("name")
    skill_from_run.add_argument("--description")
    skill_from_run.add_argument("--tags", help="Comma-separated tags.")
    skill_from_run.set_defaults(func=cmd_skill)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "profile", None):
        os.environ["GEMMA_PROFILE"] = args.profile
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
