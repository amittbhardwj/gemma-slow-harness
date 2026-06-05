from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .config import HarnessConfig
from .git_ops import create_patch_checkpoint, diff, diff_stat, is_git_repo, status
from .llm import LocalLLMClient
from .patching import apply_patch, check_patch, extract_unified_diff
from .repo_context import load_repo_instructions
from .tools import ToolRegistry


@dataclass(slots=True)
class CodeAttempt:
    index: int
    test_output_before: str
    patch_text: str
    patch_check: str
    applied: bool
    test_output_after: str = ""


@dataclass(slots=True)
class CodeRun:
    final: str
    attempts: list[CodeAttempt] = field(default_factory=list)
    checkpoint: str | None = None
    initial_status: str = ""
    final_status: str = ""


def run_command(workspace: Path, command: str, *, timeout: int) -> tuple[bool, str]:
    proc = subprocess.run(
        command,
        cwd=str(workspace),
        shell=True,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    out = ((proc.stdout or "") + (proc.stderr or "")).strip()
    return proc.returncode == 0, out or f"returncode={proc.returncode}"


class CodeRepairAgent:
    """Codex-like test-fix-test agent using git patches.

    It asks the model for a unified diff, checks it with git apply --check, applies
    it only when explicitly allowed, then reruns the configured test command.
    """

    def __init__(self, cfg: HarnessConfig) -> None:
        self.cfg = cfg
        self.llm = LocalLLMClient(cfg)
        self.tools = ToolRegistry(cfg)

    def propose_patch(self, task: str, *, test_command: str, test_output: str, extra_context: str = "") -> str:
        repo_instructions = load_repo_instructions(self.cfg.workspace).as_context(self.cfg.workspace)
        files = self.tools.list_files(max_files=250).output
        searches = self.tools.search_files(task.split()[0] if task.split() else task, max_matches=20).output
        current_diff = diff(self.cfg.workspace, max_chars=50_000) if is_git_repo(self.cfg.workspace) else "Not a git repo."
        prompt = f"""
You are a local coding repair agent. Return a git-style unified diff only. No markdown, no prose.

TASK:
{task}

TEST COMMAND:
{test_command}

TEST OUTPUT:
{test_output[-12000:]}

REPOSITORY INSTRUCTIONS:
{repo_instructions or 'No AGENTS.md-style instructions found.'}

CURRENT GIT STATUS:
{status(self.cfg.workspace)}

CURRENT DIFF:
{current_diff}

RELEVANT FILE LIST:
{files[:12000]}

SEARCH HINTS:
{searches[:8000]}

EXTRA CONTEXT:
{extra_context}

Rules:
- Return only a valid unified diff that can be applied with git apply.
- Keep changes minimal and focused.
- Do not include generated/cache files.
- Do not change unrelated formatting.
"""
        resp = self.llm.chat(
            [{"role": "system", "content": "You output only unified diffs."}, {"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=2600,
        )
        return extract_unified_diff(resp.content)

    def run(
        self,
        task: str,
        *,
        test_command: str,
        max_attempts: int = 3,
        apply: bool = False,
        checkpoint: bool = True,
        timeout: int | None = None,
    ) -> CodeRun:
        timeout = timeout or max(self.cfg.tool_timeout_sec, 60)
        run = CodeRun(final="", initial_status=status(self.cfg.workspace), final_status="")
        if checkpoint and is_git_repo(self.cfg.workspace):
            path = create_patch_checkpoint(self.cfg.workspace, label="before-code-agent")
            run.checkpoint = str(path)

        ok, test_out = run_command(self.cfg.workspace, test_command, timeout=timeout)
        if ok:
            run.final = "Test command already passes. No patch generated."
            run.final_status = status(self.cfg.workspace)
            return run

        last_output = test_out
        for idx in range(1, max_attempts + 1):
            patch_text = self.propose_patch(task, test_command=test_command, test_output=last_output)
            check = check_patch(self.cfg.workspace, patch_text)
            attempt = CodeAttempt(
                index=idx,
                test_output_before=last_output,
                patch_text=patch_text,
                patch_check=check.output,
                applied=False,
            )
            if check.ok and apply:
                applied = apply_patch(self.cfg.workspace, patch_text)
                attempt.applied = applied.ok
                attempt.patch_check = applied.output
                if applied.ok:
                    ok, last_output = run_command(self.cfg.workspace, test_command, timeout=timeout)
                    attempt.test_output_after = last_output
                    run.attempts.append(attempt)
                    if ok:
                        run.final = "Patch applied and test command now passes."
                        break
                    continue
            run.attempts.append(attempt)
            if not apply:
                run.final = "Patch proposed but not applied. Re-run with --apply after review."
                break
            if not check.ok:
                last_output = f"Patch failed to apply:\n{check.output}\n\nPrevious test output:\n{last_output}"
        if not run.final:
            run.final = "Reached max attempts without passing the test command."
        run.final_status = status(self.cfg.workspace)
        return run


def format_code_run(run: CodeRun, *, show_patch: bool = True) -> str:
    lines = []
    lines.append("# Code Agent Result")
    lines.append("")
    lines.append(run.final)
    if run.checkpoint:
        lines.append(f"\nCheckpoint patch: {run.checkpoint}")
    lines.append("\n## Initial status\n")
    lines.append(run.initial_status)
    for attempt in run.attempts:
        lines.append(f"\n## Attempt {attempt.index}\n")
        lines.append("Patch check/apply result:")
        lines.append(attempt.patch_check)
        lines.append(f"Applied: {attempt.applied}")
        if show_patch:
            lines.append("\nPatch:\n")
            lines.append(attempt.patch_text)
        if attempt.test_output_after:
            lines.append("\nTest output after patch:\n")
            lines.append(attempt.test_output_after[-8000:])
    lines.append("\n## Final status\n")
    lines.append(run.final_status)
    return "\n".join(lines)
