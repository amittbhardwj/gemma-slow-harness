from __future__ import annotations

import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(slots=True)
class GitResult:
    ok: bool
    returncode: int
    stdout: str
    stderr: str

    @property
    def output(self) -> str:
        return ((self.stdout or "") + (self.stderr or "")).strip()


def run_git(workspace: Path, args: list[str], *, timeout: int = 30) -> GitResult:
    proc = subprocess.run(
        ["git", *args],
        cwd=str(workspace),
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    return GitResult(proc.returncode == 0, proc.returncode, proc.stdout, proc.stderr)


def is_git_repo(workspace: Path) -> bool:
    result = run_git(workspace, ["rev-parse", "--is-inside-work-tree"])
    return result.ok and result.stdout.strip() == "true"


def status(workspace: Path) -> str:
    if not is_git_repo(workspace):
        return "Not a git repository."
    branch = run_git(workspace, ["branch", "--show-current"]).stdout.strip() or "<detached>"
    porcelain = run_git(workspace, ["status", "--short"]).stdout.strip()
    return f"Branch: {branch}\n" + (porcelain or "Working tree clean.")


def diff(workspace: Path, *, staged: bool = False, max_chars: int = 120_000) -> str:
    if not is_git_repo(workspace):
        return "Not a git repository."
    args = ["diff", "--staged"] if staged else ["diff"]
    text = run_git(workspace, args, timeout=60).stdout
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n[TRUNCATED: diff exceeded max_chars]\n"
    return text or "No diff."


def diff_stat(workspace: Path) -> str:
    if not is_git_repo(workspace):
        return "Not a git repository."
    text = run_git(workspace, ["diff", "--stat"], timeout=30).stdout.strip()
    return text or "No diff."


def create_patch_checkpoint(workspace: Path, *, label: str = "manual", include_untracked: bool = True) -> Path:
    """Save current worktree changes as a patch file without changing git history.

    This is safer than automatic commits on a user's machine. The patch can be
    inspected later or applied with `git apply` manually.
    """

    if not is_git_repo(workspace):
        raise RuntimeError("Not a git repository.")
    checkpoint_dir = workspace / ".gemma_harness" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_label = "".join(c if c.isalnum() or c in "-_" else "-" for c in label).strip("-") or "checkpoint"
    patch_path = checkpoint_dir / f"{ts}_{safe_label}.patch"

    base = run_git(workspace, ["diff", "--binary"], timeout=60).stdout
    staged = run_git(workspace, ["diff", "--cached", "--binary"], timeout=60).stdout
    untracked_blocks: list[str] = []
    if include_untracked:
        untracked = run_git(workspace, ["ls-files", "--others", "--exclude-standard"], timeout=30).stdout.splitlines()
        for rel in untracked:
            p = workspace / rel
            if p.is_file() and p.stat().st_size < 2_000_000:
                # git diff --no-index returns 1 when differences exist; stdout is still the patch.
                res = subprocess.run(
                    ["git", "diff", "--no-index", "--binary", "/dev/null", rel],
                    cwd=str(workspace),
                    text=True,
                    capture_output=True,
                    timeout=30,
                    check=False,
                )
                untracked_blocks.append(res.stdout)

    content = (
        f"# Gemma Harness checkpoint\n"
        f"# Created: {ts}\n"
        f"# Label: {label}\n"
        f"# Workspace: {workspace}\n"
        f"# Apply manually with: git apply {patch_path.name}\n\n"
        + staged
        + "\n"
        + base
        + "\n"
        + "\n".join(untracked_blocks)
    )
    patch_path.write_text(content, encoding="utf-8")
    return patch_path


def create_git_commit_checkpoint(workspace: Path, *, message: str) -> GitResult:
    if not is_git_repo(workspace):
        return GitResult(False, 1, "", "Not a git repository.")
    add = run_git(workspace, ["add", "-A"], timeout=60)
    if not add.ok:
        return add
    return run_git(workspace, ["commit", "-m", message], timeout=120)
