from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class PatchResult:
    ok: bool
    output: str


_FENCE_RE = re.compile(r"```(?:diff|patch)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_unified_diff(text: str) -> str:
    """Extract a unified diff from model output.

    Accepts either raw diff text or a fenced diff block. This intentionally does
    not try to parse informal descriptions; the model must provide real patches.
    """

    match = _FENCE_RE.search(text)
    candidate = match.group(1).strip() if match else text.strip()
    if "diff --git " in candidate or "--- " in candidate and "+++ " in candidate and "@@" in candidate:
        return candidate + ("\n" if not candidate.endswith("\n") else "")
    raise ValueError("No unified diff found. Ask the model to return a git-style unified diff only.")


def check_patch(workspace: Path, patch_text: str) -> PatchResult:
    proc = subprocess.run(
        ["git", "apply", "--check", "-"],
        cwd=str(workspace),
        input=patch_text,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    return PatchResult(proc.returncode == 0, out.strip() or "Patch applies cleanly.")


def apply_patch(workspace: Path, patch_text: str) -> PatchResult:
    check = check_patch(workspace, patch_text)
    if not check.ok:
        return check
    proc = subprocess.run(
        ["git", "apply", "-"],
        cwd=str(workspace),
        input=patch_text,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    return PatchResult(proc.returncode == 0, out.strip() or "Patch applied.")
