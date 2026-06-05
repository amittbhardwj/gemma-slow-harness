from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .config import HarnessConfig
from .utils import iter_text_files, read_text, safe_path
from . import macos_control as mac
from . import git_ops
from .patching import apply_patch, check_patch, extract_unified_diff
from .repo_context import load_repo_instructions


@dataclass(slots=True)
class ToolResult:
    name: str
    ok: bool
    output: str

    def as_context(self) -> str:
        status = "OK" if self.ok else "ERROR"
        return f"[TOOL {self.name} {status}]\n{self.output.strip()}"


class ToolRegistry:
    """Conservative local tools.

    Write and shell execution are disabled by default. Keep them disabled until you
    trust the workspace and model behavior.
    """

    def __init__(self, cfg: HarnessConfig) -> None:
        self.cfg = cfg
        self.tools: dict[str, Callable[..., ToolResult]] = {
            "read_file": self.read_file,
            "list_files": self.list_files,
            "search_files": self.search_files,
            "run_python": self.run_python,
            "write_file": self.write_file,
            "run_shell": self.run_shell,
            "mac_permissions": self.mac_permissions,
            "mac_screenshot": self.mac_screenshot,
            "mac_open_app": self.mac_open_app,
            "mac_activate_app": self.mac_activate_app,
            "mac_click": self.mac_click,
            "mac_type": self.mac_type,
            "mac_hotkey": self.mac_hotkey,
            "mac_applescript": self.mac_applescript,
            "repo_instructions": self.repo_instructions,
            "git_status": self.git_status,
            "git_diff": self.git_diff,
            "git_checkpoint": self.git_checkpoint,
            "patch_check": self.patch_check,
            "patch_apply": self.patch_apply,
        }

    def call(self, name: str, **kwargs) -> ToolResult:
        if name not in self.tools:
            return ToolResult(name=name, ok=False, output=f"Unknown tool: {name}")
        try:
            return self.tools[name](**kwargs)
        except Exception as exc:  # noqa: BLE001 - CLI surface should return errors as text.
            return ToolResult(name=name, ok=False, output=f"{type(exc).__name__}: {exc}")

    def read_file(self, path: str, max_chars: int = 50_000) -> ToolResult:
        p = safe_path(self.cfg.workspace, path)
        if not p.exists() or not p.is_file():
            return ToolResult("read_file", False, f"File not found: {path}")
        return ToolResult("read_file", True, read_text(p, max_chars=max_chars))

    def list_files(self, path: str = ".", max_files: int = 200) -> ToolResult:
        p = safe_path(self.cfg.workspace, path)
        if not p.exists():
            return ToolResult("list_files", False, f"Path not found: {path}")
        files = []
        for idx, f in enumerate(iter_text_files(p if p.is_dir() else p.parent)):
            if idx >= max_files:
                files.append(f"... truncated after {max_files} files")
                break
            files.append(str(f.relative_to(self.cfg.workspace)))
        return ToolResult("list_files", True, "\n".join(files) if files else "No text files found.")

    def search_files(self, query: str, path: str = ".", max_matches: int = 50) -> ToolResult:
        p = safe_path(self.cfg.workspace, path)
        q = query.lower()
        matches: list[str] = []
        for f in iter_text_files(p if p.is_dir() else p.parent):
            text = read_text(f, max_chars=200_000)
            for line_no, line in enumerate(text.splitlines(), start=1):
                if q in line.lower():
                    rel = f.relative_to(self.cfg.workspace)
                    matches.append(f"{rel}:{line_no}: {line[:220]}")
                    if len(matches) >= max_matches:
                        return ToolResult("search_files", True, "\n".join(matches))
        return ToolResult("search_files", True, "No matches.")

    def run_python(self, code: str) -> ToolResult:
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "snippet.py"
            script.write_text(code, encoding="utf-8")
            proc = subprocess.run(
                ["python3", str(script)],
                cwd=str(self.cfg.workspace),
                text=True,
                capture_output=True,
                timeout=self.cfg.tool_timeout_sec,
                check=False,
            )
        out = (proc.stdout or "") + (proc.stderr or "")
        return ToolResult("run_python", proc.returncode == 0, out.strip() or f"returncode={proc.returncode}")

    def write_file(self, path: str, content: str) -> ToolResult:
        if not self.cfg.allow_write:
            return ToolResult("write_file", False, "Writing is disabled. Set GEMMA_ALLOW_WRITE=true to enable.")
        p = safe_path(self.cfg.workspace, path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return ToolResult("write_file", True, f"Wrote {p.relative_to(self.cfg.workspace)}")

    def run_shell(self, command: str) -> ToolResult:
        if not self.cfg.allow_shell:
            return ToolResult("run_shell", False, "Shell execution is disabled. Set GEMMA_ALLOW_SHELL=true to enable.")
        proc = subprocess.run(
            command,
            cwd=str(self.cfg.workspace),
            shell=True,
            text=True,
            capture_output=True,
            timeout=self.cfg.tool_timeout_sec,
            check=False,
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        return ToolResult("run_shell", proc.returncode == 0, out.strip() or f"returncode={proc.returncode}")


    def mac_permissions(self) -> ToolResult:
        return ToolResult("mac_permissions", True, mac.permissions_help())

    def mac_screenshot(self, path: str = ".gemma_harness/screenshot.png", dry_run: bool = False) -> ToolResult:
        if not self.cfg.allow_mac_screenshot:
            return ToolResult("mac_screenshot", False, "macOS screenshots are disabled. Set GEMMA_ALLOW_MAC_SCREENSHOT=true to enable.")
        result = mac.screenshot(self.cfg.workspace, path, dry_run=dry_run)
        return ToolResult("mac_screenshot", result.ok, result.as_text())

    def mac_open_app(self, app_name: str, dry_run: bool = True) -> ToolResult:
        if not self.cfg.allow_mac_control:
            return ToolResult("mac_open_app", False, "macOS control is disabled. Set GEMMA_ALLOW_MAC_CONTROL=true to enable.")
        result = mac.open_app(app_name, dry_run=dry_run)
        return ToolResult("mac_open_app", result.ok, result.as_text())

    def mac_activate_app(self, app_name: str, dry_run: bool = True) -> ToolResult:
        if not self.cfg.allow_mac_control:
            return ToolResult("mac_activate_app", False, "macOS control is disabled. Set GEMMA_ALLOW_MAC_CONTROL=true to enable.")
        result = mac.activate_app(app_name, dry_run=dry_run)
        return ToolResult("mac_activate_app", result.ok, result.as_text())

    def mac_click(self, x: int, y: int, dry_run: bool = True) -> ToolResult:
        if not self.cfg.allow_mac_control:
            return ToolResult("mac_click", False, "macOS control is disabled. Set GEMMA_ALLOW_MAC_CONTROL=true to enable.")
        result = mac.click(int(x), int(y), dry_run=dry_run)
        return ToolResult("mac_click", result.ok, result.as_text())

    def mac_type(self, text: str, dry_run: bool = True) -> ToolResult:
        if not self.cfg.allow_mac_control:
            return ToolResult("mac_type", False, "macOS control is disabled. Set GEMMA_ALLOW_MAC_CONTROL=true to enable.")
        result = mac.type_text(text, dry_run=dry_run)
        return ToolResult("mac_type", result.ok, result.as_text())

    def mac_hotkey(self, keys: str, dry_run: bool = True) -> ToolResult:
        if not self.cfg.allow_mac_control:
            return ToolResult("mac_hotkey", False, "macOS control is disabled. Set GEMMA_ALLOW_MAC_CONTROL=true to enable.")
        result = mac.hotkey(keys, dry_run=dry_run)
        return ToolResult("mac_hotkey", result.ok, result.as_text())

    def mac_applescript(self, script: str, dry_run: bool = True) -> ToolResult:
        if not self.cfg.allow_mac_control:
            return ToolResult("mac_applescript", False, "macOS control is disabled. Set GEMMA_ALLOW_MAC_CONTROL=true to enable.")
        result = mac.run_applescript(script, dry_run=dry_run)
        return ToolResult("mac_applescript", result.ok, result.as_text())


    def repo_instructions(self) -> ToolResult:
        instructions = load_repo_instructions(self.cfg.workspace)
        text = instructions.as_context(self.cfg.workspace)
        return ToolResult("repo_instructions", True, text or "No AGENTS.md-style instructions found.")

    def git_status(self) -> ToolResult:
        return ToolResult("git_status", True, git_ops.status(self.cfg.workspace))

    def git_diff(self, staged: bool = False, max_chars: int = 120_000) -> ToolResult:
        return ToolResult("git_diff", True, git_ops.diff(self.cfg.workspace, staged=bool(staged), max_chars=int(max_chars)))

    def git_checkpoint(self, label: str = "manual") -> ToolResult:
        if not self.cfg.allow_write:
            return ToolResult("git_checkpoint", False, "Checkpoint writing is disabled. Set GEMMA_ALLOW_WRITE=true to enable.")
        try:
            path = git_ops.create_patch_checkpoint(self.cfg.workspace, label=label)
            return ToolResult("git_checkpoint", True, f"Saved checkpoint patch: {path}")
        except Exception as exc:  # noqa: BLE001
            return ToolResult("git_checkpoint", False, f"{type(exc).__name__}: {exc}")

    def patch_check(self, patch_text: str) -> ToolResult:
        try:
            diff_text = extract_unified_diff(patch_text)
            result = check_patch(self.cfg.workspace, diff_text)
            return ToolResult("patch_check", result.ok, result.output)
        except Exception as exc:  # noqa: BLE001
            return ToolResult("patch_check", False, f"{type(exc).__name__}: {exc}")

    def patch_apply(self, patch_text: str) -> ToolResult:
        if not self.cfg.allow_write:
            return ToolResult("patch_apply", False, "Patch application is disabled. Set GEMMA_ALLOW_WRITE=true to enable.")
        try:
            diff_text = extract_unified_diff(patch_text)
            result = apply_patch(self.cfg.workspace, diff_text)
            return ToolResult("patch_apply", result.ok, result.output)
        except Exception as exc:  # noqa: BLE001
            return ToolResult("patch_apply", False, f"{type(exc).__name__}: {exc}")

    def tool_manifest(self) -> str:
        manifest = {
            "read_file": {"args": {"path": "str", "max_chars": "int optional"}},
            "list_files": {"args": {"path": "str optional", "max_files": "int optional"}},
            "search_files": {"args": {"query": "str", "path": "str optional", "max_matches": "int optional"}},
            "run_python": {"args": {"code": "str"}},
            "write_file": {"args": {"path": "str", "content": "str"}, "enabled": self.cfg.allow_write},
            "run_shell": {"args": {"command": "str"}, "enabled": self.cfg.allow_shell},
            "mac_permissions": {"args": {}, "enabled": True},
            "mac_screenshot": {"args": {"path": "str optional", "dry_run": "bool optional"}, "enabled": self.cfg.allow_mac_screenshot},
            "mac_open_app": {"args": {"app_name": "str", "dry_run": "bool optional default true"}, "enabled": self.cfg.allow_mac_control},
            "mac_activate_app": {"args": {"app_name": "str", "dry_run": "bool optional default true"}, "enabled": self.cfg.allow_mac_control},
            "mac_click": {"args": {"x": "int", "y": "int", "dry_run": "bool optional default true"}, "enabled": self.cfg.allow_mac_control},
            "mac_type": {"args": {"text": "str", "dry_run": "bool optional default true"}, "enabled": self.cfg.allow_mac_control},
            "mac_hotkey": {"args": {"keys": "str like command,space", "dry_run": "bool optional default true"}, "enabled": self.cfg.allow_mac_control},
            "mac_applescript": {"args": {"script": "str", "dry_run": "bool optional default true"}, "enabled": self.cfg.allow_mac_control},
            "repo_instructions": {"args": {}, "enabled": True},
            "git_status": {"args": {}, "enabled": True},
            "git_diff": {"args": {"staged": "bool optional", "max_chars": "int optional"}, "enabled": True},
            "git_checkpoint": {"args": {"label": "str optional"}, "enabled": self.cfg.allow_write},
            "patch_check": {"args": {"patch_text": "str"}, "enabled": True},
            "patch_apply": {"args": {"patch_text": "str"}, "enabled": self.cfg.allow_write},
        }
        return json.dumps(manifest, indent=2)
