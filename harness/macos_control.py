from __future__ import annotations

import json
import os
import platform
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .utils import safe_path


MODIFIER_KEYS = {
    "cmd": "command down",
    "command": "command down",
    "shift": "shift down",
    "option": "option down",
    "alt": "option down",
    "control": "control down",
    "ctrl": "control down",
}

SPECIAL_KEY_CODES = {
    "return": 36,
    "enter": 36,
    "tab": 48,
    "escape": 53,
    "esc": 53,
    "space": 49,
    "delete": 51,
    "backspace": 51,
    "forward_delete": 117,
    "left": 123,
    "right": 124,
    "down": 125,
    "up": 126,
    "home": 115,
    "end": 119,
    "page_up": 116,
    "page_down": 121,
}


@dataclass(slots=True)
class MacActionResult:
    ok: bool
    output: str
    command: list[str] | None = None

    def as_text(self) -> str:
        command_text = ""
        if self.command:
            command_text = "\nCOMMAND: " + " ".join(shlex.quote(x) for x in self.command)
        return f"{self.output}{command_text}"


def is_macos() -> bool:
    return platform.system() == "Darwin"


def _applescript_quote(value: str) -> str:
    # JSON encoding is a valid way to produce a double-quoted AppleScript string
    # for the simple literals used here.
    return json.dumps(value)


def _run(cmd: list[str], *, timeout: int = 20, dry_run: bool = False) -> MacActionResult:
    if dry_run:
        return MacActionResult(True, "DRY RUN: command was not executed.", cmd)
    if not is_macos():
        return MacActionResult(False, "macOS automation is only available on macOS.", cmd)
    proc = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout, check=False)
    out = ((proc.stdout or "") + (proc.stderr or "")).strip()
    return MacActionResult(proc.returncode == 0, out or f"returncode={proc.returncode}", cmd)


def osascript(script: str, *, timeout: int = 20, dry_run: bool = False) -> MacActionResult:
    return _run(["osascript", "-e", script], timeout=timeout, dry_run=dry_run)


def permissions_help() -> str:
    return """macOS permissions needed for GUI automation:

1. System Settings → Privacy & Security → Accessibility
   Add and enable your terminal app, Python, or the app launching gemma-harness.

2. System Settings → Privacy & Security → Screen Recording
   Add and enable your terminal app/Python if you want screenshots.

3. Restart the terminal after changing permissions.

Security model used by this harness:
- GUI control is disabled by default.
- Enable with GEMMA_ALLOW_MAC_CONTROL=true.
- Screenshots are controlled by GEMMA_ALLOW_MAC_SCREENSHOT=true.
- Direct CLI actions such as click/type/hotkey default to dry-run unless --yes is passed.
""".strip()


def open_app(app_name: str, *, dry_run: bool = False) -> MacActionResult:
    return _run(["open", "-a", app_name], dry_run=dry_run)


def activate_app(app_name: str, *, dry_run: bool = False) -> MacActionResult:
    script = f"tell application {_applescript_quote(app_name)} to activate"
    return osascript(script, dry_run=dry_run)


def screenshot(workspace: Path, path: str = ".gemma_harness/screenshot.png", *, dry_run: bool = False) -> MacActionResult:
    p = safe_path(workspace, path)
    p.parent.mkdir(parents=True, exist_ok=True)
    result = _run(["screencapture", "-x", str(p)], timeout=30, dry_run=dry_run)
    if result.ok and not dry_run:
        result.output = f"Screenshot saved to {p}"
    return result


def click(x: int, y: int, *, dry_run: bool = False) -> MacActionResult:
    # UI scripting requires Accessibility permission.
    script = f'tell application "System Events" to click at {{{int(x)}, {int(y)}}}'
    return osascript(script, dry_run=dry_run)


def type_text(text: str, *, dry_run: bool = False) -> MacActionResult:
    # Keystroke is appropriate for short text. For long text, paste from clipboard
    # can be added as a separate explicit tool to avoid unexpected clipboard changes.
    script = f'tell application "System Events" to keystroke {_applescript_quote(text)}'
    return osascript(script, timeout=60, dry_run=dry_run)


def hotkey(keys: str | list[str], *, dry_run: bool = False) -> MacActionResult:
    if isinstance(keys, str):
        parts = [x.strip().lower() for x in keys.replace("+", ",").split(",") if x.strip()]
    else:
        parts = [str(x).strip().lower() for x in keys if str(x).strip()]
    if not parts:
        return MacActionResult(False, "No keys provided.")

    modifiers = [MODIFIER_KEYS[p] for p in parts if p in MODIFIER_KEYS]
    non_modifiers = [p for p in parts if p not in MODIFIER_KEYS]
    if len(non_modifiers) != 1:
        return MacActionResult(False, "Provide exactly one non-modifier key, for example: command,space or command,c.")
    key = non_modifiers[0]
    using = f" using {{{', '.join(modifiers)}}}" if modifiers else ""
    if key in SPECIAL_KEY_CODES:
        script = f'tell application "System Events" to key code {SPECIAL_KEY_CODES[key]}{using}'
    elif len(key) == 1:
        script = f'tell application "System Events" to keystroke {_applescript_quote(key)}{using}'
    else:
        return MacActionResult(False, f"Unknown key: {key}. Use one character or a known special key.")
    return osascript(script, dry_run=dry_run)


def run_applescript(script: str, *, dry_run: bool = False) -> MacActionResult:
    return osascript(script, timeout=60, dry_run=dry_run)
