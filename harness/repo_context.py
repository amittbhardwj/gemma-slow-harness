from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .utils import read_text, safe_path

INSTRUCTION_FILENAMES = ("AGENTS.md", "GEMMA.md", "CLAUDE.md", ".gemma_harness/INSTRUCTIONS.md")
DEFAULT_MAX_CHARS = 60_000


@dataclass(slots=True)
class InstructionFile:
    path: Path
    text: str

    def as_block(self, workspace: Path) -> str:
        try:
            rel = self.path.relative_to(workspace)
        except ValueError:
            rel = self.path
        return f"--- {rel} ---\n{self.text.strip()}"


@dataclass(slots=True)
class RepoInstructions:
    files: list[InstructionFile]

    @property
    def text(self) -> str:
        if not self.files:
            return ""
        # workspace is not stored; individual as_block is used by load_repo_instructions.
        return "\n\n".join(f"--- {f.path} ---\n{f.text.strip()}" for f in self.files)

    def as_context(self, workspace: Path) -> str:
        if not self.files:
            return ""
        blocks = [f.as_block(workspace) for f in self.files]
        return "# REPOSITORY INSTRUCTIONS\n\n" + "\n\n".join(blocks)


def _candidate_instruction_paths(workspace: Path) -> list[Path]:
    workspace = workspace.expanduser().resolve()
    candidates: list[Path] = []

    # Root-level instruction files.
    for name in INSTRUCTION_FILENAMES:
        candidates.append(workspace / name)

    # Shallow nested AGENTS.md files are useful in monorepos, but avoid expensive scans.
    for path in workspace.rglob("AGENTS.md"):
        if path == workspace / "AGENTS.md":
            continue
        parts = path.relative_to(workspace).parts
        if len(parts) <= 4 and not any(p in {".git", "node_modules", ".venv", "__pycache__", ".pytest_cache"} for p in parts):
            candidates.append(path)
    # Keep stable ordering and dedupe.
    unique: list[Path] = []
    seen = set()
    for p in candidates:
        key = str(p)
        if key not in seen:
            unique.append(p)
            seen.add(key)
    return unique


def load_repo_instructions(workspace: Path, *, max_chars: int = DEFAULT_MAX_CHARS) -> RepoInstructions:
    """Load AGENTS.md-style repository guidance.

    The harness treats these files like persistent developer instructions for the
    current workspace. Nested AGENTS.md files are included to support monorepos.
    """

    workspace = workspace.expanduser().resolve()
    files: list[InstructionFile] = []
    remaining = max_chars
    for path in _candidate_instruction_paths(workspace):
        if remaining <= 0:
            break
        if not path.exists() or not path.is_file():
            continue
        # Preserve safe_path behavior for files under workspace.
        safe_path(workspace, str(path.relative_to(workspace)))
        text = read_text(path, max_chars=remaining)
        if text.strip():
            files.append(InstructionFile(path=path, text=text))
            remaining -= len(text)
    return RepoInstructions(files=files)


def create_agents_template(workspace: Path, *, force: bool = False) -> Path:
    path = workspace.expanduser().resolve() / "AGENTS.md"
    if path.exists() and not force:
        raise FileExistsError(f"AGENTS.md already exists at {path}")
    path.write_text(
        "# AGENTS.md\n\n"
        "Instructions for local AI coding agents working in this repository.\n\n"
        "## Project overview\n"
        "- Describe the app, stack, and important directories.\n\n"
        "## Commands\n"
        "- Install: `<command>`\n"
        "- Test: `<command>`\n"
        "- Lint: `<command>`\n"
        "- Run app: `<command>`\n\n"
        "## Coding rules\n"
        "- Keep changes minimal and focused.\n"
        "- Prefer small patches over broad rewrites.\n"
        "- Do not commit secrets or generated artifacts.\n\n"
        "## Verification\n"
        "- Run the smallest relevant test first.\n"
        "- Then run the full test suite when practical.\n",
        encoding="utf-8",
    )
    return path
