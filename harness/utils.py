from __future__ import annotations

import os
from pathlib import Path


TEXT_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".vue", ".java", ".go", ".rs", ".c", ".cpp", ".h",
    ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".env",
    ".sql", ".html", ".css", ".scss", ".sh", ".ps1", ".bat",
}

DEFAULT_IGNORES = {
    ".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache", ".mypy_cache",
    ".ruff_cache", "dist", "build", ".next", ".vite", ".idea", ".vscode", ".gemma_harness",
}


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def safe_path(workspace: Path, requested: str | Path) -> Path:
    path = (workspace / requested).expanduser().resolve() if not Path(requested).is_absolute() else Path(requested).expanduser().resolve()
    root = workspace.expanduser().resolve()
    if not is_relative_to(path, root):
        raise ValueError(f"Path escapes workspace: {requested}")
    return path


def should_index(path: Path) -> bool:
    parts = set(path.parts)
    if parts & DEFAULT_IGNORES:
        return False
    return path.suffix.lower() in TEXT_EXTENSIONS


def read_text(path: Path, max_chars: int = 50_000) -> str:
    data = path.read_text(encoding="utf-8", errors="replace")
    if len(data) > max_chars:
        return data[:max_chars] + f"\n\n[TRUNCATED at {max_chars} characters]"
    return data


def iter_text_files(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in DEFAULT_IGNORES]
        for filename in filenames:
            path = Path(dirpath) / filename
            if should_index(path):
                yield path
