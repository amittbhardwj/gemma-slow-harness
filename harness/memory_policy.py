from __future__ import annotations

import os
import platform
import subprocess
from dataclasses import dataclass
from typing import Iterable

from .config import HarnessConfig


AVG_CHARS_PER_TOKEN = 4


@dataclass(frozen=True, slots=True)
class MemoryProfile:
    name: str
    description: str
    num_ctx: int
    max_tokens: int
    candidates: int
    debate_rounds: int
    kv_cache_type: str
    flash_attention: bool
    notes: tuple[str, ...]


PROFILES: dict[str, MemoryProfile] = {
    "safe": MemoryProfile(
        name="safe",
        description="Best daily profile for a Mac mini M4 base model with 16 GB unified memory.",
        num_ctx=4096,
        max_tokens=1024,
        candidates=3,
        debate_rounds=1,
        kv_cache_type="q8_0",
        flash_attention=True,
        notes=(
            "Keeps active KV cache small.",
            "Uses q8 KV as the default quality/memory compromise.",
            "Best for coding help, small RAG packs, and iterative agent loops.",
        ),
    ),
    "strong": MemoryProfile(
        name="strong",
        description="Heavier reasoning profile; close other memory-heavy apps first.",
        num_ctx=8192,
        max_tokens=1536,
        candidates=4,
        debate_rounds=2,
        kv_cache_type="q8_0",
        flash_attention=True,
        notes=(
            "Good for harder tasks when memory pressure stays green/yellow.",
            "Still avoids 16K/32K context on 16 GB machines.",
            "Candidates are sequential, so they mostly increase time rather than peak RAM.",
        ),
    ),
    "emergency": MemoryProfile(
        name="emergency",
        description="Lowest-memory profile for avoiding swap or red memory pressure.",
        num_ctx=4096,
        max_tokens=768,
        candidates=2,
        debate_rounds=1,
        kv_cache_type="q4_0",
        flash_attention=True,
        notes=(
            "Use when Activity Monitor shows high memory pressure or swap growth.",
            "q4 KV can introduce subtle quality loss; prefer q8 when possible.",
            "Good fallback for long sessions on 16 GB Macs.",
        ),
    ),
    "rag-heavy": MemoryProfile(
        name="rag-heavy",
        description="For larger documents/projects: keep context modest and rely on cold SQLite memory/RAG.",
        num_ctx=8192,
        max_tokens=1024,
        candidates=3,
        debate_rounds=1,
        kv_cache_type="q8_0",
        flash_attention=True,
        notes=(
            "Designed around retrieval and summaries instead of giant active context.",
            "Use gemma-harness index and skills rather than 16K+ prompts.",
            "Best choice for repo/document workflows on base Mac hardware.",
        ),
    ),
}


def profile_names() -> list[str]:
    return list(PROFILES)


def get_profile(name: str) -> MemoryProfile:
    key = name.strip().lower()
    if key not in PROFILES:
        valid = ", ".join(profile_names())
        raise KeyError(f"Unknown memory profile {name!r}. Valid profiles: {valid}")
    return PROFILES[key]


def env_exports_for_profile(profile: MemoryProfile) -> dict[str, str]:
    return {
        "GEMMA_NUM_CTX": str(profile.num_ctx),
        "GEMMA_MAX_TOKENS": str(profile.max_tokens),
        "GEMMA_CANDIDATES": str(profile.candidates),
        "GEMMA_DEBATE_ROUNDS": str(profile.debate_rounds),
        "OLLAMA_FLASH_ATTENTION": "1" if profile.flash_attention else "0",
        "OLLAMA_KV_CACHE_TYPE": profile.kv_cache_type,
        # Ollama reads this when it starts. It is separate from GEMMA_NUM_CTX,
        # which the harness sends per request where provider support exists.
        "OLLAMA_CONTEXT_LENGTH": str(profile.num_ctx),
    }


def shell_exports(profile: MemoryProfile) -> str:
    lines = [f"export {k}={v}" for k, v in env_exports_for_profile(profile).items()]
    return "\n".join(lines)


def estimate_kv_cache_mb(num_ctx: int, *, kv_cache_type: str = "f16") -> int:
    """Rough Gemma 4 12B KV-cache estimate.

    Approximation based on 30 layers, 4 KV heads, head_dim 256, K+V.
    Real usage depends on backend, sliding-window implementation, batch size,
    metadata buffers, and Metal/Ollama allocator overhead.
    """
    bytes_per_value = {
        "f16": 2.0,
        "q8_0": 1.0,
        "q4_0": 0.5,
    }.get(kv_cache_type, 2.0)
    bytes_per_token = 30 * 4 * 256 * 2 * bytes_per_value
    return round((bytes_per_token * num_ctx) / (1024 * 1024))


def estimate_total_memory_gb(
    cfg: HarnessConfig, *, kv_cache_type: str | None = None, num_ctx: int | None = None
) -> tuple[float, list[str]]:
    """Very rough resident-memory estimate for a local Mac run."""
    kv_type = kv_cache_type or cfg.kv_cache_type
    ctx = num_ctx or cfg.num_ctx
    kv_gb = estimate_kv_cache_mb(ctx, kv_cache_type=kv_type) / 1024
    model_gb = 7.6 if "12b" in cfg.model.lower() else 4.5
    runtime_gb = 0.7
    harness_gb = 0.3
    os_floor_gb = 4.0
    tool_allowance_gb = 1.0
    total = model_gb + kv_gb + runtime_gb + harness_gb + os_floor_gb + tool_allowance_gb
    notes = [
        f"model≈{model_gb:.1f} GB",
        f"KV({kv_type}, ctx={ctx})≈{kv_gb:.2f} GB",
        "runtime≈0.7 GB",
        "harness/RAG≈0.3 GB",
        "macOS/background floor≈4.0 GB",
        "Python/tool allowance≈1.0 GB",
    ]
    return total, notes


def input_context_budget_tokens(cfg: HarnessConfig) -> int:
    """Budget for retrieved context, not including system/prompt overhead.

    Keep enough room for the current user prompt, planner output, answer tokens,
    and final judge/repair prompts.
    """
    explicit = cfg.context_budget_tokens
    if explicit > 0:
        return explicit
    reserve = max(cfg.max_tokens + 1200, int(cfg.num_ctx * 0.35))
    return max(800, cfg.num_ctx - reserve)


def approx_tokens(text: str) -> int:
    return max(1, len(text) // AVG_CHARS_PER_TOKEN) if text else 0


def _trim_middle(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    if max_chars <= 200:
        return text[:max_chars]
    head = int(max_chars * 0.62)
    tail = max_chars - head - 120
    return (
        text[:head].rstrip()
        + "\n\n[...context compacted by memory policy; use RAG/search for exact omitted details...]\n\n"
        + text[-tail:].lstrip()
    )


def compact_context_block(context: str, cfg: HarnessConfig) -> str:
    """Keep the active prompt inside a RAM/KV-friendly context budget.

    This does not delete persistent memory. It only limits the hot context sent to
    the model. The full source remains available through SQLite RAG, learning DB,
    files, and skills.
    """
    budget_tokens = input_context_budget_tokens(cfg)
    budget_chars = budget_tokens * AVG_CHARS_PER_TOKEN
    if len(context) <= budget_chars:
        return context
    compacted = _trim_middle(context, budget_chars)
    header = (
        "[MEMORY POLICY]\n"
        f"Active context was compacted to about {budget_tokens} tokens to reduce KV-cache RAM.\n"
        "Cold memory remains in SQLite RAG/learning/skills; retrieve exact details when needed.\n\n"
    )
    return header + compacted


def current_ollama_env() -> dict[str, str | None]:
    keys = ["OLLAMA_FLASH_ATTENTION", "OLLAMA_KV_CACHE_TYPE", "OLLAMA_CONTEXT_LENGTH"]
    return {k: os.getenv(k) for k in keys}


def mac_memory_snapshot() -> str:
    """Best-effort macOS memory snapshot. Returns a human string and never raises."""
    if platform.system().lower() != "darwin":
        return "Memory snapshot is only implemented for macOS."
    try:
        vm_stat = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=3, check=False).stdout.strip()
        pressure = subprocess.run(["memory_pressure"], capture_output=True, text=True, timeout=5, check=False).stdout.strip()
        lines = []
        if pressure:
            lines.append("memory_pressure:\n" + "\n".join(pressure.splitlines()[:12]))
        if vm_stat:
            lines.append("vm_stat:\n" + "\n".join(vm_stat.splitlines()[:12]))
        return "\n\n".join(lines) if lines else "No macOS memory output captured."
    except Exception as exc:  # pragma: no cover - platform-dependent
        return f"Could not read macOS memory snapshot: {exc}"


def format_profile(profile: MemoryProfile) -> str:
    lines = [
        f"Profile: {profile.name}",
        profile.description,
        "",
        "Settings:",
        f"  GEMMA_NUM_CTX={profile.num_ctx}",
        f"  GEMMA_MAX_TOKENS={profile.max_tokens}",
        f"  GEMMA_CANDIDATES={profile.candidates}",
        f"  GEMMA_DEBATE_ROUNDS={profile.debate_rounds}",
        f"  OLLAMA_FLASH_ATTENTION={1 if profile.flash_attention else 0}",
        f"  OLLAMA_KV_CACHE_TYPE={profile.kv_cache_type}",
        f"  OLLAMA_CONTEXT_LENGTH={profile.num_ctx}",
        "",
        "Notes:",
    ]
    lines.extend(f"  - {note}" for note in profile.notes)
    return "\n".join(lines)


def format_profiles(profiles: Iterable[MemoryProfile]) -> str:
    return "\n\n".join(format_profile(p) for p in profiles)
