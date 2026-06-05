from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return float(raw)


@dataclass(slots=True)
class HarnessConfig:
    """Runtime configuration.

    Defaults are intentionally conservative for a Mac mini M4 base model with
    16 GB unified memory. Increase num_ctx, max_tokens, and candidates only
    after confirming your machine stays responsive.
    """

    provider: str = "ollama"  # ollama | openai
    model: str = "gemma4:12b"
    base_url: str = "http://localhost:11434"
    api_key: str = "ollama"

    temperature: float = 0.35
    max_tokens: int = 1536
    num_ctx: int = 8192
    request_timeout_sec: int = 600

    candidates: int = 4
    debate_rounds: int = 2
    require_final_judge_score: int = 80

    workspace: Path = Path.cwd()
    allow_write: bool = False
    allow_shell: bool = False
    tool_timeout_sec: int = 30

    # macOS desktop automation. Disabled by default because it can control
    # mouse/keyboard and may expose screen content.
    allow_mac_control: bool = False
    allow_mac_screenshot: bool = False

    rag_db_path: Path = Path(".gemma_harness/rag.sqlite3")
    rag_top_k: int = 6

    # Self-learning / skills memory. This is retrieval memory, not fine-tuning.
    learning_enabled: bool = True
    learning_db_path: Path = Path(".gemma_harness/learning.sqlite3")
    memory_top_k: int = 5
    skill_top_k: int = 3
    auto_learn_min_score: int = 80

    # Memory/KV policy. Active KV cache should stay in RAM; use SQLite/RAG/skills
    # as cold storage instead of trying to page hot KV cache to disk.
    memory_profile: str = "strong"
    kv_cache_type: str = "q8_0"  # f16 | q8_0 | q4_0 for Ollama/llama.cpp-style runtimes
    flash_attention: bool = True
    context_budget_tokens: int = 0  # 0 = auto-budget from num_ctx/max_tokens

    @classmethod
    def from_env(cls) -> "HarnessConfig":
        workspace = Path(os.getenv("GEMMA_WORKSPACE", os.getcwd())).expanduser().resolve()
        return cls(
            provider=os.getenv("GEMMA_PROVIDER", "ollama").strip().lower(),
            model=os.getenv("GEMMA_MODEL", "gemma4:12b"),
            base_url=os.getenv("GEMMA_BASE_URL", "http://localhost:11434").rstrip("/"),
            api_key=os.getenv("GEMMA_API_KEY", "ollama"),
            temperature=_env_float("GEMMA_TEMPERATURE", 0.35),
            max_tokens=_env_int("GEMMA_MAX_TOKENS", 1536),
            num_ctx=_env_int("GEMMA_NUM_CTX", 8192),
            request_timeout_sec=_env_int("GEMMA_REQUEST_TIMEOUT_SEC", 600),
            candidates=_env_int("GEMMA_CANDIDATES", 4),
            debate_rounds=_env_int("GEMMA_DEBATE_ROUNDS", 2),
            require_final_judge_score=_env_int("GEMMA_MIN_SCORE", 80),
            workspace=workspace,
            allow_write=_env_bool("GEMMA_ALLOW_WRITE", False),
            allow_shell=_env_bool("GEMMA_ALLOW_SHELL", False),
            tool_timeout_sec=_env_int("GEMMA_TOOL_TIMEOUT_SEC", 30),
            allow_mac_control=_env_bool("GEMMA_ALLOW_MAC_CONTROL", False),
            allow_mac_screenshot=_env_bool("GEMMA_ALLOW_MAC_SCREENSHOT", False),
            rag_db_path=workspace / os.getenv("GEMMA_RAG_DB", ".gemma_harness/rag.sqlite3"),
            rag_top_k=_env_int("GEMMA_RAG_TOP_K", 6),
            learning_enabled=_env_bool("GEMMA_LEARNING_ENABLED", True),
            learning_db_path=workspace / os.getenv("GEMMA_LEARNING_DB", ".gemma_harness/learning.sqlite3"),
            memory_top_k=_env_int("GEMMA_MEMORY_TOP_K", 5),
            skill_top_k=_env_int("GEMMA_SKILL_TOP_K", 3),
            auto_learn_min_score=_env_int("GEMMA_AUTO_LEARN_MIN_SCORE", 80),
            memory_profile=os.getenv("GEMMA_MEMORY_PROFILE", "strong").strip().lower(),
            kv_cache_type=os.getenv("OLLAMA_KV_CACHE_TYPE", os.getenv("GEMMA_KV_CACHE_TYPE", "q8_0")).strip(),
            flash_attention=_env_bool("OLLAMA_FLASH_ATTENTION", _env_bool("GEMMA_FLASH_ATTENTION", True)),
            context_budget_tokens=_env_int("GEMMA_CONTEXT_BUDGET_TOKENS", 0),
        )

    def summary(self) -> str:
        return (
            f"provider={self.provider}, model={self.model}, base_url={self.base_url}, "
            f"num_ctx={self.num_ctx}, max_tokens={self.max_tokens}, "
            f"candidates={self.candidates}, debate_rounds={self.debate_rounds}, "
            f"workspace={self.workspace}, allow_write={self.allow_write}, allow_shell={self.allow_shell}, "
            f"allow_mac_control={self.allow_mac_control}, allow_mac_screenshot={self.allow_mac_screenshot}, "
            f"learning_enabled={self.learning_enabled}, memory_top_k={self.memory_top_k}, skill_top_k={self.skill_top_k}, "
            f"memory_profile={self.memory_profile}, kv_cache_type={self.kv_cache_type}, "
            f"flash_attention={self.flash_attention}, context_budget_tokens={self.context_budget_tokens}"
        )
