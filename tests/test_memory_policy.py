from harness.config import HarnessConfig
from harness.memory_policy import (
    compact_context_block,
    estimate_kv_cache_mb,
    get_profile,
    input_context_budget_tokens,
    shell_exports,
)


def test_profiles_include_ollama_kv_exports():
    profile = get_profile("safe")
    exports = shell_exports(profile)
    assert "OLLAMA_FLASH_ATTENTION=1" in exports
    assert "OLLAMA_KV_CACHE_TYPE=q8_0" in exports
    assert "OLLAMA_CONTEXT_LENGTH=4096" in exports


def test_kv_estimate_q4_smaller_than_q8_smaller_than_f16():
    f16 = estimate_kv_cache_mb(8192, kv_cache_type="f16")
    q8 = estimate_kv_cache_mb(8192, kv_cache_type="q8_0")
    q4 = estimate_kv_cache_mb(8192, kv_cache_type="q4_0")
    assert f16 > q8 > q4


def test_context_compaction_respects_budget():
    cfg = HarnessConfig(num_ctx=4096, max_tokens=1024, context_budget_tokens=100)
    long_context = "A" * 5000
    compacted = compact_context_block(long_context, cfg)
    assert "[MEMORY POLICY]" in compacted
    assert len(compacted) < len(long_context)


def test_auto_budget_positive():
    cfg = HarnessConfig(num_ctx=4096, max_tokens=1024)
    assert input_context_budget_tokens(cfg) > 0
