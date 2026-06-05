from pathlib import Path
import subprocess

from harness.config import HarnessConfig
from harness.cli import _parse_json_array
from harness.llm import LocalLLMClient
from harness.repo_context import create_agents_template, load_repo_instructions
from harness.patching import extract_unified_diff
from harness.tools import ToolRegistry


def test_agents_template_and_load(tmp_path: Path):
    path = create_agents_template(tmp_path)
    assert path.name == "AGENTS.md"
    loaded = load_repo_instructions(tmp_path).as_context(tmp_path)
    assert "REPOSITORY INSTRUCTIONS" in loaded
    assert "AGENTS.md" in loaded


def test_extract_unified_diff_from_fence():
    raw = """Here is patch:\n```diff\ndiff --git a/a.txt b/a.txt\n--- a/a.txt\n+++ b/a.txt\n@@ -1 +1 @@\n-old\n+new\n```"""
    diff = extract_unified_diff(raw)
    assert "diff --git" in diff
    assert "+new" in diff


def test_parse_json_array_from_fence():
    raw = """```json\n[{\"tool\": \"read_file\", \"args\": {\"path\": \"README.md\"}}]\n```"""
    actions = _parse_json_array(raw)
    assert actions[0]["tool"] == "read_file"


def test_manifest_includes_codex_like_tools(tmp_path: Path):
    cfg = HarnessConfig(workspace=tmp_path, allow_write=True)
    manifest = ToolRegistry(cfg).tool_manifest()
    assert "repo_instructions" in manifest
    assert "git_checkpoint" in manifest
    assert "patch_apply" in manifest


def test_ollama_payload_disables_thinking_by_default(monkeypatch, tmp_path: Path):
    captured = {}

    def fake_post_json(self, url, payload, headers=None):
        captured["payload"] = payload
        return {"message": {"content": "OK"}}

    monkeypatch.setattr(LocalLLMClient, "_post_json", fake_post_json)
    cfg = HarnessConfig(workspace=tmp_path, provider="ollama")
    LocalLLMClient(cfg).chat([{"role": "user", "content": "hi"}])

    assert captured["payload"]["think"] is False


def test_gemma4_fast_profile_sets_runtime_defaults(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("GEMMA_PROFILE", "gemma4-fast")
    monkeypatch.setenv("GEMMA_WORKSPACE", str(tmp_path))
    cfg = HarnessConfig.from_env()

    assert cfg.profile == "gemma4-fast"
    assert cfg.model == "gemma4:12b"
    assert cfg.num_ctx == 4096
    assert cfg.max_tokens == 512
    assert cfg.candidates == 1
    assert cfg.debate_rounds == 0
    assert cfg.ollama_think is False


def test_profile_allows_explicit_env_override(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("GEMMA_PROFILE", "gemma4-fast")
    monkeypatch.setenv("GEMMA_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("GEMMA_MAX_TOKENS", "96")
    cfg = HarnessConfig.from_env()

    assert cfg.profile == "gemma4-fast"
    assert cfg.max_tokens == 96


def test_lmstudio_provider_defaults_to_openai_port(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("GEMMA_PROVIDER", "lmstudio")
    monkeypatch.setenv("GEMMA_WORKSPACE", str(tmp_path))
    cfg = HarnessConfig.from_env()

    assert cfg.provider == "lmstudio"
    assert cfg.base_url == "http://localhost:1234/v1"


def test_provider_override_replaces_profile_base_url(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("GEMMA_PROFILE", "gemma4-fast")
    monkeypatch.setenv("GEMMA_PROVIDER", "lmstudio")
    monkeypatch.setenv("GEMMA_WORKSPACE", str(tmp_path))
    cfg = HarnessConfig.from_env()

    assert cfg.profile == "gemma4-fast"
    assert cfg.provider == "lmstudio"
    assert cfg.base_url == "http://localhost:1234/v1"


def test_mlx_provider_defaults_to_mlx_server_port(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("GEMMA_PROVIDER", "mlx")
    monkeypatch.setenv("GEMMA_WORKSPACE", str(tmp_path))
    cfg = HarnessConfig.from_env()

    assert cfg.provider == "mlx"
    assert cfg.base_url == "http://localhost:8080/v1"


def test_openai_compatible_provider_aliases(monkeypatch, tmp_path: Path):
    calls = []

    def fake_openai(self, messages, *, temperature, max_tokens):
        calls.append((messages, temperature, max_tokens))
        return None

    monkeypatch.setattr(LocalLLMClient, "_chat_openai_compatible", fake_openai)
    for provider in ("lmstudio", "mlx"):
        LocalLLMClient(HarnessConfig(workspace=tmp_path, provider=provider)).chat([{"role": "user", "content": "hi"}])

    assert len(calls) == 2


def test_git_status_non_repo(tmp_path: Path):
    cfg = HarnessConfig(workspace=tmp_path)
    result = ToolRegistry(cfg).call("git_status")
    assert result.ok
    assert "not a git repository" in result.output.lower()
