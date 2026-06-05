from pathlib import Path
import subprocess

from harness.config import HarnessConfig
from harness.cli import _parse_json_array
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


def test_git_status_non_repo(tmp_path: Path):
    cfg = HarnessConfig(workspace=tmp_path)
    result = ToolRegistry(cfg).call("git_status")
    assert result.ok
    assert "not a git repository" in result.output.lower()
