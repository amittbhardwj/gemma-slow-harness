from pathlib import Path

from harness.config import HarnessConfig
from harness import macos_control as mac
from harness.tools import ToolRegistry


def test_applescript_quote_handles_quotes():
    assert mac._applescript_quote('hello "world"') == '"hello \\"world\\""'


def test_hotkey_dry_run():
    result = mac.hotkey("command,space", dry_run=True)
    assert result.ok
    assert result.command is not None
    assert "osascript" in result.command[0]


def test_mac_tools_disabled_by_default(tmp_path: Path):
    cfg = HarnessConfig(workspace=tmp_path)
    tools = ToolRegistry(cfg)
    result = tools.call("mac_click", x=10, y=20)
    assert not result.ok
    assert "disabled" in result.output.lower()


def test_mac_open_defaults_to_dry_run_when_enabled(tmp_path: Path):
    cfg = HarnessConfig(workspace=tmp_path, allow_mac_control=True)
    result = ToolRegistry(cfg).call("mac_open_app", app_name="Safari")
    assert result.ok
    assert "DRY RUN" in result.output


def test_mac_manifest_exposes_flags(tmp_path: Path):
    cfg = HarnessConfig(workspace=tmp_path, allow_mac_control=True, allow_mac_screenshot=True)
    tools = ToolRegistry(cfg)
    manifest = tools.tool_manifest()
    assert "mac_click" in manifest
    assert "mac_screenshot" in manifest
    assert "command,space" in manifest
