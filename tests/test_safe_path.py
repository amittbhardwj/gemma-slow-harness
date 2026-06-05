from pathlib import Path

import pytest

from harness.utils import safe_path


def test_safe_path_allows_workspace(tmp_path: Path):
    assert safe_path(tmp_path, "a.txt") == tmp_path / "a.txt"


def test_safe_path_blocks_escape(tmp_path: Path):
    with pytest.raises(ValueError):
        safe_path(tmp_path, "../outside.txt")
