from pathlib import Path

from harness.config import HarnessConfig
from harness.rag import RagStore, chunk_text


def test_chunk_text_short():
    assert chunk_text("hello") == ["hello"]


def test_chunk_text_long():
    chunks = chunk_text("a" * 5000, max_chars=1000, overlap=100)
    assert len(chunks) > 1
    assert all(len(c) <= 1000 for c in chunks)


def test_reindex_removes_stale_fts_terms(tmp_path: Path):
    first = tmp_path / "first.txt"
    first.write_text("unique_old_token", encoding="utf-8")
    store = RagStore(HarnessConfig(workspace=tmp_path, rag_db_path=tmp_path / ".gemma_harness/rag.sqlite3"))

    assert store.index_workspace(tmp_path) == 1
    assert store.search("unique_old_token")

    first.unlink()
    (tmp_path / "second.txt").write_text("unique_new_token", encoding="utf-8")
    assert store.index_workspace(tmp_path) == 1

    assert store.search("unique_old_token") == []
    assert store.search("unique_new_token")
