from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from .config import HarnessConfig
from .utils import iter_text_files, read_text


_WORD = re.compile(r"[A-Za-z0-9_./:-]+")


def chunk_text(text: str, *, max_chars: int = 2200, overlap: int = 300) -> list[str]:
    text = text.replace("\r\n", "\n")
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks


@dataclass(slots=True)
class RagHit:
    path: str
    chunk_id: int
    score: float
    text: str


class RagStore:
    """Tiny SQLite FTS5 RAG store.

    This deliberately avoids embedding models to keep memory low on 16 GB Macs.
    It is lexical, fast, private, and good enough for code/docs retrieval.
    """

    def __init__(self, cfg: HarnessConfig) -> None:
        self.cfg = cfg
        self.db_path = cfg.rag_db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self):
        return sqlite3.connect(str(self.db_path))

    def _init_db(self) -> None:
        with self._connect() as con:
            con.execute("CREATE TABLE IF NOT EXISTS chunks(path TEXT, chunk_id INTEGER, text TEXT, PRIMARY KEY(path, chunk_id))")
            try:
                con.execute("CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(path, text, content='chunks', content_rowid='rowid')")
            except sqlite3.OperationalError as exc:
                raise RuntimeError("Your Python SQLite build lacks FTS5 support. Use Homebrew Python or skip RAG.") from exc

    def clear(self) -> None:
        with self._connect() as con:
            con.execute("DELETE FROM chunks_fts")
            con.execute("DELETE FROM chunks")

    def index_workspace(self, root: Path | None = None) -> int:
        root = root or self.cfg.workspace
        count = 0
        with self._connect() as con:
            con.execute("DELETE FROM chunks_fts")
            con.execute("DELETE FROM chunks")
            for path in iter_text_files(root):
                rel = str(path.relative_to(self.cfg.workspace))
                text = read_text(path, max_chars=500_000)
                for idx, chunk in enumerate(chunk_text(text)):
                    cur = con.execute("INSERT INTO chunks(path, chunk_id, text) VALUES (?, ?, ?)", (rel, idx, chunk))
                    rowid = cur.lastrowid
                    con.execute("INSERT INTO chunks_fts(rowid, path, text) VALUES (?, ?, ?)", (rowid, rel, chunk))
                    count += 1
        return count

    def search(self, query: str, top_k: int | None = None) -> list[RagHit]:
        top_k = top_k or self.cfg.rag_top_k
        # Convert a natural query into an FTS-friendly OR query.
        terms = [t for t in _WORD.findall(query) if len(t) > 1]
        if not terms:
            return []
        fts_query = " OR ".join(dict.fromkeys(terms[:16]))
        sql = """
            SELECT chunks.path, chunks.chunk_id, bm25(chunks_fts) AS score, chunks.text
            FROM chunks_fts
            JOIN chunks ON chunks_fts.rowid = chunks.rowid
            WHERE chunks_fts MATCH ?
            ORDER BY score
            LIMIT ?
        """
        with self._connect() as con:
            rows = con.execute(sql, (fts_query, top_k)).fetchall()
        return [RagHit(path=r[0], chunk_id=int(r[1]), score=float(r[2]), text=r[3]) for r in rows]

    def context_block(self, query: str, top_k: int | None = None) -> str:
        hits = self.search(query, top_k=top_k)
        if not hits:
            return "[RAG] No relevant indexed context found."
        blocks = []
        for h in hits:
            blocks.append(f"[RAG HIT path={h.path} chunk={h.chunk_id} score={h.score:.4f}]\n{h.text}")
        return "\n\n---\n\n".join(blocks)
