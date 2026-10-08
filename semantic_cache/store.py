"""SQLite-backed storage for cached prompt/response pairs and their embeddings."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np


@dataclass
class CacheEntry:
    id: int
    prompt: str
    response: str
    embedding: np.ndarray
    model: Optional[str]
    created_at: float
    last_used_at: float
    prompt_tokens: int
    completion_tokens: int
    use_count: int


_SCHEMA = """
CREATE TABLE IF NOT EXISTS entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prompt TEXT NOT NULL,
    response TEXT NOT NULL,
    embedding BLOB NOT NULL,
    dim INTEGER NOT NULL,
    model TEXT,
    created_at REAL NOT NULL,
    last_used_at REAL NOT NULL,
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    use_count INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_last_used ON entries(last_used_at);
"""


def _to_blob(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()


def _from_blob(blob: bytes, dim: int) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32).reshape(dim).copy()


class SQLiteStore:
    """Thin SQLite store for cache entries. Embeddings kept as float32 blobs.

    A full vector DB is overkill here; for a portfolio-scale cache (thousands
    of entries) a linear scan over blobs in SQLite is simple, portable, and
    fast enough. Documented limitation: no ANN index, so lookup is O(n).
    """

    def __init__(self, path: str = ":memory:") -> None:
        self.path = path
        self._conn = sqlite3.connect(path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)

    # -- writes -----------------------------------------------------------
    def add(
        self,
        prompt: str,
        response: str,
        embedding: np.ndarray,
        model: Optional[str],
        created_at: float,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> int:
        vec = np.asarray(embedding, dtype=np.float32).ravel()
        cur = self._conn.execute(
            """INSERT INTO entries
               (prompt, response, embedding, dim, model, created_at, last_used_at,
                prompt_tokens, completion_tokens, use_count)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)""",
            (
                prompt,
                response,
                _to_blob(vec),
                vec.shape[0],
                model,
                created_at,
                created_at,
                prompt_tokens,
                completion_tokens,
            ),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def touch(self, entry_id: int, at: float) -> None:
        self._conn.execute(
            "UPDATE entries SET last_used_at = ?, use_count = use_count + 1 WHERE id = ?",
            (at, entry_id),
        )
        self._conn.commit()

    def delete(self, entry_id: int) -> None:
        self._conn.execute("DELETE FROM entries WHERE id = ?", (entry_id,))
        self._conn.commit()

    def delete_ids(self, ids: Iterable[int]) -> None:
        ids = list(ids)
        if not ids:
            return
        self._conn.executemany("DELETE FROM entries WHERE id = ?", [(i,) for i in ids])
        self._conn.commit()

    def clear(self) -> None:
        self._conn.execute("DELETE FROM entries")
        self._conn.commit()

    # -- reads ------------------------------------------------------------
    def count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0])

    def all(self) -> List[CacheEntry]:
        rows = self._conn.execute("SELECT * FROM entries").fetchall()
        return [self._row_to_entry(r) for r in rows]

    def least_recently_used_ids(self, n: int) -> List[int]:
        rows = self._conn.execute(
            "SELECT id FROM entries ORDER BY last_used_at ASC, id ASC LIMIT ?", (n,)
        ).fetchall()
        return [int(r[0]) for r in rows]

    @staticmethod
    def _row_to_entry(row: sqlite3.Row) -> CacheEntry:
        return CacheEntry(
            id=int(row["id"]),
            prompt=row["prompt"],
            response=row["response"],
            embedding=_from_blob(row["embedding"], int(row["dim"])),
            model=row["model"],
            created_at=float(row["created_at"]),
            last_used_at=float(row["last_used_at"]),
            prompt_tokens=int(row["prompt_tokens"]),
            completion_tokens=int(row["completion_tokens"]),
            use_count=int(row["use_count"]),
        )

    def close(self) -> None:
        self._conn.close()
