"""SQLite store for chunks, entities, edges and the chunk-entity join table.

  - chunks: one row per chunk; the embedding is JSON text (readable with any sqlite client).
    Retrieval builds a normalised numpy matrix from them once and reuses it until the table changes.
  - entities: canonical name, type, aliases (JSON).
  - edges: (src, predicate, dst, source_chunk) with an evidence quote; unique per source chunk.
  - chunk_entities: which entities a chunk mentions.

Writes inside `transaction()` are atomic; the connection is guarded by a lock and uses WAL with a
busy timeout. The first non-empty embedding fixes the store's dimension; a different dimension is
rejected rather than silently scoring 0.0 (it used to).
"""

from __future__ import annotations

import json
import math
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id       TEXT NOT NULL,
    chunk_idx    INTEGER NOT NULL,
    text         TEXT NOT NULL,
    heading_path TEXT NOT NULL DEFAULT '[]',
    embedding    TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);

CREATE TABLE IF NOT EXISTS entities (
    id        INTEGER PRIMARY KEY,
    canonical TEXT NOT NULL,
    type      TEXT NOT NULL,
    aliases   TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_entities_canonical ON entities(canonical);

CREATE TABLE IF NOT EXISTS edges (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    src_entity_id  INTEGER NOT NULL,
    predicate      TEXT NOT NULL,
    dst_entity_id  INTEGER NOT NULL,
    source_chunk   INTEGER NOT NULL,
    evidence       TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src_entity_id);
CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst_entity_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_edges_unique
    ON edges(src_entity_id, predicate, dst_entity_id, source_chunk);

CREATE TABLE IF NOT EXISTS chunk_entities (
    chunk_id  INTEGER NOT NULL,
    entity_id INTEGER NOT NULL,
    PRIMARY KEY(chunk_id, entity_id)
);
CREATE INDEX IF NOT EXISTS idx_chunk_entities_entity ON chunk_entities(entity_id);

CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


@dataclass
class Chunk:
    id: int
    doc_id: str
    chunk_idx: int
    text: str
    heading_path: list[str]
    embedding: list[float]
    score: float = 0.0


class GraphStore:
    def __init__(self, path: str | Path = "kg.db", *, timeout: float = 30.0) -> None:
        self.path = str(path)
        self._lock = threading.RLock()
        self._depth = 0
        self._conn = sqlite3.connect(
            self.path, check_same_thread=False, timeout=timeout, isolation_level=None,
        )
        with self._lock:
            if self.path != ":memory:":
                self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute(f"PRAGMA busy_timeout={int(timeout * 1000)}")
            self._conn.executescript(SCHEMA)
        self._matrix_key: tuple[int, int] | None = None
        self._matrix: tuple[np.ndarray, np.ndarray] | None = None

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """All-or-nothing group of writes. Re-entrant: an inner block joins the outer one."""
        with self._lock:
            outermost = self._depth == 0
            if outermost:
                self._conn.execute("BEGIN IMMEDIATE")
            self._depth += 1
            try:
                yield
            except BaseException:
                self._depth -= 1
                if outermost:
                    self._conn.execute("ROLLBACK")
                raise
            else:
                self._depth -= 1
                if outermost:
                    self._conn.execute("COMMIT")

    # ---- writes ----

    def add_chunk(self, *, doc_id: str, chunk_idx: int, text: str, heading_path: list[str],
                  embedding: list[float]) -> int:
        with self._lock, self.transaction():
            self._check_dimension(embedding)
            cur = self._conn.execute(
                "INSERT INTO chunks (doc_id, chunk_idx, text, heading_path, embedding) "
                "VALUES (?, ?, ?, ?, ?)",
                (doc_id, chunk_idx, text, json.dumps(heading_path), json.dumps(list(embedding))),
            )
            return int(cur.lastrowid)

    def _check_dimension(self, embedding: list[float]) -> None:
        if not embedding:
            return                                  # "no embedding": excluded from vector search
        if not all(math.isfinite(x) for x in embedding):
            raise ValueError("embedding contains NaN or infinity")
        row = self._conn.execute("SELECT value FROM meta WHERE key = 'dimension'").fetchone()
        if row is None:
            self._conn.execute("INSERT INTO meta (key, value) VALUES ('dimension', ?)",
                               (str(len(embedding)),))
        elif int(row[0]) != len(embedding):
            raise ValueError(
                f"embedding has {len(embedding)} dimensions but this store holds "
                f"{row[0]}-dimensional vectors (a different embedder was used to build it)"
            )

    def upsert_entity(self, *, entity_id: int, canonical: str, type_: str,
                      aliases: list[str]) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO entities (id, canonical, type, aliases) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET canonical=excluded.canonical, type=excluded.type, "
                "aliases=excluded.aliases",
                (entity_id, canonical, type_, json.dumps(sorted(aliases))),
            )

    def add_edge(self, *, src: int, predicate: str, dst: int, source_chunk: int,
                 evidence: str = "") -> bool:
        """Insert an edge; returns False if the same edge from the same chunk already exists."""
        with self._lock:
            cur = self._conn.execute(
                "INSERT OR IGNORE INTO edges (src_entity_id, predicate, dst_entity_id, "
                "source_chunk, evidence) VALUES (?, ?, ?, ?, ?)",
                (src, predicate, dst, source_chunk, evidence),
            )
            return cur.rowcount > 0

    def link_chunk_entity(self, chunk_id: int, entity_id: int) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO chunk_entities (chunk_id, entity_id) VALUES (?, ?)",
                (chunk_id, entity_id),
            )

    def delete_document(self, doc_id: str) -> int:
        """Remove a document's chunks and everything derived from them. Returns chunks removed.

        Entities stay (other documents may mention them, and their ids must not be reissued).
        """
        with self._lock, self.transaction():
            ids = [r[0] for r in self._conn.execute(
                "SELECT id FROM chunks WHERE doc_id = ?", (doc_id,))]
            for chunk_id in ids:
                self._conn.execute("DELETE FROM chunk_entities WHERE chunk_id = ?", (chunk_id,))
                self._conn.execute("DELETE FROM edges WHERE source_chunk = ?", (chunk_id,))
            self._conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            return len(ids)

    # ---- reads ----

    @staticmethod
    def _chunk_from(row: tuple) -> Chunk:
        return Chunk(id=row[0], doc_id=row[1], chunk_idx=row[2], text=row[3],
                     heading_path=json.loads(row[4]), embedding=json.loads(row[5]))

    def all_chunks(self) -> list[Chunk]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, doc_id, chunk_idx, text, heading_path, embedding FROM chunks "
                "ORDER BY id").fetchall()
        return [self._chunk_from(r) for r in rows]

    def chunk(self, chunk_id: int) -> Chunk | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT id, doc_id, chunk_idx, text, heading_path, embedding FROM chunks "
                "WHERE id = ?", (chunk_id,)).fetchone()
        return self._chunk_from(row) if row else None

    def chunks(self, chunk_ids: list[int]) -> dict[int, Chunk]:
        """Several chunks in one query, keyed by id (missing ids are absent)."""
        if not chunk_ids:
            return {}
        out: dict[int, Chunk] = {}
        with self._lock:
            for i in range(0, len(chunk_ids), 500):
                part = chunk_ids[i:i + 500]
                rows = self._conn.execute(
                    "SELECT id, doc_id, chunk_idx, text, heading_path, embedding FROM chunks "
                    f"WHERE id IN ({','.join('?' * len(part))})", part).fetchall()
                out.update({r[0]: self._chunk_from(r) for r in rows})
        return out

    def chunk_count(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])

    def doc_ids(self) -> list[str]:
        with self._lock:
            return [r[0] for r in self._conn.execute(
                "SELECT DISTINCT doc_id FROM chunks ORDER BY doc_id")]

    def data_version(self) -> int:
        """SQLite's `data_version`: changes whenever ANOTHER connection (this process or another)
        commits to the database, and never for this connection's own writes. A cheap staleness
        check for state cached from the database, such as the entity resolver."""
        with self._lock:
            return int(self._conn.execute("PRAGMA data_version").fetchone()[0])

    def dimension(self) -> int | None:
        with self._lock:
            row = self._conn.execute("SELECT value FROM meta WHERE key = 'dimension'").fetchone()
        return int(row[0]) if row else None

    def vector_scores(self, query: list[float]) -> dict[int, float]:
        """Cosine similarity of `query` to every chunk that has an embedding.

        Raises ValueError if the query's dimension differs from the store's. The unit-normalised
        matrix is built once and reused until the chunk table changes.
        """
        ids, matrix = self._embedding_matrix()
        if len(ids) == 0:
            return {}
        q = np.asarray(query, dtype=float)
        if q.shape != (matrix.shape[1],):
            raise ValueError(
                f"query embedding has {q.shape[0] if q.ndim else 0} dimensions, "
                f"the store holds {matrix.shape[1]}"
            )
        if not np.all(np.isfinite(q)):
            raise ValueError("query embedding contains NaN or infinity")
        norm = float(np.linalg.norm(q))
        scores = (matrix @ (q / norm)) if norm > 0 else np.zeros(len(ids))
        return {int(i): float(s) for i, s in zip(ids, scores, strict=True)}

    def _embedding_matrix(self) -> tuple[np.ndarray, np.ndarray]:
        with self._lock:
            key = self._conn.execute("SELECT COUNT(*), COALESCE(MAX(id), 0) FROM chunks").fetchone()
            key = (int(key[0]), int(key[1]))
            if self._matrix is not None and self._matrix_key == key:
                return self._matrix
            rows = self._conn.execute(
                "SELECT id, embedding FROM chunks WHERE embedding != '[]' ORDER BY id").fetchall()
        ids = np.array([r[0] for r in rows], dtype=np.int64)
        if not rows:
            matrix = np.zeros((0, 0))
        else:
            matrix = np.array([json.loads(r[1]) for r in rows], dtype=float)
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            matrix = np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)
        self._matrix_key, self._matrix = key, (ids, matrix)
        return self._matrix

    def chunks_for_entities(self, entity_ids: list[int]) -> list[int]:
        if not entity_ids:
            return []
        with self._lock:
            rows = self._conn.execute(
                "SELECT DISTINCT chunk_id FROM chunk_entities WHERE entity_id IN "
                f"({','.join('?' * len(entity_ids))}) ORDER BY chunk_id", list(entity_ids),
            ).fetchall()
        return [r[0] for r in rows]

    def chunks_by_entity(self, entity_ids: list[int]) -> dict[int, list[int]]:
        """entity id -> ids of the chunks that mention it (one query)."""
        out: dict[int, list[int]] = {}
        if not entity_ids:
            return out
        with self._lock:
            for i in range(0, len(entity_ids), 500):
                part = list(entity_ids[i:i + 500])
                for eid, cid in self._conn.execute(
                    "SELECT entity_id, chunk_id FROM chunk_entities WHERE entity_id IN "
                    f"({','.join('?' * len(part))}) ORDER BY entity_id, chunk_id", part,
                ):
                    out.setdefault(eid, []).append(cid)
        return out

    def neighbours(self, entity_id: int, *, limit: int | None = None) -> list[tuple[int, str, int]]:
        """Entities one hop away: (other_entity_id, predicate, edge_id), outgoing edges first.

        `limit` caps the result (in edge-id order) so a hub entity cannot fan out to the graph.
        """
        with self._lock:
            out_rows = self._conn.execute(
                "SELECT dst_entity_id, predicate, id FROM edges "
                "WHERE src_entity_id = ? ORDER BY id",
                (entity_id,)).fetchall()
            in_rows = self._conn.execute(
                "SELECT src_entity_id, predicate, id FROM edges "
                "WHERE dst_entity_id = ? ORDER BY id",
                (entity_id,)).fetchall()
        out = [(r[0], r[1], r[2]) for r in out_rows] + [(r[0], r[1] + " (inv)", r[2])
                                                          for r in in_rows]
        return out if limit is None else out[:limit]

    def predicates(self) -> list[str]:
        """Distinct edge predicates in the graph, sorted."""
        with self._lock:
            return [r[0] for r in self._conn.execute(
                "SELECT DISTINCT predicate FROM edges ORDER BY predicate")]

    def edges_among(self, entity_ids: list[int], *, limit: int | None = None) -> list[dict]:
        """Edges whose two endpoints are both in `entity_ids`, in edge-id order."""
        if not entity_ids:
            return []
        marks = ",".join("?" * len(entity_ids))
        sql = ("SELECT id, src_entity_id, predicate, dst_entity_id, source_chunk, evidence "
               "FROM edges "
               f"WHERE src_entity_id IN ({marks}) AND dst_entity_id IN ({marks}) ORDER BY id")
        with self._lock:
            rows = self._conn.execute(sql, list(entity_ids) * 2).fetchall()
        rows = rows if limit is None else rows[:limit]
        return [{"id": r[0], "src": r[1], "predicate": r[2], "dst": r[3], "source_chunk": r[4],
                 "evidence": r[5]} for r in rows]

    def all_entities(self) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, canonical, type, aliases FROM entities ORDER BY id").fetchall()
        return [{"id": r[0], "canonical": r[1], "type": r[2], "aliases": json.loads(r[3])}
                for r in rows]

    def entity(self, entity_id: int) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT id, canonical, type, aliases FROM entities WHERE id = ?",
                (entity_id,)).fetchone()
        if not row:
            return None
        return {"id": row[0], "canonical": row[1], "type": row[2], "aliases": json.loads(row[3])}

    def edge(self, edge_id: int) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT id, src_entity_id, predicate, dst_entity_id, source_chunk, evidence "
                "FROM edges WHERE id = ?", (edge_id,)).fetchone()
        if not row:
            return None
        return {"id": row[0], "src": row[1], "predicate": row[2], "dst": row[3],
                "source_chunk": row[4], "evidence": row[5]}

    def edge_count(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0])

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> GraphStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine of two vectors; 0.0 if either is empty or zero.

    Raises ValueError on a length mismatch: it used to return 0.0, which hid a store built with
    a different embedder as "no chunk is similar to anything".
    """
    if not a or not b:
        return 0.0
    if len(a) != len(b):
        raise ValueError(f"vectors have different lengths ({len(a)} vs {len(b)})")
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
