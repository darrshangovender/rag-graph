"""SQLite store for chunks, entities, edges, and the chunk-entity join table.

Schema rationale:
  - One row per chunk, with the embedding stored as a JSON-encoded float list.
    For larger workloads, swap to pgvector — the interface is small.
  - Entities have a canonical name and a type. Aliases live as a separate column
    JSON-encoded.
  - Edges are (src, predicate, dst, source_chunk_id). Source chunk lets us cite.
  - chunk_entities is the join: which entities are mentioned in which chunks.
    Used by the graph retriever to find chunks containing seed entities.
"""

from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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

CREATE TABLE IF NOT EXISTS chunk_entities (
    chunk_id  INTEGER NOT NULL,
    entity_id INTEGER NOT NULL,
    PRIMARY KEY(chunk_id, entity_id)
);
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
    def __init__(self, path: str | Path = "kg.db"):
        self.path = str(path)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    # ---- writes ----

    def add_chunk(self, *, doc_id: str, chunk_idx: int, text: str, heading_path: list[str], embedding: list[float]) -> int:
        cur = self._conn.execute(
            "INSERT INTO chunks (doc_id, chunk_idx, text, heading_path, embedding) VALUES (?, ?, ?, ?, ?)",
            (doc_id, chunk_idx, text, json.dumps(heading_path), json.dumps(embedding)),
        )
        self._conn.commit()
        return cur.lastrowid

    def upsert_entity(self, *, entity_id: int, canonical: str, type_: str, aliases: list[str]) -> None:
        self._conn.execute(
            "INSERT INTO entities (id, canonical, type, aliases) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET canonical=excluded.canonical, type=excluded.type, aliases=excluded.aliases",
            (entity_id, canonical, type_, json.dumps(sorted(aliases))),
        )
        self._conn.commit()

    def add_edge(self, *, src: int, predicate: str, dst: int, source_chunk: int, evidence: str = "") -> None:
        self._conn.execute(
            "INSERT INTO edges (src_entity_id, predicate, dst_entity_id, source_chunk, evidence) VALUES (?, ?, ?, ?, ?)",
            (src, predicate, dst, source_chunk, evidence),
        )
        self._conn.commit()

    def link_chunk_entity(self, chunk_id: int, entity_id: int) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO chunk_entities (chunk_id, entity_id) VALUES (?, ?)",
            (chunk_id, entity_id),
        )
        self._conn.commit()

    # ---- reads ----

    def all_chunks(self) -> list[Chunk]:
        rows = self._conn.execute("SELECT id, doc_id, chunk_idx, text, heading_path, embedding FROM chunks").fetchall()
        return [Chunk(id=r[0], doc_id=r[1], chunk_idx=r[2], text=r[3], heading_path=json.loads(r[4]), embedding=json.loads(r[5])) for r in rows]

    def chunks_for_entities(self, entity_ids: list[int]) -> list[int]:
        if not entity_ids:
            return []
        placeholders = ",".join("?" * len(entity_ids))
        rows = self._conn.execute(
            f"SELECT DISTINCT chunk_id FROM chunk_entities WHERE entity_id IN ({placeholders})",
            entity_ids,
        ).fetchall()
        return [r[0] for r in rows]

    def neighbours(self, entity_id: int) -> list[tuple[int, str, int]]:
        """All entities one hop away. Returns list of (other_entity_id, predicate, edge_id)."""
        out: list[tuple[int, str, int]] = []
        for row in self._conn.execute("SELECT dst_entity_id, predicate, id FROM edges WHERE src_entity_id = ?", (entity_id,)):
            out.append((row[0], row[1], row[2]))
        for row in self._conn.execute("SELECT src_entity_id, predicate, id FROM edges WHERE dst_entity_id = ?", (entity_id,)):
            out.append((row[0], row[1] + " (inv)", row[2]))
        return out

    def entity(self, entity_id: int) -> dict | None:
        row = self._conn.execute("SELECT id, canonical, type, aliases FROM entities WHERE id = ?", (entity_id,)).fetchone()
        if not row:
            return None
        return {"id": row[0], "canonical": row[1], "type": row[2], "aliases": json.loads(row[3])}

    def edge(self, edge_id: int) -> dict | None:
        row = self._conn.execute(
            "SELECT id, src_entity_id, predicate, dst_entity_id, source_chunk, evidence FROM edges WHERE id = ?",
            (edge_id,),
        ).fetchone()
        if not row:
            return None
        return {"id": row[0], "src": row[1], "predicate": row[2], "dst": row[3], "source_chunk": row[4], "evidence": row[5]}

    def chunk(self, chunk_id: int) -> Chunk | None:
        row = self._conn.execute(
            "SELECT id, doc_id, chunk_idx, text, heading_path, embedding FROM chunks WHERE id = ?",
            (chunk_id,),
        ).fetchone()
        if not row:
            return None
        return Chunk(id=row[0], doc_id=row[1], chunk_idx=row[2], text=row[3], heading_path=json.loads(row[4]), embedding=json.loads(row[5]))

    def close(self) -> None:
        self._conn.close()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)