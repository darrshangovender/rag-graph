"""GraphStore: transactions, dimension safety, vector scores, graph queries."""

from __future__ import annotations

import threading

import numpy as np
import pytest

from rag_graph import GraphStore
from rag_graph.store import cosine_similarity


@pytest.fixture
def store(tmp_path):
    s = GraphStore(tmp_path / "kg.db")
    yield s
    s.close()


def add(store, doc="d", idx=0, text="t", emb=(1.0, 0.0)):
    return store.add_chunk(doc_id=doc, chunk_idx=idx, text=text, heading_path=[],
                           embedding=list(emb))


# ---- transactions ------------------------------------------------------------------------------

def test_a_failed_transaction_rolls_everything_back(store):
    with pytest.raises(RuntimeError):
        with store.transaction():
            add(store, text="inside")
            raise RuntimeError("boom")
    assert store.chunk_count() == 0
    assert add(store) == 1 or store.chunk_count() == 1       # and the store still works


def test_nested_transactions_join_the_outer_one(store):
    with pytest.raises(RuntimeError):
        with store.transaction():
            with store.transaction():
                add(store, text="inner")
            add(store, text="outer")
            raise RuntimeError("late failure")
    assert store.chunk_count() == 0


def test_a_committed_transaction_persists(tmp_path):
    with GraphStore(tmp_path / "t.db") as s:
        with s.transaction():
            add(s, text="kept")
    with GraphStore(tmp_path / "t.db") as s:
        assert [c.text for c in s.all_chunks()] == ["kept"]


# ---- dimensions and embeddings ------------------------------------------------------------------

def test_the_first_embedding_fixes_the_dimension(store):
    assert store.dimension() is None
    add(store, emb=(1, 2, 3))
    assert store.dimension() == 3


def test_a_chunk_with_a_different_dimension_is_rejected(store):
    """Mismatched lengths used to score 0.0 silently ('nothing is similar to anything')."""
    add(store, emb=(1, 2, 3))
    with pytest.raises(ValueError, match="3-dimensional"):
        add(store, idx=1, emb=(1, 2))
    assert store.chunk_count() == 1


def test_nan_and_infinite_embeddings_are_rejected(store):
    for bad in ((float("nan"), 0.0), (float("inf"), 1.0)):
        with pytest.raises(ValueError, match="NaN or infinity"):
            add(store, emb=bad)
    assert store.chunk_count() == 0


def test_a_chunk_with_no_embedding_is_stored_but_never_scored(store):
    a = add(store, text="vec", emb=(1.0, 0.0))
    b = store.add_chunk(doc_id="d", chunk_idx=1, text="no vec", heading_path=[], embedding=[])
    assert store.chunk_count() == 2
    assert set(store.vector_scores([1.0, 0.0])) == {a} and b not in store.vector_scores([1.0, 0.0])


def test_query_dimension_is_checked_when_scoring(store):
    add(store, emb=(1, 0, 0))
    with pytest.raises(ValueError, match="dimensions"):
        store.vector_scores([1.0, 0.0])
    with pytest.raises(ValueError, match="NaN"):
        store.vector_scores([float("nan"), 0, 0])


def test_vector_scores_on_an_empty_store(store):
    assert store.vector_scores([1.0, 2.0]) == {}


def test_a_zero_query_vector_scores_zero_everywhere(store):
    add(store, emb=(1, 0))
    assert list(store.vector_scores([0.0, 0.0]).values()) == [0.0]


def test_vector_scores_are_cosines_in_minus_one_to_one(store):
    ids = [add(store, idx=i, emb=e) for i, e in enumerate([(1, 0), (0, 1), (-1, 0), (3, 4)])]
    s = store.vector_scores([1.0, 0.0])
    assert [round(s[i], 6) for i in ids] == [1.0, 0.0, -1.0, 0.6]


def test_the_matrix_cache_is_refreshed_after_adds_and_deletes(store):
    a = add(store, doc="x", emb=(1, 0))
    assert set(store.vector_scores([1.0, 0.0])) == {a}
    b = add(store, doc="y", idx=1, emb=(0, 1))
    assert set(store.vector_scores([1.0, 0.0])) == {a, b}
    store.delete_document("x")
    assert set(store.vector_scores([1.0, 0.0])) == {b}


def test_cosine_similarity_helper():
    assert cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine_similarity([], [1]) == 0.0 and cosine_similarity([0, 0], [1, 1]) == 0.0
    with pytest.raises(ValueError, match="different lengths"):
        cosine_similarity([1, 2], [1, 2, 3])


def test_cosine_matches_numpy():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=16), rng.normal(size=16)
    assert cosine_similarity(a.tolist(), b.tolist()) == pytest.approx(
        float(a @ b / np.linalg.norm(a) / np.linalg.norm(b)))


# ---- documents -------------------------------------------------------------------------------------

def test_delete_document_removes_chunks_edges_and_links_but_keeps_entities(store):
    store.upsert_entity(entity_id=1, canonical="A", type_="Concept", aliases=["A"])
    store.upsert_entity(entity_id=2, canonical="B", type_="Concept", aliases=["B"])
    keep = add(store, doc="keep", emb=(1, 0))
    drop = add(store, doc="drop", idx=0, emb=(0, 1))
    for cid in (keep, drop):
        store.link_chunk_entity(cid, 1)
        store.add_edge(src=1, predicate="rel", dst=2, source_chunk=cid)
    assert store.delete_document("drop") == 1
    assert [c.id for c in store.all_chunks()] == [keep]
    assert store.edge_count() == 1 and store.chunks_for_entities([1]) == [keep]
    assert store.entity(1) and store.entity(2)
    assert store.delete_document("nonexistent") == 0


# ---- edges and entities --------------------------------------------------------------------------------

def test_duplicate_edges_from_the_same_chunk_are_ignored(store):
    c = add(store)
    assert store.add_edge(src=1, predicate="rel", dst=2, source_chunk=c) is True
    assert store.add_edge(src=1, predicate="rel", dst=2, source_chunk=c) is False
    other = add(store, idx=1)
    assert store.add_edge(src=1, predicate="rel", dst=2, source_chunk=other) is True   # new evidence
    assert store.edge_count() == 2


def test_neighbours_include_inverse_edges_in_edge_order_and_honour_a_limit(store):
    c = add(store)
    store.add_edge(src=1, predicate="p1", dst=2, source_chunk=c)
    store.add_edge(src=1, predicate="p2", dst=3, source_chunk=c)
    store.add_edge(src=9, predicate="p3", dst=1, source_chunk=c)
    got = store.neighbours(1)
    assert got == [(2, "p1", 1), (3, "p2", 2), (9, "p3 (inv)", 3)]
    assert store.neighbours(1, limit=2) == got[:2]
    assert store.neighbours(404) == []


def test_edges_among_returns_only_edges_inside_the_set(store):
    c = add(store)
    store.add_edge(src=1, predicate="a", dst=2, source_chunk=c)
    store.add_edge(src=2, predicate="b", dst=3, source_chunk=c)
    store.add_edge(src=3, predicate="c", dst=4, source_chunk=c)
    assert [e["predicate"] for e in store.edges_among([1, 2, 3])] == ["a", "b"]
    assert store.edges_among([]) == [] and store.edges_among([9]) == []
    assert [e["predicate"] for e in store.edges_among([1, 2, 3], limit=1)] == ["a"]


def test_predicates_are_distinct_and_sorted(store):
    c = add(store)
    for i, p in enumerate(["b", "a", "b"]):
        store.add_edge(src=i, predicate=p, dst=i + 10, source_chunk=c)
    assert store.predicates() == ["a", "b"]


def test_entity_upsert_overwrites_by_id_and_sorts_aliases(store):
    store.upsert_entity(entity_id=1, canonical="A", type_="Concept", aliases=["z", "a"])
    store.upsert_entity(entity_id=1, canonical="A2", type_="Person", aliases=["a2"])
    assert store.entity(1) == {"id": 1, "canonical": "A2", "type": "Person", "aliases": ["a2"]}
    store.upsert_entity(entity_id=2, canonical="B", type_="Concept", aliases=["z", "a"])
    assert store.entity(2)["aliases"] == ["a", "z"]
    assert store.entity(99) is None and [e["id"] for e in store.all_entities()] == [1, 2]


def test_chunks_by_entity_and_batch_fetch(store):
    c1, c2 = add(store, idx=0), add(store, idx=1)
    store.link_chunk_entity(c1, 5)
    store.link_chunk_entity(c2, 5)
    store.link_chunk_entity(c2, 6)
    store.link_chunk_entity(c2, 6)                                   # duplicate link ignored
    assert store.chunks_by_entity([5, 6, 7]) == {5: [c1, c2], 6: [c2]}
    assert store.chunks_by_entity([]) == {}
    assert set(store.chunks([c1, c2, 999])) == {c1, c2}
    assert store.chunks([]) == {}
    assert store.chunk(999) is None and store.doc_ids() == ["d"]


def test_edge_lookup(store):
    c = add(store)
    store.add_edge(src=1, predicate="rel", dst=2, source_chunk=c, evidence="because")
    assert store.edge(1) == {"id": 1, "src": 1, "predicate": "rel", "dst": 2, "source_chunk": c,
                             "evidence": "because"}
    assert store.edge(2) is None


# ---- concurrency and persistence --------------------------------------------------------------------------

def test_concurrent_writers_do_not_corrupt_the_store(tmp_path):
    s = GraphStore(tmp_path / "c.db")
    errors = []

    def worker(k):
        try:
            for i in range(25):
                add(s, doc=f"d{k}", idx=i, emb=(float(k + 1), float(i + 1)))
        except Exception as e:  # pragma: no cover
            errors.append(e)

    ts = [threading.Thread(target=worker, args=(k,)) for k in range(6)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert errors == [] and s.chunk_count() == 150
    assert len(s.vector_scores([1.0, 1.0])) == 150
    s.close()


def test_reopening_keeps_the_dimension_and_blocks_a_mismatched_embedder(tmp_path):
    with GraphStore(tmp_path / "r.db") as s:
        add(s, emb=(1, 2, 3, 4))
    with GraphStore(tmp_path / "r.db") as s:
        assert s.dimension() == 4
        with pytest.raises(ValueError):
            add(s, idx=1, emb=(1, 2))


def test_file_store_uses_wal(tmp_path):
    with GraphStore(tmp_path / "w.db") as s:
        assert s._conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


def test_in_memory_store_works():
    with GraphStore(":memory:") as s:
        add(s)
        assert s.chunk_count() == 1
