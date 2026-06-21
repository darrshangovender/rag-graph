"""Tests for hybrid retriever — uses the toy HashEmbedder so we never hit a real API."""

from rag_graph.embeddings import HashEmbedder
from rag_graph.resolver import EntityResolver
from rag_graph.store import GraphStore


def test_store_writes_and_reads_chunks(tmp_path):
    store = GraphStore(tmp_path / "kg.db")
    cid = store.add_chunk(doc_id="d1", chunk_idx=0, text="hello", heading_path=["A"], embedding=[0.1, 0.2])
    chunks = store.all_chunks()
    assert len(chunks) == 1
    assert chunks[0].id == cid
    assert chunks[0].text == "hello"
    assert chunks[0].heading_path == ["A"]


def test_edges_neighbours_work(tmp_path):
    store = GraphStore(tmp_path / "kg.db")
    store.upsert_entity(entity_id=1, canonical="A", type_="Concept", aliases=["A"])
    store.upsert_entity(entity_id=2, canonical="B", type_="Concept", aliases=["B"])
    store.upsert_entity(entity_id=3, canonical="C", type_="Concept", aliases=["C"])
    chunk_id = store.add_chunk(doc_id="d", chunk_idx=0, text="x", heading_path=[], embedding=[])
    store.add_edge(src=1, predicate="rel", dst=2, source_chunk=chunk_id)
    store.add_edge(src=2, predicate="rel", dst=3, source_chunk=chunk_id)
    nbrs = [n[0] for n in store.neighbours(2)]
    assert 1 in nbrs
    assert 3 in nbrs


def test_hash_embedder_is_deterministic():
    e = HashEmbedder(dimension=8)
    a = e.embed("hello")
    b = e.embed("hello")
    assert a == b
    assert len(a) == 8