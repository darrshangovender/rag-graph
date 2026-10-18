"""Store round trips (original tests) and the hybrid retriever."""

import pytest

from rag_graph import GraphStore, HashingEmbedder, HybridRetriever, RuleBasedExtractor
from rag_graph.embeddings import HashEmbedder
from rag_graph.store import cosine_similarity
from tests.conftest import DOCS

# ---- original tests, kept ---------------------------------------------------------------------


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


# ---- helpers ----------------------------------------------------------------------------------

def retr(rag, **kw):
    return HybridRetriever(rag.store, rag.embedder, rag.extractor, rag.resolver, **kw)


def docs_of(results):
    return [r.chunk.doc_id for r in results]


# ---- seeds -------------------------------------------------------------------------------------

def test_seed_entities_are_found_by_whole_word_match(loaded):
    out = retr(loaded).retrieve_detailed("Where is Acme Corp based?")
    names = {loaded.store.entity(e)["canonical"] for e in out.seeds}
    assert names == {"Acme Corp"}


def test_a_short_entity_name_does_not_match_inside_a_longer_word(tmp_path):
    from rag_graph import GraphRAG

    g = GraphRAG(db_path=str(tmp_path / "k.db"))
    g.ingest(doc_id="a", text="Ann Lee founded Annex Works in 2010.")
    out = retr(g).retrieve_detailed("Describe the annual report")        # 'ann' is inside 'annual'
    assert out.seeds == []
    out = retr(g).retrieve_detailed("Who is Ann Lee?")
    assert {g.store.entity(e)["canonical"] for e in out.seeds} == {"Ann Lee"}


def test_seed_matching_is_case_and_punctuation_insensitive_and_alias_aware(loaded):
    for q in ("tell me about ZENITH LTD", "what did Zenith, Ltd. do?", "zenith ltd"):
        out = retr(loaded).retrieve_detailed(q)
        assert {loaded.store.entity(e)["canonical"] for e in out.seeds} == {"Zenith Ltd"}, q


def test_open_ai_matches_the_openai_entity(tmp_path):
    from rag_graph import GraphRAG

    g = GraphRAG(db_path=str(tmp_path / "k.db"))
    g.ingest(doc_id="a", text="Sam Cole founded OpenAI in 2015.")
    out = retr(g).retrieve_detailed("Who started Open AI?")
    assert {g.store.entity(e)["canonical"] for e in out.seeds} == {"OpenAI"}


def test_the_extractor_is_only_a_fallback_when_nothing_is_known_in_the_query(loaded):
    calls = []

    class Spy:
        def extract(self, text):
            calls.append(text)
            return RuleBasedExtractor().extract(text)

    r = HybridRetriever(loaded.store, loaded.embedder, Spy(), loaded.resolver)
    r.retrieve("Where is Acme Corp?")
    assert calls == []
    r.retrieve("What about nothing in particular?")
    assert calls == ["What about nothing in particular?"]


def test_a_failing_fallback_extractor_means_no_seeds_not_an_error(loaded):
    class Boom:
        def extract(self, text):
            raise RuntimeError("api down")

    out = HybridRetriever(loaded.store, loaded.embedder, Boom(), loaded.resolver) \
        .retrieve_detailed("unrelated words")
    assert out.seeds == [] and out.chunks


def test_no_extractor_at_all_is_allowed(loaded):
    out = HybridRetriever(loaded.store, loaded.embedder, None, loaded.resolver) \
        .retrieve_detailed("unrelated words")
    assert out.seeds == []


# ---- vector part ---------------------------------------------------------------------------------

def test_alpha_zero_is_plain_vector_search_and_ignores_the_graph(loaded):
    r = retr(loaded)
    out = r.retrieve_detailed("Which product does Zenith Ltd make?", k=6, alpha=0.0)
    assert out.facts == [] and out.proximity == {}
    assert all(x.graph_proximity == 0.0 for x in out.chunks)
    vec = loaded.store.vector_scores(loaded.embedder.embed("Which product does Zenith Ltd make?"))
    expected = sorted(vec, key=lambda cid: (-max(0.0, vec[cid]), cid))[:6]
    assert [x.chunk.id for x in out.chunks] == expected


def test_vector_scores_equal_a_plain_python_cosine(loaded):
    q = loaded.embedder.embed("Portsea is a port")
    fast = loaded.store.vector_scores(q)
    for c in loaded.store.all_chunks():
        assert fast[c.id] == pytest.approx(cosine_similarity(q, c.embedding), abs=1e-9)


def test_results_are_sorted_by_final_score_with_a_stable_tie_break(loaded):
    out = retr(loaded).retrieve("Acme Corp", k=6)
    keys = [(-r.final_score, r.chunk.id) for r in out]
    assert keys == sorted(keys)


def test_final_score_is_the_documented_blend(loaded):
    for r in retr(loaded, alpha=0.5).retrieve("Who founded Acme Corp?", k=6):
        assert r.final_score == pytest.approx(max(0.0, r.vec_score) + 0.5 * r.graph_proximity)


def test_a_negative_cosine_is_clamped_in_the_blend_but_reported_raw(tmp_path):
    from rag_graph import GraphRAG

    class Opp:
        dimension = 2

        def embed(self, text):
            return [-1.0, 0.0] if text.startswith("query") else [1.0, 0.0]

        def embed_batch(self, texts):
            return [self.embed(t) for t in texts]

    g = GraphRAG(db_path=str(tmp_path / "k.db"), embedder=Opp())
    g.ingest(doc_id="a", text="Some stored text.")
    (r,) = retr(g).retrieve("query text", k=1)
    assert r.vec_score == pytest.approx(-1.0) and r.final_score == 0.0


def test_k_larger_than_the_corpus_returns_everything_once(loaded):
    out = retr(loaded).retrieve("anything", k=100)
    assert len(out) == len(DOCS) and len(set(docs_of(out))) == len(DOCS)


@pytest.mark.parametrize("kw", [{"k": 0}, {"k": -1}, {"hops": -1}, {"alpha": -0.1}])
def test_bad_retrieval_arguments_rejected(loaded, kw):
    with pytest.raises(ValueError):
        retr(loaded).retrieve("Acme Corp", **kw)


def test_bad_constructor_arguments_rejected(loaded):
    for kw in ({"alpha": -1}, {"max_entities": 0}, {"max_degree": 0}, {"max_facts": -1},
               {"traversal": "random"}, {"off_relation_cost": 0}):
        with pytest.raises(ValueError):
            retr(loaded, **kw)


def test_an_empty_store_returns_nothing(tmp_path):
    from rag_graph import GraphRAG

    g = GraphRAG(db_path=str(tmp_path / "k.db"))
    assert retr(g).retrieve("anything") == []


def test_a_query_embedding_of_the_wrong_dimension_is_an_error_not_a_zero_score(loaded):
    """cosine_similarity returned 0.0 for mismatched lengths, hiding a store built with another embedder."""
    r = HybridRetriever(loaded.store, HashingEmbedder(dimension=64), loaded.extractor,
                        loaded.resolver)
    with pytest.raises(ValueError, match="dimensions"):
        r.retrieve("Acme Corp")


# ---- graph part --------------------------------------------------------------------------------------

def entity_id(rag, name):
    return rag.resolver.lookup(name)


def test_proximity_decays_with_hop_distance(loaded):
    out = retr(loaded).retrieve_detailed("Ann Lee", k=6, hops=3)
    prox = {loaded.store.entity(e)["canonical"]: p for e, p in out.proximity.items()}
    assert prox["Ann Lee"] == 1.0
    assert prox["Acme Corp"] == pytest.approx(0.5)
    assert prox["Zenith Ltd"] == pytest.approx(1 / 3)
    assert prox["Widget-9"] == pytest.approx(0.25)


def test_hops_limit_the_traversal(loaded):
    for hops, expected in ((0, {"Ann Lee"}), (1, {"Ann Lee", "Acme Corp"}),
                           (2, {"Ann Lee", "Acme Corp", "Zenith Ltd", "Portsea"})):
        out = retr(loaded).retrieve_detailed("Ann Lee", hops=hops)
        assert {loaded.store.entity(e)["canonical"] for e in out.proximity} == expected, hops


def test_edges_are_followed_in_both_directions(loaded):
    out = retr(loaded).retrieve_detailed("Zenith Ltd", hops=1)
    names = {loaded.store.entity(e)["canonical"] for e in out.proximity}
    assert {"Acme Corp", "Widget-9", "Brightport"} <= names             # acquired (out), makes, hq


def test_the_graph_pulls_in_a_chunk_that_shares_no_words_with_the_question(loaded):
    q = "In which port city is the company founded by Ann Lee based?"
    vector_only = docs_of(retr(loaded).retrieve(q, k=3, alpha=0.0))
    graph = docs_of(retr(loaded).retrieve(q, k=3, alpha=1.0, hops=2))
    assert "d-acme" not in vector_only            # it shares no word with the question
    assert {"d-founder", "d-acme"} <= set(graph)   # the graph reaches it through Ann Lee -> Acme Corp


def test_max_entities_bounds_the_traversal(loaded):
    out = retr(loaded, max_entities=3).retrieve_detailed("Ann Lee", hops=5)
    assert len(out.proximity) <= 3


def test_max_degree_bounds_hub_fan_out(tmp_path):
    from rag_graph import GraphRAG

    g = GraphRAG(db_path=str(tmp_path / "hub.db"))
    g.ingest(doc_id="hub", text="\n".join(f"Hub Corp acquired Target{i}x Ltd." for i in range(40)))
    full = retr(g, max_degree=1000).retrieve_detailed("Hub Corp", hops=1)
    capped = retr(g, max_degree=5).retrieve_detailed("Hub Corp", hops=1)
    assert len(full.proximity) == 41 and len(capped.proximity) == 6


def test_traversal_is_deterministic(loaded):
    a = retr(loaded).retrieve_detailed("Ann Lee", hops=3)
    b = retr(loaded).retrieve_detailed("Ann Lee", hops=3)
    assert (a.proximity, docs_of(a.chunks)) == (b.proximity, docs_of(b.chunks))


def test_graph_facts_connect_the_visited_entities_nearest_first(loaded):
    out = retr(loaded).retrieve_detailed("Ann Lee", hops=3, k=6)
    triples = [(f.subject, f.predicate, f.object) for f in out.facts]
    assert ("Ann Lee", "founded", "Acme Corp") in triples
    assert ("Zenith Ltd", "acquired", "Acme Corp") in triples
    hops = [f.hop for f in out.facts]
    assert hops == sorted(hops)
    assert all(f.evidence for f in out.facts)


def test_max_facts_caps_the_facts(loaded):
    assert len(retr(loaded, max_facts=1).retrieve_detailed("Ann Lee", hops=3).facts) == 1
    assert retr(loaded, max_facts=0).retrieve_detailed("Ann Lee", hops=3).facts == []


def test_relation_guided_traversal_prefers_edges_the_question_names(tmp_path):
    from rag_graph import GraphRAG

    g = GraphRAG(db_path=str(tmp_path / "rg.db"))
    g.ingest(doc_id="a", text="Ann Lee founded Acme Corp.\nAcme Corp is a software company "
                              "headquartered in Portsea.")
    uniform = retr(g, traversal="uniform").retrieve_detailed("Which company did Ann Lee found?",
                                                              hops=2)
    guided = retr(g, traversal="relation_guided").retrieve_detailed(
        "Which company did Ann Lee found?", hops=2)
    names = lambda o: {g.store.entity(e)["canonical"] for e in o.proximity}      # noqa: E731
    assert "Portsea" in names(uniform)                  # reached over the unrelated 'headquartered' edge
    assert "Portsea" not in names(guided)               # that edge costs 2 hops when the question says 'found'
    assert "Acme Corp" in names(guided)


def test_relation_guided_falls_back_to_uniform_when_no_predicate_matches(loaded):
    q = "Ann Lee zzzz"
    a = retr(loaded, traversal="uniform").retrieve_detailed(q, hops=2).proximity
    b = retr(loaded, traversal="relation_guided").retrieve_detailed(q, hops=2).proximity
    assert a == b


def test_relation_stems():
    from rag_graph.retriever import relation_stems, stem

    assert stem("founded") == stem("founds") == "found"
    assert stem("acquires") == stem("acquired") == "acquir"
    assert stem("is") == "is" and stem("sees") == "see"
    assert relation_stems("Who founded the company that acquired X?") >= {"found", "acquir"}


def test_graph_score_is_the_best_proximity_among_a_chunks_entities(loaded):
    out = retr(loaded).retrieve_detailed("Ann Lee", hops=3, k=6)
    by_doc = {r.chunk.doc_id: r for r in out.chunks}
    assert by_doc["d-founder"].graph_proximity == 1.0           # mentions the seed itself
    assert by_doc["d-acme"].graph_proximity == pytest.approx(0.5)
    assert len(by_doc["d-founder"].via_entities) >= 1
    assert by_doc["d-city"].graph_proximity == pytest.approx(1 / 3)       # Portsea is 2 hops away
    assert "d-noise" not in by_doc or by_doc["d-noise"].graph_proximity == 0.0


def test_a_chunk_deleted_between_scoring_and_fetch_is_skipped(loaded, monkeypatch):
    real = loaded.store.chunks
    monkeypatch.setattr(loaded.store, "chunks", lambda ids: {i: c for i, c in real(ids).items()
                                                            if c.doc_id != "d-acme"})
    out = retr(loaded).retrieve("Acme Corp", k=6)
    assert "d-acme" not in docs_of(out)


def test_resolver_changes_are_visible_to_a_long_lived_retriever(tmp_path):
    from rag_graph import GraphRAG

    g = GraphRAG(db_path=str(tmp_path / "k.db"))
    r = g.retriever
    g.ingest(doc_id="a", text="Ann Lee founded Acme Corp.")
    assert r.retrieve_detailed("Ann Lee").seeds                  # the same retriever sees new entities

