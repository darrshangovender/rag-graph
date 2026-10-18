"""GraphRAG end to end: ingest, persistence, atomicity, answers, citations."""

from __future__ import annotations

import pytest

from rag_graph import (
    AnswerInput,
    ExtractiveAnswerer,
    GraphRAG,
    HashingEmbedder,
    LLMAnswerer,
    RuleBasedExtractor,
)
from rag_graph.answerers import Passage
from tests.conftest import DOCS


def names(rag):
    return {e["canonical"] for e in rag.store.all_entities()}


# ---- ingest ----------------------------------------------------------------------------------------

def test_defaults_are_fully_offline(rag):
    assert isinstance(rag.embedder, HashingEmbedder)
    assert isinstance(rag.extractor, RuleBasedExtractor)
    assert isinstance(rag.answerer, ExtractiveAnswerer)


def test_ingest_report_counts(rag):
    rep = rag.ingest(doc_id="d", text=DOCS["d-founder"])
    assert (rep.chunks, rep.triples, rep.dropped_triples, rep.replaced_chunks) == (1, 1, 0, 0)
    assert rep.extraction_errors == [] and int(rep) == 1 and rep.doc_id == "d"
    assert rep.entities == 2


def test_ingesting_the_corpus_builds_the_expected_graph(loaded):
    assert names(loaded) == {"Ann Lee", "Acme Corp", "Zenith Ltd", "Brightport", "Widget-9",
                             "Portsea", "Norland"}
    assert loaded.store.edge_count() == 6
    assert loaded.store.chunk_count() == len(DOCS)


def test_re_ingesting_a_document_replaces_it_instead_of_duplicating(loaded):
    before = (loaded.store.chunk_count(), loaded.store.edge_count())
    rep = loaded.ingest(doc_id="d-founder", text=DOCS["d-founder"])
    assert rep.replaced_chunks == 1
    assert (loaded.store.chunk_count(), loaded.store.edge_count()) == before


def test_re_ingesting_with_changed_text_drops_the_old_facts(loaded):
    loaded.ingest(doc_id="d-founder", text="# Ann Lee\nAnn Lee founded Other Co in 2011.")
    triples = {(f.subject, f.predicate, f.object)
               for f in loaded.retrieve("Ann Lee", hops=2, k=10).facts}
    assert ("Ann Lee", "founded", "Other Co") in triples
    assert ("Ann Lee", "founded", "Acme Corp") not in triples


def test_empty_and_blank_documents_store_nothing(rag):
    for text in ("", "  \n\n"):
        rep = rag.ingest(doc_id="e", text=text)
        assert rep.chunks == 0
    assert rag.store.chunk_count() == 0


def test_bad_doc_ids_and_questions_are_rejected(rag):
    for bad in ("", "  ", None, 5):
        with pytest.raises(ValueError, match="doc_id"):
            rag.ingest(doc_id=bad, text="x")  # type: ignore[arg-type]
    for q in ("", "  ", None):
        with pytest.raises(ValueError, match="question"):
            rag.ask(q)  # type: ignore[arg-type]


def test_a_long_document_is_split_into_several_chunks(rag):
    text = "# Big\n" + "\n\n".join(f"Paragraph {i} " + "filler words here " * 40 for i in range(8))
    rep = GraphRAG(db_path=":memory:", max_chars=500).ingest(doc_id="big", text=text)
    assert rep.chunks > 3 and rag.ingest(doc_id="big", text=text).chunks >= 1


def test_the_breadcrumb_is_embedded_so_a_heading_only_name_is_findable(rag):
    rag.ingest(doc_id="p", text="# Zebulon Quark\nA physicist who studies tiny things.")
    rag.ingest(doc_id="q", text="# Other\nAn unrelated paragraph about tables.")
    top = rag.retrieve("Zebulon Quark", k=1, alpha=0.0).chunks[0]
    assert top.chunk.doc_id == "p" and "Zebulon" not in top.chunk.text


# ---- persistence (the id-hijack bug) ----------------------------------------------------------------

def test_ids_continue_across_restarts_and_nothing_is_overwritten(tmp_path):
    """The resolver restarted at id 1 in every process, so a second ingest reissued ids that belonged
    to other entities, overwrote them, and left their old edges pointing at the wrong nodes."""
    db = str(tmp_path / "kg.db")
    a = GraphRAG(db_path=db)
    a.ingest(doc_id="d1", text=DOCS["d-founder"])
    first = {e["id"]: e["canonical"] for e in a.store.all_entities()}
    a.close()

    b = GraphRAG(db_path=db)                                  # a new process, same file
    b.ingest(doc_id="d2", text=DOCS["d-zenith"])
    after = {e["id"]: e["canonical"] for e in b.store.all_entities()}
    assert all(after[i] == n for i, n in first.items()), "an existing entity was overwritten"
    assert len(after) == len(first) + 3                       # Zenith Ltd, Brightport, Widget-9
    b.close()


def test_the_graph_works_in_a_fresh_process_without_re_ingesting(tmp_path):
    """The surface index came from the in-memory resolver, so a reopened database had no seeds, no
    traversal and silently degraded to vector search."""
    db = str(tmp_path / "kg.db")
    with GraphRAG(db_path=db) as a:
        for doc_id, text in DOCS.items():
            a.ingest(doc_id=doc_id, text=text)
    with GraphRAG(db_path=db) as b:
        out = b.retrieve("In which port city is the company founded by Ann Lee based?", k=3, hops=2)
        assert out.seeds, "no seed entities in a reopened database"
        assert "d-acme" in {r.chunk.doc_id for r in out.chunks}
        assert out.facts


def test_two_open_instances_on_one_file_never_reissue_an_entity_id(tmp_path):
    """Each GraphRAG built its resolver once, at construction. Two objects open on one file (two
    workers, or a worker and a notebook) both started the next id at the same number, so the
    second writer's `upsert_entity` overwrote the first writer's entities while their edges still
    pointed at them."""
    db = str(tmp_path / "kg.db")
    a, b = GraphRAG(db_path=db), GraphRAG(db_path=db)
    a.ingest(doc_id="a1", text="Ann Lee founded Zed Corp.")
    b.ingest(doc_id="b1", text="Bob Ray founded Yarn Inc.")
    a.ingest(doc_id="a2", text="Cy Dun founded Xylo Ltd.")
    assert names(a) == {"Ann Lee", "Zed Corp", "Bob Ray", "Yarn Inc", "Cy Dun", "Xylo Ltd"}
    ids = [e["id"] for e in a.store.all_entities()]
    assert len(ids) == len(set(ids)) == 6 and a.store.edge_count() == 3
    a.close()
    b.close()


def test_an_instance_sees_entities_another_instance_added_after_it_opened(tmp_path):
    """The seed index came from the resolver built at construction, so a long-lived reader
    never found entities that a writer added later and silently fell back to vector search."""
    db = str(tmp_path / "kg.db")
    reader, writer = GraphRAG(db_path=db), GraphRAG(db_path=db)
    writer.ingest(doc_id="w", text="Bob Ray founded Yarn Inc.")
    out = reader.retrieve("What did Bob Ray found?")
    assert out.seeds and {f.object for f in out.facts} == {"Yarn Inc"}
    reader.close()
    writer.close()


def test_concurrent_writers_on_one_file_keep_every_entity(tmp_path):
    import threading

    db = str(tmp_path / "kg.db")
    writers = [GraphRAG(db_path=db) for _ in range(3)]
    errors: list[BaseException] = []

    def work(n: int) -> None:
        try:
            for i in range(8):
                writers[n].ingest(doc_id=f"w{n}-{i}", text=f"Pat{n}x{i} founded Firm{n}x{i}.")
        except BaseException as e:  # noqa: BLE001 - surfaced by the assertion below
            errors.append(e)

    threads = [threading.Thread(target=work, args=(n,)) for n in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert len(names(writers[0])) == 3 * 8 * 2 and writers[0].store.edge_count() == 24
    for w in writers:
        w.close()


def test_two_processes_writing_one_file_keep_every_entity(tmp_path):
    import subprocess
    import sys
    import textwrap

    script = tmp_path / "writer.py"
    script.write_text(textwrap.dedent("""
        import pathlib, sys, time
        from rag_graph import GraphRAG
        who, db, flags = sys.argv[1], sys.argv[2], pathlib.Path(sys.argv[3])
        with GraphRAG(db_path=db) as g:
            (flags / f"ready-{who}").touch()            # both objects exist before either writes
            deadline = time.time() + 60
            while len(list(flags.glob("ready-*"))) < 2 and time.time() < deadline:
                time.sleep(0.02)
            for i in range(6):
                g.ingest(doc_id=f"{who}-{i}", text=f"Pat{who}x{i} founded Firm{who}x{i}.")
    """), encoding="utf-8")
    db = str(tmp_path / "kg.db")
    procs = [subprocess.Popen([sys.executable, str(script), who, db, str(tmp_path)],
                              stderr=subprocess.PIPE, text=True) for who in ("A", "B")]
    errs = [p.communicate(timeout=120)[1] for p in procs]
    assert [p.returncode for p in procs] == [0, 0], errs
    with GraphRAG(db_path=db) as g:
        assert len(names(g)) == 24 and g.store.edge_count() == 12
        ids = [e["id"] for e in g.store.all_entities()]
        assert len(ids) == len(set(ids))


def test_own_writes_do_not_trigger_a_resolver_rebuild(tmp_path):
    g = GraphRAG(db_path=str(tmp_path / "kg.db"))
    g.ingest(doc_id="a", text="Ann Lee founded Zed Corp.")
    resolver = g.resolver
    g.ingest(doc_id="b", text="Bob Ray founded Yarn Inc.")
    g.retrieve("Ann Lee")
    assert g.resolver is resolver
    g.close()


def test_aliases_survive_a_restart(tmp_path):
    db = str(tmp_path / "kg.db")
    with GraphRAG(db_path=db) as a:
        a.ingest(doc_id="1", text="Ann Lee founded OpenAI in 2015.")
        a.ingest(doc_id="2", text="Bo Chan joined Open AI as CEO.")
    with GraphRAG(db_path=db) as b:
        ent = [e for e in b.store.all_entities() if e["canonical"] == "OpenAI"]
        assert len(ent) == 1 and set(ent[0]["aliases"]) == {"OpenAI", "Open AI"}


# ---- atomicity and failure handling ----------------------------------------------------------------------

class FlakyEmbedder(HashingEmbedder):
    def __init__(self, fail_on):
        super().__init__()
        self.fail_on, self.calls = fail_on, 0

    def embed(self, text):
        self.calls += 1
        if self.calls == self.fail_on:
            raise ConnectionError("embedding service down")
        return super().embed(text)


def test_an_embedder_failure_part_way_through_a_document_stores_nothing_of_it(tmp_path):
    g = GraphRAG(db_path=str(tmp_path / "k.db"), embedder=FlakyEmbedder(fail_on=2), max_chars=60)
    text = "# A\nAnn Lee founded Acme Corp.\n\nBig Co acquired Acme Corp.\n\nAcme Corp makes Gizmo-2."
    with pytest.raises(ConnectionError):
        g.ingest(doc_id="doc", text=text)
    assert g.store.chunk_count() == 0 and g.store.edge_count() == 0 and g.store.all_entities() == []


def test_a_failed_re_ingest_keeps_the_previous_version(tmp_path):
    emb = FlakyEmbedder(fail_on=10**9)
    g = GraphRAG(db_path=str(tmp_path / "k.db"), embedder=emb)
    g.ingest(doc_id="doc", text="Ann Lee founded Acme Corp.")
    emb.fail_on, emb.calls = emb.calls + 1, emb.calls
    with pytest.raises(ConnectionError):
        g.ingest(doc_id="doc", text="Different text about Other Co.")
    assert [c.text for c in g.store.all_chunks()] == ["Ann Lee founded Acme Corp."]
    assert g.store.edge_count() == 1


def test_a_failed_ingest_leaves_the_resolver_consistent_with_the_database(tmp_path):
    emb = FlakyEmbedder(fail_on=2)
    g = GraphRAG(db_path=str(tmp_path / "k.db"), embedder=emb, max_chars=40)
    with pytest.raises(ConnectionError):
        g.ingest(doc_id="doc", text="Ann Lee founded Acme Corp.\n\nBig Co acquired Zed Ltd.")
    assert len(g.resolver) == 0
    rep = g.ingest(doc_id="doc", text="Ann Lee founded Acme Corp.")
    assert rep.triples == 1 and {e["id"] for e in g.store.all_entities()} == {1, 2}


class BrokenExtractor:
    def __init__(self):
        self.calls = 0

    def extract(self, text):
        self.calls += 1
        if self.calls == 2:
            raise RuntimeError("model overloaded")
        return RuleBasedExtractor().extract(text)


def test_extraction_failures_are_reported_and_the_chunk_is_still_searchable(tmp_path):
    g = GraphRAG(db_path=str(tmp_path / "k.db"), extractor=BrokenExtractor(), max_chars=45)
    rep = g.ingest(doc_id="d", text="Ann Lee founded Acme Corp.\n\nBig Co acquired Zed Ltd.\n\n"
                                    "Zed Ltd makes Gizmo-2.")
    assert rep.chunks == 3 and len(rep.extraction_errors) == 1
    assert "RuntimeError: model overloaded" in rep.extraction_errors[0]
    assert g.store.chunk_count() == 3                       # the failed chunk is stored anyway


class LyingExtractor:
    def extract(self, text):
        from rag_graph import Entity, Extraction, Triple

        return Extraction(entities=[Entity(name="Ann Lee", type="Person")],
                          triples=[Triple(subject="Ann Lee", predicate="knows", object="Ghost")])


def test_triples_that_refer_to_unlisted_entities_are_counted_not_inserted(tmp_path):
    g = GraphRAG(db_path=str(tmp_path / "k.db"), extractor=LyingExtractor())
    rep = g.ingest(doc_id="d", text="anything at all")
    assert rep.triples == 0 and rep.dropped_triples == 1 and g.store.edge_count() == 0


class PunctuationEntityExtractor:
    def extract(self, text):
        from rag_graph import Entity, Extraction, Triple

        return Extraction(
            entities=[Entity(name="Acme", type="Organisation"), Entity(name="—", type="Other"),
                      Entity(name="Ann Lee", type="Person")],
            triples=[Triple(subject="Ann Lee", predicate="founded", object="Acme"),
                     Triple(subject="—", predicate="is", object="Acme")],
        )


def test_an_entity_name_with_no_word_characters_is_dropped_not_fatal(tmp_path):
    """A model that returned the entity "—" aborted (and rolled back) the whole document."""
    g = GraphRAG(db_path=str(tmp_path / "k.db"), extractor=PunctuationEntityExtractor())
    rep = g.ingest(doc_id="d", text="Ann Lee founded Acme.")
    assert (rep.entities, rep.dropped_entities, rep.triples, rep.dropped_triples) == (2, 1, 1, 1)
    assert g.store.chunk_count() == 1 and names(g) == {"Acme", "Ann Lee"}


def test_an_embedder_with_a_different_dimension_cannot_be_mixed_into_a_store(tmp_path):
    db = str(tmp_path / "k.db")
    GraphRAG(db_path=db, embedder=HashingEmbedder(64)).ingest(doc_id="a", text="First text.")
    other = GraphRAG(db_path=db, embedder=HashingEmbedder(32))
    with pytest.raises(ValueError, match="64-dimensional"):
        other.ingest(doc_id="b", text="Second text.")
    with pytest.raises(ValueError, match="dimensions"):
        other.ask("anything")


# ---- ask ---------------------------------------------------------------------------------------------------

def test_ask_returns_an_answer_sources_entities_and_graph_facts(loaded):
    r = loaded.ask("Which product is made by the company that acquired the company founded by Ann Lee?",
                   k=4, hops=3)
    assert "Widget-9" in "".join(s["text"] for s in r.sources)
    assert {"Ann Lee", "Acme Corp"} <= set(r.entities_used)
    assert any(f["predicate"] == "acquired" for f in r.graph_facts)
    assert r.answer and not r.invalid_citations


def test_source_labels_match_the_passages_the_answerer_saw(loaded):
    """The context labelled chunks [c1], [c2], ... by rank while sources carried database ids, so
    '[c2]' in an answer did not identify any source field. Now sources carry the same labels."""
    spy = SpyAnswerer("Acme [c1] [c2]")
    loaded.answerer = spy
    r = loaded.ask("Who founded Acme Corp?", k=3)
    assert [s["cite"] for s in r.sources] == ["c1", "c2", "c3"]
    assert [(p.label, p.text) for p in spy.inputs[0].passages] == [
        (s["cite"], s["text"]) for s in r.sources]
    assert r.cited == ["c1", "c2"]


class SpyAnswerer:
    def __init__(self, reply="ok"):
        self.reply, self.inputs = reply, []

    def answer(self, inp: AnswerInput) -> str:
        self.inputs.append(inp)
        return self.reply


def test_the_answerer_receives_passages_and_graph_facts(tmp_path, loaded):
    spy = SpyAnswerer()
    loaded.answerer = spy
    loaded.ask("Who founded the company that Zenith Ltd acquired?", k=3, hops=2)
    inp = spy.inputs[0]
    assert [p.label for p in inp.passages] == ["c1", "c2", "c3"]
    assert any("Zenith Ltd --acquired--> Acme Corp" in f for f in inp.facts)
    assert "Graph facts:" in inp.render()


def test_facts_name_the_chunk_they_came_from_when_it_is_in_the_context(loaded):
    spy = SpyAnswerer()
    loaded.answerer = spy
    loaded.ask("Zenith Ltd acquired which company?", k=6, hops=1)
    facts = spy.inputs[0].facts
    assert any(f.endswith("]") and "--acquired-->" in f for f in facts)


def test_alpha_zero_sends_no_graph_facts(loaded):
    spy = SpyAnswerer()
    loaded.answerer = spy
    loaded.ask("Ann Lee", alpha=0.0)
    assert spy.inputs[0].facts == []


def test_invalid_citations_in_the_answer_are_reported(loaded):
    loaded.answerer = SpyAnswerer("It is Acme [c1] and also [c9] and [c2].")
    r = loaded.ask("Acme Corp", k=3)
    assert r.cited == ["c1", "c2"] and r.invalid_citations == ["c9"]


def test_an_answer_with_no_citations(loaded):
    loaded.answerer = SpyAnswerer("No idea.")
    r = loaded.ask("Acme Corp", k=3)
    assert r.cited == [] and r.invalid_citations == []


def test_ask_on_an_empty_store_still_answers(rag):
    r = rag.ask("Anything?")
    assert r.sources == [] and "does not answer" in r.answer


# ---- answerers --------------------------------------------------------------------------------------------------

def passages():
    return [Passage("c1", "d1", "A", "Ann Lee founded Acme Corp in 2010. Unrelated sentence."),
            Passage("c2", "d2", "", "Zenith Ltd acquired Acme Corp in 2020.")]


def test_extractive_answerer_quotes_the_best_sentences_with_labels():
    a = ExtractiveAnswerer(max_sentences=2).answer(
        AnswerInput(question="Who founded Acme Corp?", passages=passages()))
    assert "Ann Lee founded Acme Corp in 2010. [c1]" in a and "[c2]" in a
    assert "Unrelated" not in a


def test_extractive_answerer_with_no_overlap_says_so():
    a = ExtractiveAnswerer().answer(AnswerInput(question="zzz qqq", passages=passages()))
    assert "does not answer" in a


def test_extractive_answerer_appends_the_graph_path():
    a = ExtractiveAnswerer().answer(AnswerInput(
        question="Acme Corp", passages=passages(), facts=["A --x--> B [c1]"]))
    assert a.endswith("Graph path: A --x--> B [c1]")


def test_extractive_answerer_rejects_a_bad_limit():
    with pytest.raises(ValueError):
        ExtractiveAnswerer(max_sentences=0)


def test_llm_answerer_sends_the_rendered_context_with_graph_facts():
    seen = []
    ans = LLMAnswerer("claude-sonnet-4-5",
                      chat=lambda system, user, n: seen.append((system, user, n)) or "reply [c1]")
    out = ans.answer(AnswerInput(question="Q?", passages=passages(), facts=["A --x--> B"]))
    assert out == "reply [c1]"
    system, user, max_tokens = seen[0]
    assert "Graph facts" in system and "Question: Q?" in user
    assert "[c1] doc=d1 | A" in user and "Graph facts:\n  A --x--> B" in user and max_tokens == 1024


def test_render_marks_a_chunk_with_no_heading():
    assert "(no heading)" in AnswerInput(question="q", passages=passages()).render()


def test_extractive_answerer_quotes_a_hard_wrapped_sentence_whole():
    """Single newlines used to split sentences, so the demo quoted "...a company that moves city"
    and dropped "loses people." from a wrapped paragraph."""
    text = "Which city a company is in matters, and a company that moves city\nloses people."
    a = ExtractiveAnswerer().answer(AnswerInput(
        question="Which city?", passages=[Passage("c1", "d1", "", text)]))
    assert a == ("Which city a company is in matters, and a company that moves city loses people."
                 " [c1]")
