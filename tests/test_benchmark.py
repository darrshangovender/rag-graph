"""The bundled synthetic benchmark: the data is sound and the README's claims are what it measures."""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, ROOT / relpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


bench = load("bench_multihop", "benchmarks/multihop.py")
make_data = load("make_data", "data/make_data.py")

DOCS = bench.load_jsonl(ROOT / "data" / "corpus.jsonl")
QUESTIONS = bench.load_jsonl(ROOT / "data" / "questions.jsonl")
README = (ROOT / "README.md").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def rag():
    g = bench.build()
    yield g
    g.close()


# ---- the data ---------------------------------------------------------------------------------

def test_committed_data_is_exactly_what_the_generator_produces():
    docs, qs = make_data.build()
    assert docs == DOCS and qs == QUESTIONS


def test_sizes_and_hop_mix():
    assert len(DOCS) == 232 and len(QUESTIONS) == 80
    assert {h: sum(q["hops"] == h for q in QUESTIONS) for h in (1, 2, 3)} == {1: 20, 2: 40, 3: 20}


def test_ids_are_unique_and_gold_documents_exist():
    ids = [d["id"] for d in DOCS]
    assert len(set(ids)) == len(ids)
    assert len({q["id"] for q in QUESTIONS}) == len(QUESTIONS)
    for q in QUESTIONS:
        assert q["gold_docs"] and set(q["gold_docs"]) <= set(ids)
        assert len(q["gold_docs"]) == q["hops"], q["id"]


def test_every_answer_is_stated_in_the_gold_documents():
    by_id = {d["id"]: d["text"] for d in DOCS}
    for q in QUESTIONS:
        gold_text = " ".join(by_id[g] for g in q["gold_docs"])
        assert q["answer"] in gold_text, (q["id"], q["answer"])


def test_no_answer_is_stated_in_a_non_gold_document_except_hubs():
    """A question is answerable from its gold documents; an answer string appearing elsewhere would
    let 'answer in context' succeed by accident. Cities/countries are shared hubs and exempt."""
    by_id = {d["id"]: d["text"] for d in DOCS}
    for q in QUESTIONS:
        if q["kind"] in ("hq", "founder>hq", "acquirer>hq", "founder>hq>country"):
            continue
        others = [t for i, t in by_id.items() if i not in q["gold_docs"]]
        stray = [t for t in others if q["answer"] in t]
        # a product is named in its company doc and its product doc; a founder only in one doc
        assert len(stray) <= 1, (q["id"], q["answer"])


def test_questions_do_not_contain_their_answers():
    for q in QUESTIONS:
        assert q["answer"] not in q["question"], q["id"]


def test_the_data_is_labelled_synthetic():
    readme = (ROOT / "data" / "README.md").read_text(encoding="utf-8")
    assert "SYNTHETIC" in readme and "invented" in readme.lower()
    assert "SYNTHETIC" in (ROOT / "data" / "make_data.py").read_text(encoding="utf-8")


def test_the_rule_based_extractor_reads_the_corpus_as_designed(rag):
    assert rag.store.edge_count() == 60 + 60 + 60 + 20 + 12          # founded, hq, makes, acquired, located
    preds = rag.store.predicates()
    assert preds == ["acquired", "founded", "headquartered in", "located in", "makes"]


# ---- the metric code -------------------------------------------------------------------------------

def test_evaluate_computes_recall_full_support_and_answer_in_context(rag):
    q = [{"question": "In which city is the company founded by X headquartered?", "hops": 2,
          "gold_docs": [QUESTIONS[0]["gold_docs"][0], "does-not-exist"], "answer": "zzz-nope"}]
    s = bench.evaluate(rag, q, k=5, alpha=0.0)["all"].mean()
    assert s["n"] == 1 and s["recall"] in (0.0, 0.5) and s["full"] == 0.0 and s["answer_in"] == 0.0


def test_scores_accumulate_per_hop_group(rag):
    r = bench.evaluate(rag, QUESTIONS[:20], k=5, alpha=0.0)
    assert set(r) == {"1", "all"} and r["1"].n == 20 == r["all"].n


def test_dropping_extractor_is_deterministic_and_respects_its_extremes():
    from rag_graph import RuleBasedExtractor

    text = "Ann Lee founded Acme Corp. Big Co acquired Acme Corp. Acme Corp makes Gizmo-2."
    inner = RuleBasedExtractor()
    full = inner.extract(text)
    assert bench.DroppingExtractor(inner, 1.0).extract(text) == full
    none = bench.DroppingExtractor(inner, 0.0).extract(text)
    assert none.triples == [] and none.entities == full.entities
    half = bench.DroppingExtractor(inner, 0.5)
    assert half.extract(text) == half.extract(text)


# ---- the claims ------------------------------------------------------------------------------------------

def test_one_hop_questions_are_solved_by_both_so_the_graph_costs_nothing_there(rag):
    for alpha in (0.0, rag.retriever.alpha):
        r = bench.evaluate(rag, QUESTIONS, k=5, alpha=alpha)["1"].mean()
        assert r["recall"] == 1.0 and r["full"] == 1.0


def test_vector_only_cannot_fully_support_a_multi_hop_question_here(rag):
    """By construction: the second document shares no distinctive words with the question."""
    r = bench.evaluate(rag, QUESTIONS, k=5, alpha=0.0)
    assert r["2"].mean()["full"] == 0.0 and r["3"].mean()["full"] == 0.0


def test_the_graph_adds_recall_and_full_support_on_multi_hop_questions(rag):
    v = bench.evaluate(rag, QUESTIONS, k=5, alpha=0.0)
    g = bench.evaluate(rag, QUESTIONS, k=5, alpha=rag.retriever.alpha)
    for hop in ("2", "3"):
        assert g[hop].mean()["recall"] > v[hop].mean()["recall"] + 0.15
        assert g[hop].mean()["full"] > v[hop].mean()["full"] + 0.3
    assert g["all"].mean()["answer_in"] > v["all"].mean()["answer_in"] + 0.2


def test_more_retrieved_chunks_never_hurt_recall(rag):
    prev = 0.0
    for k in (1, 3, 5, 8, 12):
        r = bench.evaluate(rag, QUESTIONS, k=k, alpha=rag.retriever.alpha)["all"].mean()["recall"]
        assert r >= prev - 1e-9
        prev = r


def test_without_relations_the_graph_has_nothing_to_add():
    rag0 = bench.build(bench.DroppingExtractor(bench.RuleBasedExtractor(), 0.0))
    try:
        r = bench.evaluate(rag0, QUESTIONS, k=5, alpha=rag0.retriever.alpha)
        assert r["2"].mean()["full"] == 0.0 and r["3"].mean()["full"] == 0.0
    finally:
        rag0.close()


def test_the_weak_question_types_are_really_weak_as_the_readme_says(rag):
    for kind in ("acquirer>founder", "founder>hq>country"):
        qs = [q for q in QUESTIONS if q["kind"] == kind]
        assert bench.evaluate(rag, qs, k=5, alpha=rag.retriever.alpha)["all"].mean()["full"] <= 0.1


# ---- the README quotes the script --------------------------------------------------------------------------

def test_readme_tables_are_what_the_benchmark_prints(rag):
    for table in (bench.main_table(rag, QUESTIONS), bench.mode_summary(rag, QUESTIONS),
                  bench.kind_table(rag, QUESTIONS), bench.alpha_table(rag, QUESTIONS),
                  bench.hops_table(rag, QUESTIONS)):
        assert table in README


def test_readme_extraction_table_is_what_the_benchmark_prints():
    assert bench.extraction_table(QUESTIONS) in README


def test_readme_names_the_corpus_size_and_labels_it_synthetic():
    assert "232" in README and "SYNTHETIC" in README and "80 questions" in README
    assert json.loads(json.dumps(QUESTIONS[0]))["id"] == "q001"


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_gitignore_does_not_hide_any_shipped_file(tmp_path):
    """`data/*` re-included only data formats, so data/make_data.py (imported by this module and
    by `make data`) never reached a clone: a fresh checkout failed at test collection."""
    shutil.copy(ROOT / ".gitignore", tmp_path / ".gitignore")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    skip = {"__pycache__", ".pytest_cache", ".ruff_cache", ".git"}
    shipped = [
        p.relative_to(ROOT).as_posix()
        for p in ROOT.rglob("*")
        if p.is_file() and not skip & set(p.relative_to(ROOT).parts)
        and not any(part.endswith(".egg-info") for part in p.parts)
    ]
    assert "data/make_data.py" in shipped
    r = subprocess.run(["git", "check-ignore", "--no-index", "--stdin"], cwd=tmp_path,
                       input="\n".join(shipped).encode(), capture_output=True)
    ignored = r.stdout.decode().split()
    assert ignored == [], f"git-ignored: {ignored}"
