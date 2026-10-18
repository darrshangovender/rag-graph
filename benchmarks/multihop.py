"""Vector-only vs graph-augmented retrieval on a bundled SYNTHETIC multi-hop question set.

    python benchmarks/multihop.py

Corpus: `data/corpus.jsonl` (232 short documents about invented people, companies, products, cities
and countries). Questions: `data/questions.jsonl` (80, with the documents that contain the evidence:
20 one-hop, 40 two-hop, 20 three-hop). Everything runs offline: lexical HashingEmbedder, rule-based
extractor, SQLite in a temp file.

Metrics, per question, over the top-k retrieved chunks (each document is one chunk):
  recall@k       share of the gold documents retrieved
  full-support@k every gold document retrieved (the answer is derivable from the context)
  answer-in@k    the answer string appears in the retrieved text (an upper bound for a reader)

"vector-only" is the SAME pipeline with alpha=0 (no graph term). The corpus is templated and the
extractor is rule-based, so extraction here is near-perfect: this measures what the graph adds when
extraction works, and the ablations show what happens when it does not.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from rag_graph import Extraction, GraphRAG, HashingEmbedder, RuleBasedExtractor

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
KS = (3, 5, 8)
HOPS = 2


def load_jsonl(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


class DroppingExtractor:
    """Wraps an extractor and keeps each RELATION with probability `keep` (entities are kept).

    Simulates relation-extraction recall below 100%. Deterministic: a triple is kept or dropped by
    hashing its text, so every run drops the same ones.
    """

    def __init__(self, inner, keep: float) -> None:
        self.inner, self.keep = inner, keep

    def extract(self, text: str) -> Extraction:
        ex = self.inner.extract(text)

        def kept(t) -> bool:
            h = hashlib.sha256(f"{t.subject}|{t.predicate}|{t.object}".encode()).digest()
            return int.from_bytes(h[:4], "big") / 2**32 < self.keep

        return Extraction(entities=ex.entities, triples=[t for t in ex.triples if kept(t)])


@dataclass
class Scores:
    n: int = 0
    recall: float = 0.0
    full: float = 0.0
    answer_in: float = 0.0

    def add(self, recall: float, full: bool, answer_in: bool) -> None:
        self.n += 1
        self.recall += recall
        self.full += full
        self.answer_in += answer_in

    def mean(self) -> dict[str, float]:
        n = max(self.n, 1)
        return {"n": self.n, "recall": self.recall / n, "full": self.full / n,
                "answer_in": self.answer_in / n}


def build(extractor=None, db: Path | None = None) -> GraphRAG:
    rag = GraphRAG(db_path=str(db or Path(tempfile.mkdtemp()) / "kg.db"),
                   embedder=HashingEmbedder(), extractor=extractor or RuleBasedExtractor())
    for d in load_jsonl(DATA / "corpus.jsonl"):
        rag.ingest(doc_id=d["id"], text=d["text"])
    return rag


def evaluate(rag: GraphRAG, questions: list[dict], *, k: int, alpha: float,
             hops: int = HOPS) -> dict[str, Scores]:
    """Scores per hop group ("1", "2", "3") and "all"."""
    out: dict[str, Scores] = defaultdict(Scores)
    for q in questions:
        res = rag.retrieve(q["question"], k=k, hops=hops, alpha=alpha)
        got_docs = {r.chunk.doc_id for r in res.chunks}
        gold = set(q["gold_docs"])
        text = " ".join(r.chunk.text for r in res.chunks).casefold()
        recall = len(gold & got_docs) / len(gold)
        for key in (str(q["hops"]), "all"):
            out[key].add(recall, gold <= got_docs, q["answer"].casefold() in text)
    return out


def pct(x: float) -> str:
    return f"{x:.0%}"


def main_table(rag: GraphRAG, questions: list[dict]) -> str:
    lines = ["| hops | n | metric | " + " | ".join(
        f"vector-only @{k} | graph @{k}" for k in KS) + " |",
        "|---|---|---|" + "---|---|" * len(KS)]
    res = {(k, a): evaluate(rag, questions, k=k, alpha=a)
           for k in KS for a in (0.0, rag.retriever.alpha)}
    for group, label in (("1", "1"), ("2", "2"), ("3", "3"), ("all", "all")):
        for metric, name in (("recall", "recall"), ("full", "full-support"),
                             ("answer_in", "answer in context")):
            cells = []
            for k in KS:
                v = res[(k, 0.0)][group].mean()[metric]
                g = res[(k, rag.retriever.alpha)][group].mean()[metric]
                cells += [pct(v), pct(g)]
            n = res[(KS[0], 0.0)][group].n
            lines.append(f"| {label} | {n} | {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def alpha_table(rag: GraphRAG, questions: list[dict], k: int = 5) -> str:
    lines = [f"| alpha | recall@{k} (2-hop) | full-support@{k} (2-hop) | recall@{k} (3-hop) "
             f"| full-support@{k} (3-hop) | recall@{k} (1-hop) |", "|---|---|---|---|---|---|"]
    for a in (0.0, 0.25, 0.5, 1.0, 2.0):
        r = evaluate(rag, questions, k=k, alpha=a)
        m1, m2, m3 = r["1"].mean(), r["2"].mean(), r["3"].mean()
        lines.append(f"| {a:g} | {pct(m2['recall'])} | {pct(m2['full'])} | {pct(m3['recall'])} "
                     f"| {pct(m3['full'])} | {pct(m1['recall'])} |")
    return "\n".join(lines)


def extraction_table(questions: list[dict], k: int = 5) -> str:
    lines = [f"| relations kept by the extractor | graph recall@{k} (2-hop) "
             f"| graph full-support@{k} (2-hop) | graph full-support@{k} (3-hop) |",
             "|---|---|---|---|"]
    for keep in (1.0, 0.7, 0.4, 0.0):
        rag = build(DroppingExtractor(RuleBasedExtractor(), keep))
        r = evaluate(rag, questions, k=k, alpha=rag.retriever.alpha)
        rag.close()
        m2, m3 = r["2"].mean(), r["3"].mean()
        lines.append(f"| {keep:.0%} | {pct(m2['recall'])} | {pct(m2['full'])} "
                     f"| {pct(m3['full'])} |")
    return "\n".join(lines)


def kind_table(rag: GraphRAG, questions: list[dict], k: int = 5) -> str:
    """Full-support@k per question type: vector-only, uniform graph traversal, relation-guided."""
    lines = ["| question type | hops | n | vector-only | graph (uniform) "
             "| graph (relation-guided) |", "|---|---|---|---|---|---|"]
    kinds = sorted({(q["hops"], q["kind"]) for q in questions})
    original = rag.retriever.traversal
    try:
        for hops, kind in kinds:
            qs = [q for q in questions if q["kind"] == kind]
            row = [pct(evaluate(rag, qs, k=k, alpha=0.0)["all"].mean()["full"])]
            for mode in ("uniform", "relation_guided"):
                rag.retriever.traversal = mode
                scored = evaluate(rag, qs, k=k, alpha=rag.retriever.alpha)["all"].mean()
                row.append(pct(scored["full"]))
            lines.append(f"| {kind} | {hops} | {len(qs)} | " + " | ".join(row) + " |")
    finally:
        rag.retriever.traversal = original
    return "\n".join(lines)


def mode_summary(rag: GraphRAG, questions: list[dict], k: int = 5) -> str:
    lines = [f"| retrieval | recall@{k} (2-hop) | full-support@{k} (2-hop) | recall@{k} (3-hop) "
             f"| full-support@{k} (3-hop) | full-support@{k} (all 80) |",
             "|---|---|---|---|---|---|"]
    original = rag.retriever.traversal
    try:
        for label, alpha, mode in (("vector-only", 0.0, "uniform"),
                                   ("graph, uniform traversal", rag.retriever.alpha, "uniform"),
                                   ("graph, relation-guided traversal", rag.retriever.alpha,
                                    "relation_guided")):
            rag.retriever.traversal = mode
            r = evaluate(rag, questions, k=k, alpha=alpha)
            m2, m3, ma = r["2"].mean(), r["3"].mean(), r["all"].mean()
            lines.append(
                f"| {label} | {pct(m2['recall'])} | {pct(m2['full'])} | {pct(m3['recall'])} "
                f"| {pct(m3['full'])} | {pct(ma['full'])} |"
            )
    finally:
        rag.retriever.traversal = original
    return "\n".join(lines)


def hops_table(rag: GraphRAG, questions: list[dict], k: int = 5) -> str:
    lines = [f"| traversal depth | full-support@{k} (2-hop) | full-support@{k} (3-hop) |",
             "|---|---|---|"]
    for hops in (0, 1, 2, 3):
        r = evaluate(rag, questions, k=k, alpha=rag.retriever.alpha, hops=hops)
        lines.append(f"| {hops} | {pct(r['2'].mean()['full'])} | {pct(r['3'].mean()['full'])} |")
    return "\n".join(lines)


def main() -> int:
    questions = load_jsonl(DATA / "questions.jsonl")
    docs = load_jsonl(DATA / "corpus.jsonl")
    rag = build()
    print(f"{len(docs)} synthetic documents, {len(questions)} questions; "
          f"{rag.store.edge_count()} graph edges, {len(rag.resolver)} entities; "
          f"alpha={rag.retriever.alpha}, hops={HOPS}\n")
    print("1. vector-only (alpha=0) vs graph-augmented, by number of hops the question needs")
    print(main_table(rag, questions))
    print("\n1b. traversal mode (k=5)")
    print(mode_summary(rag, questions))
    print("\n1c. by question type (full-support@5)")
    print(kind_table(rag, questions))
    print("\n2. blend weight alpha (graph traversal depth 2, k=5)")
    print(alpha_table(rag, questions))
    print("\n3. traversal depth (k=5; depth 0 means seeds only)")
    print(hops_table(rag, questions))
    print("\n4. extraction recall: what if the extractor misses relations (k=5)")
    print(extraction_table(questions))
    rag.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
