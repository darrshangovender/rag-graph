"""Offline demo: ingest a small synthetic corpus, ask a connection-style question.

    python examples/demo.py

No API key and no network: a lexical hashing embedder, the rule-based extractor and the extractive
answerer (it quotes the best-matching sentences and prints the graph path; it does not generate).
The documents below are INVENTED (names and facts are made up). To use real models pass
`embedder=OpenAIEmbedder()`, `extractor_model="claude-sonnet-4-5"` and
`answer_model="claude-sonnet-4-5"`.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from rag_graph import GraphRAG

DOCS = {
    "doc-1": """\
# Brindle Labs
Brindle Labs is a mapping company headquartered in Halmouth. Brindle Labs makes Kestrel-4.
""",
    "doc-2": """\
# Tamsin Velcott
Tamsin Velcott is an engineer. Tamsin Velcott founded Brindle Labs in 2011.
""",
    "doc-3": """\
# Deal note
Orvane Systems acquired Brindle Labs in 2019.
""",
    "doc-4": """\
# Industry note
Which city a company is headquartered in matters to its staff, and a company that moves city
loses people.
""",
}

QUESTION = "In which city is the company founded by Tamsin Velcott headquartered?"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=None)
    args = ap.parse_args()
    db = args.db or Path(tempfile.mkdtemp()) / "demo_kg.db"
    with GraphRAG(db_path=str(db)) as rag:
        for doc_id, text in DOCS.items():
            rep = rag.ingest(doc_id=doc_id, text=text)
            print(f"ingested {doc_id}: {rep.chunks} chunk(s), {rep.triples} relation(s)")
        print(f"\nQ: {QUESTION}")
        for label, alpha in (("vector only", 0.0), ("vector + graph", None)):
            r = rag.ask(QUESTION, k=2, hops=2, alpha=alpha)
            print(f"\n--- {label} (top 2 chunks) ---")
            print(r.answer)
            print("retrieved:", ", ".join(f"{s['doc_id']}" for s in r.sources))
            if r.graph_facts:
                print("graph facts:")
                for f in r.graph_facts:
                    print(f"  {f['subject']} --{f['predicate']}--> {f['object']} "
                          f"(from chunk {f['source_chunk']}, hop {f['hop']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
