"""Demo: ingest a small corpus, ask a connection-style question.

Requires either ANTHROPIC_API_KEY or OPENAI_API_KEY.
"""

from rag_graph import GraphRAG
from rag_graph.embeddings import OpenAIEmbedder


DOCS = {
    "doc-1": """\
# DeepMind acquisition
Google acquired DeepMind, a London-based AI research lab, in 2014. The deal was reported at about $500m.
DeepMind was founded by Demis Hassabis, Shane Legg, and Mustafa Suleyman in 2010.
""",
    "doc-2": """\
# Demis Hassabis
Demis Hassabis is a neuroscientist who co-founded DeepMind. He has led DeepMind since the acquisition and is now CEO of Google DeepMind.
""",
    "doc-3": """\
# Mustafa Suleyman after DeepMind
After leaving DeepMind, Mustafa Suleyman co-founded Inflection AI in 2022. In 2024, he joined Microsoft as CEO of Microsoft AI.
""",
}


def main() -> None:
    rag = GraphRAG(
        db_path="demo_kg.db",
        embedder=OpenAIEmbedder(),
        extractor_model="claude-sonnet-4-5",
        answer_model="claude-sonnet-4-5",
    )
    for doc_id, text in DOCS.items():
        n = rag.ingest(doc_id=doc_id, text=text)
        print(f"ingested {doc_id}: {n} chunks")

    question = "Which person co-founded a company that was acquired by Google, and what did they do after?"
    result = rag.ask(question, k=6, hops=2)
    print("\nANSWER\n======")
    print(result.answer)
    print("\nENTITIES USED IN GRAPH TRAVERSAL\n================================")
    for e in result.entities_used:
        print(f"  - {e}")
    print("\nSOURCES\n=======")
    for s in result.sources:
        print(f"  [c{s['chunk_id']}] {s['doc_id']} (vec={s['vec_score']:.2f}, graph={s['graph_proximity']:.2f}, final={s['final_score']:.2f})")


if __name__ == "__main__":
    main()