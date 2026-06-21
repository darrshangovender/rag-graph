<div align="center">

# rag-graph — knowledge-graph-augmented RAG, with real working code

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org)
[![SQLite](https://img.shields.io/badge/SQLite-3-003B57?logo=sqlite&logoColor=white)](https://sqlite.org)
[![Anthropic](https://img.shields.io/badge/Anthropic-Claude-CC785C)](https://anthropic.com)
[![Pydantic](https://img.shields.io/badge/Pydantic-2.7+-E92063?logo=pydantic&logoColor=white)](https://pydantic.dev)
[![Status](https://img.shields.io/badge/Status-Working%20code-blue)](#)

</div>

---

> A small but realistic implementation of **graph-augmented RAG**. Documents go through LLM-based entity + relation extraction; the resulting triples live in a SQLite knowledge graph alongside the chunk embeddings. At query time we run **hybrid retrieval**: vector kNN over chunks PLUS multi-hop graph traversal from the entities mentioned in the query. The retrieved evidence is sent to the LLM with both source chunks and the traversed sub-graph as context.

**Why this exists.** Vector-only RAG is great for "what does this document say about X" but loses on questions that need to **connect** facts across documents — "who funded the company that acquired Y?". Knowledge-graph retrieval handles connection-style questions. Doing both well, in production, in one library, is what this repo demonstrates.

---

## Pipeline

```
Documents
   │
   ▼
┌──────────────────┐    ┌──────────────────────┐
│ Chunker          │    │ Entity + Relation    │  ← LLM call per chunk,
│ (heading-aware)  │    │ Extractor            │    JSON-mode + Pydantic
└────────┬─────────┘    └──────────┬───────────┘
         ▼                         ▼
┌──────────────────┐    ┌──────────────────────┐
│ Embedder         │    │ EntityResolver       │  ← canonicalises aliases
└────────┬─────────┘    └──────────┬───────────┘
         ▼                         ▼
┌─────────────────────────────────────────────┐
│ SQLite store: chunks + entities + edges      │
│   chunks(id, doc_id, text, embedding)        │
│   entities(id, canonical_name, type, aliases)│
│   edges(src_id, predicate, dst_id, source)   │
│   chunk_entities(chunk_id, entity_id)        │
└────────────────────┬────────────────────────┘
                     │
   query ────────────┴──────────────────────────┐
                     │                          │
                     ▼                          ▼
       ┌──────────────────────┐    ┌──────────────────────┐
       │ Vector kNN over      │    │ Entity extraction    │
       │ chunks (top-k)       │    │ on the query         │
       └──────────┬───────────┘    └──────────┬───────────┘
                  │                           ▼
                  │             ┌──────────────────────┐
                  │             │ Graph BFS up to N    │
                  │             │ hops from query ents │
                  │             └──────────┬───────────┘
                  ▼                        ▼
       ┌──────────────────────────────────────────┐
       │ Merge: dedupe chunks, score-blend        │
       │ vec_score + graph_proximity              │
       └──────────────────┬───────────────────────┘
                          ▼
                LLM with chunks + sub-graph
                          ▼
                  Cited answer
```

## Usage

```python
from rag_graph import GraphRAG
from rag_graph.embeddings import OpenAIEmbedder

rag = GraphRAG(
    db_path="kg.db",
    embedder=OpenAIEmbedder(),
    extractor_model="claude-sonnet-4-5",
)

# Ingest
for doc in load_docs():
    rag.ingest(doc_id=doc.id, text=doc.text)

# Ask a connection-style question
result = rag.ask(
    "Which companies founded by ex-Google engineers acquired AI labs in 2024?",
    k=8,         # top-k chunks
    hops=2,      # graph traversal depth
)

print(result.answer)
print(result.sources)         # chunk citations
print(result.entities_used)   # entities the graph traversal touched
```

## Why hybrid retrieval, not pure graph

Knowledge graphs miss everything that wasn't extracted as a triple. Vector retrieval catches fuzzy, narrative, "soft" knowledge that doesn't reduce to (entity, predicate, entity). Doing both means the system handles:

- **Connection questions** ("who funded the company that acquired Y") via graph
- **Soft / narrative questions** ("how did the founder describe the pivot") via vector
- **Lookup questions** ("when was Y founded") via either, with the other as confirmation

The score blender (`vec_score + alpha * graph_proximity`) is tuned per-deployment on a labelled question set.

## Why SQLite + a single embedding column

Same reason as the rest of this stack: one file, zero infra. The vector column is stored as a JSON-encoded blob; cosine similarity is computed in Python. For workloads beyond a few hundred thousand chunks, swap in pgvector — the `GraphStore` interface is intentionally small.

## Entity resolution

Most RAG-graph implementations duplicate "OpenAI" and "Open AI" as separate entities. We avoid this with a two-stage process:

1. **Surface form normalisation** — lowercase, strip punctuation, sort tokens.
2. **Alias index** — when ingesting a new entity, check the normalised form against existing entities. If a match, merge as alias instead of creating a new node.

Production deployments should add LLM-based disambiguation for hard cases (e.g. two unrelated "Apple"s). The hook is there.

## Repo structure

```
.
├── rag_graph/
│   ├── __init__.py
│   ├── core.py            # GraphRAG main class
│   ├── extractor.py       # LLM entity + relation extraction (Pydantic-validated)
│   ├── resolver.py        # entity disambiguation + alias merging
│   ├── store.py           # SQLite store: chunks, entities, edges, joins
│   ├── retriever.py       # hybrid vector + graph retrieval
│   ├── embeddings.py      # pluggable embedder (OpenAI default)
│   └── chunker.py         # heading-aware splitter
├── examples/
│   └── demo.py
├── tests/
│   ├── test_resolver.py
│   ├── test_chunker.py
│   └── test_retriever.py
└── pyproject.toml
```

## Status

- [x] Heading-aware chunker
- [x] LLM entity + relation extractor (Pydantic-validated)
- [x] Entity resolver with alias merging
- [x] SQLite graph store
- [x] Hybrid retriever: vector kNN + graph BFS
- [x] Score blending: `final = vec + alpha * graph_proximity`
- [x] Citation-aware answer generation
- [ ] Multi-hop traversal cost limits (avoid blowing up on dense graphs)
- [ ] pgvector backend (drop-in replacement for SQLite)
- [ ] Eval harness for connection-vs-narrative question types

## Author

Darrshan Govender · Founder, [Agulhas Code](https://agulhascode.co.za)