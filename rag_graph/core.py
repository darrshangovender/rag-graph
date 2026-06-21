"""GraphRAG — the user-facing class. Wires together extractor, resolver, store, retriever.

Two public methods:
  - ingest(doc_id, text) — chunk, embed, extract, write to store
  - ask(question)        — retrieve, format context, call LLM, return cited answer
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from .chunker import chunk_document
from .embeddings import Embedder, OpenAIEmbedder
from .extractor import EntityRelationExtractor
from .resolver import EntityResolver
from .retriever import HybridRetriever, Retrieved
from .store import GraphStore


@dataclass
class AnswerResult:
    answer: str
    sources: list[dict]            # cited chunks with score breakdown
    entities_used: list[str]       # canonical names traversed in the graph


ANSWER_SYSTEM = """\
You answer questions from supplied context. Rules:
  - Use ONLY facts present in the context.
  - Cite chunks inline like [c1], [c2].
  - When the context includes a sub-graph of related entities, USE it for connection-style answers.
  - If the context does not answer the question, say so plainly. Do not guess.
"""


class GraphRAG:
    def __init__(
        self,
        *,
        db_path: str = "kg.db",
        embedder: Embedder | None = None,
        extractor_model: str = "claude-sonnet-4-5",
        answer_model: str = "claude-sonnet-4-5",
    ):
        self.store = GraphStore(db_path)
        self.embedder = embedder or OpenAIEmbedder()
        self.extractor = EntityRelationExtractor(model=extractor_model)
        self.resolver = EntityResolver()
        self.answer_model = answer_model

    def ingest(self, *, doc_id: str, text: str) -> int:
        """Returns the number of chunks ingested."""
        chunks = chunk_document(text)
        n = 0
        for idx, ch in enumerate(chunks):
            emb = self.embedder.embed(ch.text)
            chunk_id = self.store.add_chunk(
                doc_id=doc_id, chunk_idx=idx, text=ch.text,
                heading_path=ch.heading_path, embedding=emb,
            )
            # Extract + link entities
            try:
                ex = self.extractor.extract(ch.text)
            except Exception:
                # Fail open: a bad extraction shouldn't lose the chunk's vector value
                ex = None
            if ex is not None:
                name_to_id: dict[str, int] = {}
                for ent in ex.entities:
                    eid, is_new = self.resolver.resolve(ent.name, entity_type=ent.type)
                    self.store.upsert_entity(
                        entity_id=eid, canonical=self.resolver.canonical(eid),
                        type_=ent.type, aliases=list(self.resolver.aliases(eid)),
                    )
                    self.store.link_chunk_entity(chunk_id, eid)
                    name_to_id[ent.name] = eid
                for tr in ex.triples:
                    src_id = name_to_id.get(tr.subject)
                    dst_id = name_to_id.get(tr.object)
                    if src_id is None or dst_id is None:
                        continue
                    self.store.add_edge(
                        src=src_id, predicate=tr.predicate, dst=dst_id,
                        source_chunk=chunk_id, evidence=tr.evidence,
                    )
            n += 1
        return n

    def ask(self, question: str, *, k: int = 5, hops: int = 2) -> AnswerResult:
        retriever = HybridRetriever(self.store, self.embedder, self.extractor, self.resolver)
        retrieved = retriever.retrieve(question, k=k, hops=hops)
        context = self._format_context(retrieved)
        answer = self._answer(question, context)
        entities_used = sorted({eid for r in retrieved for eid in r.via_entities})
        canonical_names = [self.store.entity(e)["canonical"] for e in entities_used if self.store.entity(e)]
        return AnswerResult(
            answer=answer,
            sources=[{
                "chunk_id": r.chunk.id, "doc_id": r.chunk.doc_id,
                "heading_path": r.chunk.heading_path, "text": r.chunk.text,
                "vec_score": r.vec_score, "graph_proximity": r.graph_proximity,
                "final_score": r.final_score,
            } for r in retrieved],
            entities_used=canonical_names,
        )

    def _format_context(self, retrieved: list[Retrieved]) -> str:
        lines = []
        for i, r in enumerate(retrieved, 1):
            crumb = " > ".join(r.chunk.heading_path) if r.chunk.heading_path else "(no heading)"
            lines.append(f"[c{i}] doc={r.chunk.doc_id} | {crumb}\n{r.chunk.text}\n")
        return "\n".join(lines)

    def _answer(self, question: str, context: str) -> str:
        provider = "anthropic" if self.answer_model.startswith("claude-") else "openai"
        user_msg = f"Question: {question}\n\nContext:\n{context}"
        if provider == "anthropic":
            from anthropic import Anthropic
            client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            resp = client.messages.create(
                model=self.answer_model, max_tokens=1024,
                system=ANSWER_SYSTEM, messages=[{"role": "user", "content": user_msg}],
            )
            return resp.content[0].text
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        resp = client.chat.completions.create(
            model=self.answer_model, max_tokens=1024,
            messages=[{"role": "system", "content": ANSWER_SYSTEM}, {"role": "user", "content": user_msg}],
        )
        return resp.choices[0].message.content or ""