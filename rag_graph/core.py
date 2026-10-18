"""GraphRAG: the user-facing class. Wires chunker, embedder, extractor, resolver, store, retriever.

  - ingest(doc_id, text)  chunk, embed, extract, resolve, write; atomic per document, idempotent
  - ask(question)         retrieve (vector + graph), build context WITH graph facts, answer, cite

Everything runs offline by default: HashingEmbedder, RuleBasedExtractor, ExtractiveAnswerer. Pass an
`OpenAIEmbedder`, an `EntityRelationExtractor` / `extractor_model=` and an `LLMAnswerer` /
`answer_model=` for real models.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .answerers import Answerer, AnswerInput, ExtractiveAnswerer, LLMAnswerer, Passage
from .chunker import chunk_document
from .embeddings import Embedder, HashingEmbedder
from .extractor import EntityRelationExtractor, Extractor, RuleBasedExtractor
from .resolver import EntityResolver, tokens_of
from .retriever import HybridRetriever, RetrievalOutput
from .store import GraphStore

_CITE = re.compile(r"\[c(\d+)\]")


@dataclass
class IngestReport:
    """What `ingest` did. `int(report)` is the number of chunks (the old return value)."""
    doc_id: str
    chunks: int
    entities: int = 0
    triples: int = 0
    dropped_triples: int = 0          # subject or object missing from the chunk's entity list
    dropped_entities: int = 0         # names with no letter or digit, which cannot be keyed
    replaced_chunks: int = 0          # chunks of an earlier version of this document, removed
    extraction_errors: list[str] = field(default_factory=list)

    def __int__(self) -> int:
        return self.chunks


@dataclass
class AnswerResult:
    answer: str
    sources: list[dict]            # cited chunks with score breakdown
    entities_used: list[str]       # canonical names traversed in the graph
    graph_facts: list[dict] = field(default_factory=list)
    cited: list[str] = field(default_factory=list)              # e.g. ["c1", "c3"], valid only
    invalid_citations: list[str] = field(default_factory=list)  # [cN] markers that do not exist


class GraphRAG:
    def __init__(
        self,
        *,
        db_path: str = "kg.db",
        embedder: Embedder | None = None,
        extractor: Extractor | None = None,
        answerer: Answerer | None = None,
        extractor_model: str | None = None,
        answer_model: str | None = None,
        alpha: float = 0.5,
        sort_tokens: bool = False,
        max_chars: int = 1500,
    ):
        self.store = GraphStore(db_path)
        self.embedder = embedder or HashingEmbedder()
        self.extractor: Extractor = extractor or (
            EntityRelationExtractor(model=extractor_model) if extractor_model
            else RuleBasedExtractor()
        )
        self.answerer: Answerer = answerer or (
            LLMAnswerer(model=answer_model) if answer_model else ExtractiveAnswerer()
        )
        self.sort_tokens = sort_tokens
        self.max_chars = max_chars
        # Rebuilt from the database so entity ids continue across process restarts.
        self._seen_version = self.store.data_version()
        self.resolver = EntityResolver.from_store(self.store, sort_tokens=sort_tokens)
        self.retriever = HybridRetriever(
            self.store, self.embedder, self.extractor, self.resolver, alpha=alpha,
        )

    def _sync_resolver(self) -> None:
        """Rebuild the resolver if another connection has written to the database since it was
        built. Without this, two `GraphRAG` objects on one file each issued the same new entity ids
        and overwrote each other's entities (the single-process restart bug, with two writers)."""
        version = self.store.data_version()
        if version != self._seen_version:
            self._rebuild_resolver()

    def _rebuild_resolver(self) -> None:
        self._seen_version = self.store.data_version()
        self.resolver = EntityResolver.from_store(self.store, sort_tokens=self.sort_tokens)
        self.retriever.resolver = self.resolver

    # ---- ingest ----

    def ingest(self, *, doc_id: str, text: str) -> IngestReport:
        """Index a document. Re-ingesting a doc_id REPLACES its earlier chunks and edges.

        The whole document is written in one transaction: if the embedder fails part-way nothing
        of the document is stored (and the previous version, if any, is kept). A chunk whose
        extraction fails is still stored and searchable by vector; the error is listed in
        `extraction_errors` rather than swallowed.
        """
        if not isinstance(doc_id, str) or not doc_id.strip():
            raise ValueError("doc_id must be a non-empty string")
        drafts = chunk_document(text, max_chars=self.max_chars,
                                min_chars=min(200, self.max_chars // 4))
        report = IngestReport(doc_id=doc_id, chunks=len(drafts))
        # Embed and extract before touching the database: failures here change nothing.
        prepared = []
        for ch in drafts:
            emb = self.embedder.embed(ch.embed_text)
            try:
                ex = self.extractor.extract(ch.embed_text)
            except Exception as e:  # noqa: BLE001 - fail open per chunk, but record why
                report.extraction_errors.append(f"chunk {len(prepared)}: {type(e).__name__}: {e}")
                ex = None
            prepared.append((ch, emb, ex))
        try:
            with self.store.transaction():
                self._sync_resolver()           # write lock held: no one else adds entities now
                report.replaced_chunks = self.store.delete_document(doc_id)
                for idx, (ch, emb, ex) in enumerate(prepared):
                    chunk_id = self.store.add_chunk(
                        doc_id=doc_id, chunk_idx=idx, text=ch.text,
                        heading_path=ch.heading_path, embedding=emb,
                    )
                    if ex is not None:
                        self._write_extraction(chunk_id, ex, report)
        except BaseException:
            self._rebuild_resolver()
            raise
        return report

    def _write_extraction(self, chunk_id: int, ex, report: IngestReport) -> None:
        name_to_id: dict[str, int] = {}
        for ent in ex.entities:
            if not tokens_of(ent.name):         # "—", "***": nothing to key an entity on
                report.dropped_entities += 1
                continue
            eid, _is_new = self.resolver.resolve(ent.name, entity_type=ent.type)
            self.store.upsert_entity(
                entity_id=eid, canonical=self.resolver.canonical(eid), type_=ent.type,
                aliases=list(self.resolver.aliases(eid)),
            )
            self.store.link_chunk_entity(chunk_id, eid)
            name_to_id[ent.name] = eid
            report.entities += 1
        for tr in ex.triples:
            src, dst = name_to_id.get(tr.subject), name_to_id.get(tr.object)
            if src is None or dst is None:
                report.dropped_triples += 1
                continue
            if self.store.add_edge(src=src, predicate=tr.predicate, dst=dst,
                                   source_chunk=chunk_id, evidence=tr.evidence):
                report.triples += 1

    # ---- ask ----

    def retrieve(self, question: str, *, k: int = 5, hops: int = 2,
                 alpha: float | None = None) -> RetrievalOutput:
        self._sync_resolver()
        return self.retriever.retrieve_detailed(question, k=k, hops=hops, alpha=alpha)

    def ask(self, question: str, *, k: int = 5, hops: int = 2,
            alpha: float | None = None) -> AnswerResult:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")
        out = self.retrieve(question, k=k, hops=hops, alpha=alpha)
        passages = [
            Passage(label=f"c{i}", doc_id=r.chunk.doc_id,
                    breadcrumb=" > ".join(r.chunk.heading_path), text=r.chunk.text)
            for i, r in enumerate(out.chunks, 1)
        ]
        label_of_chunk = {r.chunk.id: f"c{i}" for i, r in enumerate(out.chunks, 1)}
        facts = [
            f"{f.subject} --{f.predicate}--> {f.object}"
            + (f" [{label_of_chunk[f.source_chunk]}]" if f.source_chunk in label_of_chunk else "")
            for f in out.facts
        ]
        answer = self.answerer.answer(
            AnswerInput(question=question, passages=passages, facts=facts)
        )
        valid_labels = {p.label for p in passages}
        marks = [f"c{n}" for n in _CITE.findall(answer)]
        cited = sorted({m for m in marks if m in valid_labels}, key=lambda s: int(s[1:]))
        invalid = sorted({m for m in marks if m not in valid_labels}, key=lambda s: int(s[1:]))
        entity_ids = sorted({eid for r in out.chunks for eid in r.via_entities} | set(out.seeds))
        names = []
        for e in entity_ids:
            ent = self.store.entity(e)
            if ent:
                names.append(ent["canonical"])
        return AnswerResult(
            answer=answer,
            sources=[{
                "cite": label_of_chunk[r.chunk.id], "chunk_id": r.chunk.id,
                "doc_id": r.chunk.doc_id, "heading_path": r.chunk.heading_path,
                "text": r.chunk.text, "vec_score": r.vec_score,
                "graph_proximity": r.graph_proximity, "final_score": r.final_score,
            } for r in out.chunks],
            entities_used=names,
            graph_facts=[{
                "subject": f.subject, "predicate": f.predicate, "object": f.object,
                "evidence": f.evidence, "source_chunk": f.source_chunk, "hop": f.hop,
            } for f in out.facts],
            cited=cited,
            invalid_citations=invalid,
        )

    def close(self) -> None:
        self.store.close()

    def __enter__(self) -> GraphRAG:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
