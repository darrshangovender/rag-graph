"""Hybrid retriever: vector similarity + graph traversal, score-blended.

Algorithm:
  1. Seed entities: find known entity names in the query by whole-word match (alias-aware, tolerant
     of "Open AI" vs "OpenAI"); if none, fall back to the extractor on the query.
  2. Vector scores for every chunk (numpy, one matrix product).
  3. Breadth-first search from the seeds up to `hops`, capped at `max_entities` visited and
     `max_degree` neighbours per entity. Each visited entity gets proximity 1 / (1 + distance).
  4. A chunk's graph score is the best proximity among the entities it mentions.
  5. final = max(0, cosine) + alpha * graph_score. `alpha=0` is plain vector search.
  6. Sort by (-final, chunk id) and return the top k, plus the graph facts that connect the
     visited entities, for the answerer to read.
"""

from __future__ import annotations

import heapq
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Literal

from .embeddings import Embedder
from .extractor import Extractor
from .resolver import EntityResolver, tokens_of
from .store import Chunk, GraphStore


@dataclass
class Retrieved:
    chunk: Chunk
    vec_score: float
    graph_proximity: float
    final_score: float
    via_entities: list[int]  # entities that contributed to this chunk's graph score


@dataclass
class GraphFact:
    edge_id: int
    subject: str
    predicate: str
    object: str
    evidence: str
    source_chunk: int
    hop: int                 # distance of the nearer endpoint from a seed entity


@dataclass
class RetrievalOutput:
    chunks: list[Retrieved]
    seeds: list[int]
    proximity: dict[int, float] = field(default_factory=dict)
    facts: list[GraphFact] = field(default_factory=list)


_STEM_WORD = re.compile(r"[^\W_]+")
_STEM_STOP = frozenset("the and who what which where when how why for from with that this was "
                       "were are has have had did does into than then".split())


def stem(word: str) -> str:
    """A crude English stem (strip -ing/-ed/-es/-s): enough to match 'founded' with 'founded' and
    'acquires' with 'acquired'. It does not know irregular forms ('made' does not match 'makes')."""
    w = word.casefold()
    for suffix in ("ing", "ed", "es", "s"):
        if w.endswith(suffix) and len(w) - len(suffix) >= 3:
            return w[: -len(suffix)]
    return w


def relation_stems(text: str) -> set[str]:
    return {
        stem(w) for w in _STEM_WORD.findall(text)
        if len(w) >= 3 and w.casefold() not in _STEM_STOP
    }


class HybridRetriever:
    def __init__(
        self,
        store: GraphStore,
        embedder: Embedder,
        extractor: Extractor | None,
        resolver: EntityResolver,
        *,
        alpha: float = 0.5,
        max_entities: int = 200,
        max_degree: int = 50,
        max_facts: int = 12,
        traversal: Literal["uniform", "relation_guided"] = "uniform",
        off_relation_cost: int = 2,
    ):
        if traversal not in ("uniform", "relation_guided"):
            raise ValueError("traversal must be 'uniform' or 'relation_guided'")
        if off_relation_cost < 1:
            raise ValueError("off_relation_cost must be >= 1")
        if alpha < 0:
            raise ValueError("alpha must be >= 0")
        if max_entities < 1 or max_degree < 1 or max_facts < 0:
            raise ValueError("max_entities and max_degree must be >= 1, max_facts >= 0")
        self.store = store
        self.embedder = embedder
        self.extractor = extractor
        self.resolver = resolver
        self.alpha = alpha
        self.max_entities = max_entities
        self.max_degree = max_degree
        self.max_facts = max_facts
        self.traversal = traversal
        self.off_relation_cost = off_relation_cost

    # ---- public ----

    def retrieve(self, query: str, *, k: int = 5, hops: int = 2,
                 alpha: float | None = None) -> list[Retrieved]:
        return self.retrieve_detailed(query, k=k, hops=hops, alpha=alpha).chunks

    def retrieve_detailed(self, query: str, *, k: int = 5, hops: int = 2,
                          alpha: float | None = None) -> RetrievalOutput:
        if k < 1:
            raise ValueError("k must be >= 1")
        if hops < 0:
            raise ValueError("hops must be >= 0")
        a = self.alpha if alpha is None else alpha
        if a < 0:
            raise ValueError("alpha must be >= 0")

        seeds = self._seed_entities(query)
        vec_scores = self.store.vector_scores(self.embedder.embed(query))
        # alpha=0 is pure vector search: skip the graph entirely.
        distance = self._bfs(seeds, hops=hops, query=query) if a > 0 else {}
        proximity = {eid: 1.0 / (1.0 + d) for eid, d in distance.items()}

        chunk_graph: dict[int, float] = defaultdict(float)
        chunk_via: dict[int, set[int]] = defaultdict(set)
        for eid, cids in self.store.chunks_by_entity(list(proximity)).items():
            for cid in cids:
                chunk_graph[cid] = max(chunk_graph[cid], proximity[eid])
                chunk_via[cid].add(eid)

        candidates = set(vec_scores) | set(chunk_graph)
        scored = []
        for cid in candidates:
            vec = vec_scores.get(cid, 0.0)
            graph = chunk_graph.get(cid, 0.0)
            scored.append((-(max(0.0, vec) + a * graph), cid, vec, graph))
        scored.sort()
        top = scored[:k]
        chunks = self.store.chunks([cid for _, cid, _, _ in top])
        results: list[Retrieved] = []
        for neg_final, cid, vec, graph in top:
            chunk = chunks.get(cid)
            if chunk is None:
                continue
            chunk.score = -neg_final
            results.append(Retrieved(
                chunk=chunk, vec_score=vec, graph_proximity=graph, final_score=-neg_final,
                via_entities=sorted(chunk_via.get(cid, set())),
            ))
        facts = self._facts(distance, results) if a > 0 else []
        return RetrievalOutput(chunks=results, seeds=seeds, proximity=proximity, facts=facts)

    # ---- seeds ----

    def _seed_entities(self, query: str) -> list[int]:
        """Known entities named in the query (whole-word, alias-aware), else extractor fallback."""
        found = self._match_known_entities(query)
        if found or self.extractor is None:
            return found
        try:
            ex = self.extractor.extract(query)
        except Exception:  # noqa: BLE001 - optional fallback; "no seeds" is a valid outcome
            return []
        ids = {self.resolver.lookup(e.name) for e in ex.entities}
        return sorted(i for i in ids if i is not None)

    def _match_known_entities(self, query: str, max_window: int = 6) -> list[int]:
        toks = tokens_of(query)
        matched: set[int] = set()
        for i in range(len(toks)):
            for j in range(i + 1, min(len(toks), i + max_window) + 1):
                eid = self.resolver.lookup(" ".join(toks[i:j]))
                if eid is not None:
                    matched.add(eid)
        return sorted(matched)

    # ---- graph ----

    def _edge_costs(self, query: str):
        """cost(predicate) -> traversal cost of an edge.

        Uniform: every edge costs 1 hop. Relation-guided: an edge whose predicate shares a stem
        with the question (the question says "founded", the edge is `founded`) costs 1, any other
        edge costs `off_relation_cost`, so hub entities reached through an unrelated relation (a
        city that hosts many companies) fall outside the hop budget. If NO predicate in the graph
        matches the question, every edge costs 1 (nothing to guide by).
        """
        if self.traversal == "uniform":
            return lambda predicate: 1
        qstems = relation_stems(query)
        graph_stems = set()
        for pred in self.store.predicates():
            graph_stems |= relation_stems(pred)
        if not (qstems & graph_stems):
            return lambda predicate: 1

        def cost(predicate: str) -> int:
            return 1 if relation_stems(predicate.replace("(inv)", "")) & qstems \
                else self.off_relation_cost
        return cost

    def _bfs(self, seed_ids: list[int], *, hops: int, query: str = "") -> dict[int, int]:
        """entity -> distance from the nearest seed (cheapest path within `hops`), within caps.

        Dijkstra with small integer edge costs (all 1 for uniform traversal, i.e. plain BFS).
        Ties break by entity id and neighbours come in edge-id order, so results are deterministic.
        """
        if not seed_ids:
            return {}
        cost = self._edge_costs(query)
        dist: dict[int, int] = {}
        heap: list[tuple[int, int]] = []
        for eid in sorted(set(seed_ids))[: self.max_entities]:
            dist[eid] = 0
            heapq.heappush(heap, (0, eid))
        settled: set[int] = set()
        while heap:
            d, cur = heapq.heappop(heap)
            if cur in settled or d > dist.get(cur, d):
                continue
            settled.add(cur)
            if d >= hops:
                continue
            for nbr, pred, _edge in self.store.neighbours(cur, limit=self.max_degree):
                nd = d + cost(pred)
                if nd > hops:
                    continue
                if nbr not in dist:
                    if len(dist) >= self.max_entities:
                        continue
                    dist[nbr] = nd
                    heapq.heappush(heap, (nd, nbr))
                elif nd < dist[nbr]:
                    dist[nbr] = nd
                    heapq.heappush(heap, (nd, nbr))
        return dist

    def _facts(self, distance: dict[int, int], results: list[Retrieved]) -> list[GraphFact]:
        """Edges among the visited entities, nearest the seeds first, capped at max_facts."""
        if not distance or self.max_facts == 0:
            return []
        edges = self.store.edges_among(list(distance))
        edges.sort(key=lambda e: (min(distance[e["src"]], distance[e["dst"]]), e["id"]))
        facts: list[GraphFact] = []
        names: dict[int, str] = {}

        def name(eid: int) -> str:
            if eid not in names:
                ent = self.store.entity(eid)
                names[eid] = ent["canonical"] if ent else str(eid)
            return names[eid]

        for e in edges[: self.max_facts]:
            facts.append(GraphFact(
                edge_id=e["id"], subject=name(e["src"]), predicate=e["predicate"],
                object=name(e["dst"]), evidence=e["evidence"], source_chunk=e["source_chunk"],
                hop=min(distance[e["src"]], distance[e["dst"]]),
            ))
        return facts
