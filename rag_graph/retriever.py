"""Hybrid retriever: vector kNN + graph traversal, score-blended.

Algorithm:
  1. Run the extractor on the query to find seed entities.
  2. Resolve seed entities against the store (alias-aware).
  3. Vector kNN over chunks (top-k).
  4. Graph BFS up to `hops` from each seed entity. Each visited entity contributes
     graph_proximity = 1 / (1 + hop_distance). Map entities back to chunks via
     chunk_entities.
  5. Final score = vec_score + alpha * graph_proximity.
  6. Return de-duplicated chunks sorted by final score.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

from .embeddings import Embedder
from .extractor import EntityRelationExtractor
from .resolver import EntityResolver, normalize_surface
from .store import Chunk, GraphStore, cosine_similarity


@dataclass
class Retrieved:
    chunk: Chunk
    vec_score: float
    graph_proximity: float
    final_score: float
    via_entities: list[int]  # entities that contributed to this chunk's graph score


class HybridRetriever:
    def __init__(
        self,
        store: GraphStore,
        embedder: Embedder,
        extractor: EntityRelationExtractor,
        resolver: EntityResolver,
        *,
        alpha: float = 0.5,
    ):
        self.store = store
        self.embedder = embedder
        self.extractor = extractor
        self.resolver = resolver
        self.alpha = alpha
        # Build a reverse index from normalised surface forms -> entity id, sourced from the resolver.
        self._surface_index: dict[str, int] = {}
        for eid, canon in self.resolver._id_to_canonical.items():
            self._surface_index[normalize_surface(canon)] = eid
            for alias in self.resolver._id_to_aliases.get(eid, ()):
                self._surface_index[normalize_surface(alias)] = eid

    def retrieve(self, query: str, *, k: int = 5, hops: int = 2) -> list[Retrieved]:
        # 1+2. Seed entities — try the cheap surface match first to avoid an LLM call
        seed_ids = self._extract_seed_entities(query)

        # 3. Vector kNN
        query_emb = self.embedder.embed(query)
        vec_scores: dict[int, float] = {}
        for c in self.store.all_chunks():
            vec_scores[c.id] = cosine_similarity(query_emb, c.embedding)

        # 4. Graph BFS, collect entity -> proximity
        entity_proximity = self._bfs(seed_ids, hops=hops)
        # Map to chunks via chunk_entities
        chunk_graph_score: dict[int, float] = defaultdict(float)
        chunk_via: dict[int, set[int]] = defaultdict(set)
        for eid, prox in entity_proximity.items():
            for chunk_id in self.store.chunks_for_entities([eid]):
                chunk_graph_score[chunk_id] = max(chunk_graph_score[chunk_id], prox)
                chunk_via[chunk_id].add(eid)

        # 5+6. Blend + sort
        all_chunk_ids = set(vec_scores) | set(chunk_graph_score)
        results: list[Retrieved] = []
        for cid in all_chunk_ids:
            vec = vec_scores.get(cid, 0.0)
            graph = chunk_graph_score.get(cid, 0.0)
            chunk = self.store.chunk(cid)
            if chunk is None:
                continue
            chunk.score = vec + self.alpha * graph
            results.append(Retrieved(
                chunk=chunk, vec_score=vec, graph_proximity=graph,
                final_score=chunk.score, via_entities=sorted(chunk_via.get(cid, set())),
            ))
        results.sort(key=lambda r: r.final_score, reverse=True)
        return results[:k]

    def _extract_seed_entities(self, query: str) -> list[int]:
        """Try surface-form match first (cheap); fall back to LLM extraction."""
        # Cheap path: substring match against known surface forms.
        norm_query = normalize_surface(query)
        matched: set[int] = set()
        for surface, eid in self._surface_index.items():
            if not surface:
                continue
            if surface in norm_query.split() or surface in norm_query:
                matched.add(eid)
        if matched:
            return sorted(matched)

        # Fallback: full LLM extraction on the query
        try:
            ex = self.extractor.extract(query)
        except Exception:
            return []
        for e in ex.entities:
            eid = self._surface_index.get(normalize_surface(e.name))
            if eid is not None:
                matched.add(eid)
        return sorted(matched)

    def _bfs(self, seed_ids: list[int], *, hops: int) -> dict[int, float]:
        """BFS up to N hops. Returns entity -> proximity (1 / (1 + hop_distance))."""
        if not seed_ids or hops < 0:
            return {}
        visited: dict[int, int] = {eid: 0 for eid in seed_ids}
        queue: deque[int] = deque(seed_ids)
        while queue:
            cur = queue.popleft()
            d = visited[cur]
            if d >= hops:
                continue
            for nbr, _pred, _eid in self.store.neighbours(cur):
                if nbr not in visited:
                    visited[nbr] = d + 1
                    queue.append(nbr)
        return {eid: 1.0 / (1.0 + dist) for eid, dist in visited.items()}