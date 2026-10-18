"""Embedders.

`HashingEmbedder` is the offline default: a deterministic bag of hashed words, so cosine
similarity measures shared vocabulary. It is LEXICAL, not semantic; it exists so the whole
pipeline (and the benchmark) runs with no model, key or network. `OpenAIEmbedder` is the
optional real one. `HashEmbedder` (a SHA-256 digest turned into a vector) is kept only for
code that needs *a* deterministic vector: its similarities are unrelated to meaning.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from typing import Protocol, runtime_checkable

_WORD = re.compile(r"[^\W_]+")
_STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the their this to was "
    "were which who whom whose with".split()
)


@runtime_checkable
class Embedder(Protocol):
    dimension: int

    def embed(self, text: str) -> list[float]: ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...


class HashingEmbedder:
    """Deterministic lexical embedder: signed hashed word counts, L2-normalised.

    Words are case-folded and common stopwords are dropped, so two texts are similar when they
    share content words. Text with no content words embeds to the zero vector.
    """

    def __init__(self, dimension: int = 512) -> None:
        if dimension < 8:
            raise ValueError("dimension must be >= 8")
        self.dimension = dimension

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dimension
        for word in _WORD.findall(text.casefold()):
            if word in _STOPWORDS:
                continue
            n = int.from_bytes(hashlib.blake2b(word.encode(), digest_size=8).digest(), "big")
            vec[n % self.dimension] += 1.0 if (n >> 63) & 1 else -1.0
        norm = math.sqrt(sum(x * x for x in vec))
        return [x / norm for x in vec] if norm else vec

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


class OpenAIEmbedder:
    """OpenAI embeddings (optional extra: `pip install rag-graph[openai]`)."""

    _DIMENSIONS = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072,
                   "text-embedding-ada-002": 1536}

    def __init__(self, model: str = "text-embedding-3-small", api_key: str | None = None) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:  # pragma: no cover - only without the extra
            raise ImportError("OpenAIEmbedder needs: pip install 'rag-graph[openai]'") from e
        if model not in self._DIMENSIONS:
            raise ValueError(f"unknown model {model!r}; known: {sorted(self._DIMENSIONS)}")
        self.model = model
        self.dimension = self._DIMENSIONS[model]
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def embed(self, text: str) -> list[float]:
        resp = self.client.embeddings.create(model=self.model, input=text)
        return list(resp.data[0].embedding)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        resp = self.client.embeddings.create(model=self.model, input=texts)
        return [list(d.embedding) for d in resp.data]


class HashEmbedder:
    """A SHA-256 digest as a vector. Deterministic, but its similarities mean nothing.

    Kept for tests that need a stable vector. Use `HashingEmbedder` for retrieval.
    """

    def __init__(self, dimension: int = 16) -> None:
        if not 1 <= dimension <= 32:
            raise ValueError("dimension must be between 1 and 32 (a SHA-256 digest has 32 bytes)")
        self.dimension = dimension

    def embed(self, text: str) -> list[float]:
        h = hashlib.sha256(text.encode("utf-8")).digest()
        return [(b - 128) / 128.0 for b in h[: self.dimension]]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]
