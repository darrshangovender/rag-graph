"""rag-graph — knowledge-graph-augmented RAG."""

from .core import GraphRAG, AnswerResult
from .extractor import EntityRelationExtractor, Triple, Entity
from .resolver import EntityResolver
from .store import GraphStore, Chunk
from .retriever import HybridRetriever
from .embeddings import Embedder, OpenAIEmbedder
from .chunker import chunk_document

__version__ = "0.1.0"
__all__ = [
    "GraphRAG", "AnswerResult",
    "EntityRelationExtractor", "Triple", "Entity",
    "EntityResolver",
    "GraphStore", "Chunk",
    "HybridRetriever",
    "Embedder", "OpenAIEmbedder",
    "chunk_document",
]