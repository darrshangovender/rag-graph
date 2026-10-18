"""rag-graph: knowledge-graph-augmented RAG that runs offline and shows its own benchmark."""

from .answerers import Answerer, AnswerInput, ExtractiveAnswerer, LLMAnswerer, Passage
from .chunker import ChunkDraft, chunk_document
from .core import AnswerResult, GraphRAG, IngestReport
from .embeddings import Embedder, HashEmbedder, HashingEmbedder, OpenAIEmbedder
from .extractor import (
    Entity,
    EntityRelationExtractor,
    Extraction,
    Extractor,
    Pattern,
    RuleBasedExtractor,
    Triple,
)
from .resolver import EntityResolver, normalize_key, normalize_surface
from .retriever import GraphFact, HybridRetriever, RetrievalOutput, Retrieved
from .store import Chunk, GraphStore

__version__ = "0.1.0"
__all__ = [
    "AnswerInput",
    "AnswerResult",
    "Answerer",
    "Chunk",
    "ChunkDraft",
    "Embedder",
    "Entity",
    "EntityRelationExtractor",
    "EntityResolver",
    "Extraction",
    "ExtractiveAnswerer",
    "Extractor",
    "GraphFact",
    "GraphRAG",
    "GraphStore",
    "HashEmbedder",
    "HashingEmbedder",
    "HybridRetriever",
    "IngestReport",
    "LLMAnswerer",
    "OpenAIEmbedder",
    "Passage",
    "Pattern",
    "RetrievalOutput",
    "Retrieved",
    "RuleBasedExtractor",
    "Triple",
    "chunk_document",
    "normalize_key",
    "normalize_surface",
]
