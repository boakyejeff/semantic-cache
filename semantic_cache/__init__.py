"""Semantic caching layer for LLM responses.

Stores prompt -> response pairs with embeddings in SQLite and serves cached
responses for semantically similar prompts, cutting cost and latency on
repeated or paraphrased queries.
"""

from .cache import CacheHit, CacheStats, SemanticCache
from .embeddings import (
    Embedder,
    HashingEmbedder,
    SentenceTransformerEmbedder,
    cosine_similarity,
    default_embedder,
    is_sentence_transformers_available,
)
from .store import CacheEntry, SQLiteStore
from .wrapper import CallResult, cached_call

__all__ = [
    "SemanticCache",
    "CacheHit",
    "CacheStats",
    "SQLiteStore",
    "CacheEntry",
    "Embedder",
    "HashingEmbedder",
    "SentenceTransformerEmbedder",
    "default_embedder",
    "is_sentence_transformers_available",
    "cosine_similarity",
    "CallResult",
    "cached_call",
]

__version__ = "0.1.0"
