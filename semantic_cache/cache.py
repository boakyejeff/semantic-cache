"""The semantic cache itself: similarity lookup, TTL, LRU eviction, and stats."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .embeddings import Embedder, cosine_similarity, default_embedder
from .store import CacheEntry, SQLiteStore


def _utcnow() -> float:
    return time.time()


@dataclass
class CacheHit:
    """Result of a successful :meth:`SemanticCache.lookup`."""

    response: str
    similarity: float
    matched_prompt: str
    model: Optional[str]
    age_seconds: float


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    prompt_tokens_saved: int = 0
    completion_tokens_saved: int = 0

    @property
    def total_lookups(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        total = self.total_lookups
        return self.hits / total if total else 0.0

    @property
    def total_tokens_saved(self) -> int:
        return self.prompt_tokens_saved + self.completion_tokens_saved

    def estimated_cost_saved_usd(
        self, input_price_per_1k: Optional[float], output_price_per_1k: Optional[float]
    ) -> Optional[float]:
        """Estimated dollars saved, or None when prices are not supplied."""
        if input_price_per_1k is None or output_price_per_1k is None:
            return None
        return (
            self.prompt_tokens_saved / 1000.0 * input_price_per_1k
            + self.completion_tokens_saved / 1000.0 * output_price_per_1k
        )


class SemanticCache:
    """Semantic cache over LLM prompt/response pairs.

    Parameters
    ----------
    embedder:
        Backend used to embed prompts. Defaults to :func:`default_embedder`.
    store_path:
        SQLite file path, or ``":memory:"`` for an ephemeral cache.
    similarity_threshold:
        Cosine similarity in (0, 1] required for a hit. Higher = stricter.
        Tune on labeled paraphrase/non-paraphrase pairs (see README).
    ttl_seconds:
        Optional time-to-live for entries; expired entries are treated as
        misses and deleted.
    max_size:
        Optional cap on entries; least-recently-used entries are evicted.
    input_price_per_1k / output_price_per_1k:
        Optional USD prices per 1k tokens, used only for the estimated
        savings figure in :meth:`stats`.
    """

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        store_path: str = ":memory:",
        similarity_threshold: float = 0.90,
        ttl_seconds: Optional[float] = None,
        max_size: Optional[int] = None,
        input_price_per_1k: Optional[float] = None,
        output_price_per_1k: Optional[float] = None,
    ) -> None:
        if not 0.0 < similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be in (0, 1]")
        if ttl_seconds is not None and ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        if max_size is not None and max_size < 1:
            raise ValueError("max_size must be >= 1")
        self.embedder: Embedder = embedder or default_embedder()
        self._store = SQLiteStore(store_path)
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self.input_price_per_1k = input_price_per_1k
        self.output_price_per_1k = output_price_per_1k
        self._stats = CacheStats()

    # -- core API ---------------------------------------------------------
    def lookup(self, prompt: str) -> Optional[CacheHit]:
        """Return the cached response for a semantically similar prompt.

        Returns None on a miss. A hit updates recency (for LRU) and stats;
        expired entries are deleted and count as misses.
        """
        if not prompt or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        now = _utcnow()
        query_vec = self.embedder.embed([prompt])[0]
        best: Optional[CacheEntry] = None
        best_sim = -1.0
        for entry in self._store.all():
            sim = float(cosine_similarity(entry.embedding.reshape(1, -1), query_vec)[0])
            if sim > best_sim:
                best_sim = sim
                best = entry
        if best is None or best_sim < self.similarity_threshold:
            self._stats.misses += 1
            return None
        if self._is_expired(best, now):
            self._store.delete(best.id)
            self._stats.misses += 1
            return None
        self._store.touch(best.id, now)
        self._stats.hits += 1
        self._stats.prompt_tokens_saved += best.prompt_tokens
        self._stats.completion_tokens_saved += best.completion_tokens
        return CacheHit(
            response=best.response,
            similarity=best_sim,
            matched_prompt=best.prompt,
            model=best.model,
            age_seconds=now - best.created_at,
        )

    def store(
        self,
        prompt: str,
        response: str,
        model: Optional[str] = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> int:
        """Store a prompt/response pair. Returns the entry id.

        Evicts least-recently-used entries first when ``max_size`` is set.
        """
        if not prompt or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        if prompt_tokens < 0 or completion_tokens < 0:
            raise ValueError("token counts must be non-negative")
        self._enforce_max_size()
        vec = self.embedder.embed([prompt])[0]
        now = _utcnow()
        return self._store.add(
            prompt=prompt,
            response=response,
            embedding=vec,
            model=model,
            created_at=now,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    def stats(self) -> Dict[str, object]:
        """Return hits, misses, hit rate, tokens saved, and estimated savings."""
        s = self._stats
        return {
            "hits": s.hits,
            "misses": s.misses,
            "hit_rate": round(s.hit_rate, 4),
            "prompt_tokens_saved": s.prompt_tokens_saved,
            "completion_tokens_saved": s.completion_tokens_saved,
            "total_tokens_saved": s.total_tokens_saved,
            "estimated_cost_saved_usd": s.estimated_cost_saved_usd(
                self.input_price_per_1k, self.output_price_per_1k
            ),
            "entries": self._store.count(),
        }

    def purge_expired(self) -> int:
        """Delete all TTL-expired entries. Returns the number removed."""
        if self.ttl_seconds is None:
            return 0
        now = _utcnow()
        expired = [e.id for e in self._store.all() if self._is_expired(e, now)]
        self._store.delete_ids(expired)
        return len(expired)

    def clear(self) -> None:
        self._store.clear()
        self._stats = CacheStats()

    # -- internals --------------------------------------------------------
    def _is_expired(self, entry: CacheEntry, now: float) -> bool:
        return (
            self.ttl_seconds is not None
            and (now - entry.created_at) > self.ttl_seconds
        )

    def _enforce_max_size(self) -> None:
        if self.max_size is None:
            return
        overflow = self._store.count() - self.max_size + 1
        if overflow > 0:
            self._store.delete_ids(self._store.least_recently_used_ids(overflow))

    def close(self) -> None:
        self._store.close()
