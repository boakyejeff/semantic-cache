"""Shared fixtures for the semantic-cache test suite."""

import numpy as np
import pytest

from semantic_cache.cache import SemanticCache
from semantic_cache.embeddings import Embedder


class StubEmbedder(Embedder):
    """Deterministic embedder with caller-controlled vectors.

    Maps each known prompt to a fixed unit vector so tests control cosine
    similarity exactly, with no model and no network.
    """

    def __init__(self, mapping: dict):
        vecs = {k: np.asarray(v, dtype=np.float32) for k, v in mapping.items()}
        norms = {k: np.linalg.norm(v) for k, v in vecs.items()}
        assert all(n > 0 for n in norms.values()), "vectors must be non-zero"
        self.mapping = {k: v / norms[k] for k, v in vecs.items()}
        self.dim = len(next(iter(vecs.values())))

    def embed(self, texts):
        try:
            return np.stack([self.mapping[t] for t in texts]).astype(np.float32)
        except KeyError as exc:
            raise KeyError(f"StubEmbedder has no vector for prompt: {exc}") from exc


@pytest.fixture
def vectors():
    # [1, 0] is the "exact" direction; paraphrase sits at cosine 0.95 to it.
    return {
        "exact prompt": [1.0, 0.0],
        "paraphrase of exact prompt": [0.95, float(np.sqrt(1 - 0.95**2))],
        "something completely unrelated": [0.0, 1.0],
    }


@pytest.fixture
def embedder(vectors):
    return StubEmbedder(vectors)


@pytest.fixture
def cache(embedder):
    c = SemanticCache(embedder=embedder, similarity_threshold=0.90)
    yield c
    c.close()


@pytest.fixture
def frozen_time(monkeypatch):
    """Freeze semantic_cache.cache._utcnow at a controllable value."""
    import semantic_cache.cache as cache_mod

    state = {"t": 1_000.0}
    monkeypatch.setattr(cache_mod, "_utcnow", lambda: state["t"])
    return state
