"""Pluggable text-embedding backends for the semantic cache.

The default backend is a real sentence-embedding model (``sentence-transformers``)
when it is installed. Otherwise :func:`default_embedder` falls back to
:class:`HashingEmbedder`, a deterministic, offline, dependency-free embedder
built on hashed character n-grams.

Be honest about the trade-off: the hashing fallback captures surface lexical
overlap only. It cannot do true semantic matching the way a trained model can,
so similarity thresholds tuned on the fallback do NOT transfer to the
sentence-transformer backend (and vice versa). The fallback exists so the
cache, tests, and demo run anywhere with zero network access.
"""

from __future__ import annotations

import hashlib
import re
from typing import List, Optional, Sequence

import numpy as np


class Embedder:
    """Interface every embedder backend must implement."""

    dim: int

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return an (n, dim) float32 array of L2-normalized embeddings."""
        raise NotImplementedError


class SentenceTransformerEmbedder(Embedder):
    """Embedder backed by a ``sentence-transformers`` model.

    The import is lazy so that merely importing this module never fails when
    the optional dependency is missing. The model is downloaded on first use,
    which requires network access exactly once.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on env
            raise ImportError(
                "sentence-transformers is not installed. Install it with "
                "`pip install sentence-transformers`, or use HashingEmbedder "
                "for a dependency-free offline fallback."
            ) from exc
        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self._model.encode(list(texts), normalize_embeddings=True)
        return np.asarray(vectors, dtype=np.float32)


class HashingEmbedder(Embedder):
    """Deterministic offline fallback embedder.

    Builds a sparse-ish vector from hashed character n-grams (3-5 grams) with
    sign hashing, then L2-normalizes. Two prompts with heavy character-level
    overlap score high; true paraphrases with different wording score lower
    than they would under a trained model.

    Honest limitations:
      * No notion of meaning: "not good" and "good" look similar.
      * Sensitive to punctuation/casing unless you normalize first (we lowercase).
      * Thresholds tuned here do not transfer to neural embedders.
    """

    def __init__(self, dim: int = 512, ngram_range: tuple[int, int] = (3, 5)) -> None:
        self.dim = dim
        self.ngram_range = ngram_range

    @staticmethod
    def _normalize(text: str) -> str:
        text = text.lower()
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _featurize(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float64)
        text = self._normalize(text)
        lo, hi = self.ngram_range
        for n in range(lo, hi + 1):
            for i in range(len(text) - n + 1):
                gram = text[i : i + n]
                h = int(hashlib.md5(gram.encode("utf-8")).hexdigest(), 16)
                idx = h % self.dim
                # sign hashing: decorrelates collisions
                vec[idx] += 1.0 if (h >> 7) & 1 else -1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec.astype(np.float32)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self._featurize(t) for t in texts])


def is_sentence_transformers_available() -> bool:
    """True when the optional neural backend can be imported."""
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        return False
    return True


def default_embedder(
    model_name: str = "all-MiniLM-L6-v2", fallback_dim: int = 512
) -> Embedder:
    """Return the best available embedder.

    Prefers :class:`SentenceTransformerEmbedder`; silently falls back to
    :class:`HashingEmbedder` when ``sentence-transformers`` is not installed.
    Never raises at import or call time for a missing optional dependency.
    """
    if is_sentence_transformers_available():
        return SentenceTransformerEmbedder(model_name)
    return HashingEmbedder(dim=fallback_dim)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between each row of ``a`` and vector ``b``.

    Assumes rows are L2-normalized (both bundled embedders guarantee this),
    so this is a plain dot product.
    """
    return np.asarray(a, dtype=np.float32) @ np.asarray(b, dtype=np.float32).ravel()
