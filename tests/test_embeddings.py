"""Tests for the embedding backends (offline; no model downloads)."""

import numpy as np
import pytest

from semantic_cache import embeddings
from semantic_cache.embeddings import (
    HashingEmbedder,
    SentenceTransformerEmbedder,
    cosine_similarity,
    default_embedder,
)


def test_hashing_embedder_deterministic():
    emb = HashingEmbedder(dim=128)
    a = emb.embed(["hello world"])
    b = emb.embed(["hello world"])
    np.testing.assert_array_equal(a, b)


def test_hashing_embedder_normalized():
    emb = HashingEmbedder(dim=128)
    vecs = emb.embed(["some text here", "another piece of text"])
    norms = np.linalg.norm(vecs, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-6)


def test_hashing_embedder_shape():
    emb = HashingEmbedder(dim=64)
    vecs = emb.embed(["a", "b", "c"])
    assert vecs.shape == (3, 64)
    assert vecs.dtype == np.float32


def test_hashing_embedder_similar_beats_dissimilar():
    emb = HashingEmbedder(dim=256)
    base = emb.embed(["the cat sat on the mat"])[0]
    near = emb.embed(["the cat sat on a mat"])[0]
    far = emb.embed(["quantum chromodynamics lecture notes"])[0]
    sim_near = float(cosine_similarity(near.reshape(1, -1), base)[0])
    sim_far = float(cosine_similarity(far.reshape(1, -1), base)[0])
    assert sim_near > sim_far


def test_default_embedder_never_raises_without_optional_dep(monkeypatch):
    # Simulate sentence-transformers being absent: default must fall back.
    monkeypatch.setattr(embeddings, "is_sentence_transformers_available", lambda: False)
    emb = default_embedder()
    assert isinstance(emb, HashingEmbedder)
    vecs = emb.embed(["offline works"])
    assert vecs.shape == (1, emb.dim)


def test_sentence_transformer_embedder_import_is_lazy():
    # Importing the class must not require the optional dependency.
    assert SentenceTransformerEmbedder is not None


def test_sentence_transformer_embedder_raises_helpfully_when_missing(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("sentence_transformers"):
            raise ImportError("No module named 'sentence_transformers'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="sentence-transformers is not installed"):
        SentenceTransformerEmbedder()
