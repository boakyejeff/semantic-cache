"""Tests for SemanticCache: lookup, threshold, TTL, eviction, stats."""

import pytest

from semantic_cache.cache import SemanticCache


def test_exact_match_hit(cache):
    cache.store("exact prompt", "the answer")
    hit = cache.lookup("exact prompt")
    assert hit is not None
    assert hit.response == "the answer"
    assert hit.similarity == pytest.approx(1.0)
    assert hit.matched_prompt == "exact prompt"


def test_paraphrase_hit_above_threshold(cache):
    cache.store("exact prompt", "the answer")
    hit = cache.lookup("paraphrase of exact prompt")
    assert hit is not None
    assert hit.similarity == pytest.approx(0.95)
    assert hit.response == "the answer"


def test_miss_below_threshold(embedder):
    cache = SemanticCache(embedder=embedder, similarity_threshold=0.97)
    try:
        cache.store("exact prompt", "the answer")
        # paraphrase sits at cosine 0.95 < 0.97 -> miss
        assert cache.lookup("paraphrase of exact prompt") is None
        # unrelated prompt is far below any sane threshold
        assert cache.lookup("something completely unrelated") is None
    finally:
        cache.close()


def test_miss_on_empty_store(cache):
    assert cache.lookup("exact prompt") is None


def test_lookup_rejects_blank_prompt(cache):
    with pytest.raises(ValueError):
        cache.lookup("   ")


def test_store_rejects_blank_prompt(cache):
    with pytest.raises(ValueError):
        cache.store("", "response")


def test_threshold_validation(embedder):
    with pytest.raises(ValueError):
        SemanticCache(embedder=embedder, similarity_threshold=0.0)
    with pytest.raises(ValueError):
        SemanticCache(embedder=embedder, similarity_threshold=1.5)


def test_ttl_expiry(embedder, frozen_time):
    cache = SemanticCache(embedder=embedder, ttl_seconds=60.0)
    try:
        cache.store("exact prompt", "the answer")
        assert cache.lookup("exact prompt") is not None  # fresh: hit
        frozen_time["t"] += 61.0  # move past TTL
        assert cache.lookup("exact prompt") is None  # expired: miss
        assert cache._store.count() == 0  # expired entry was deleted
    finally:
        cache.close()


def test_ttl_not_expired_yet(embedder, frozen_time):
    cache = SemanticCache(embedder=embedder, ttl_seconds=60.0)
    try:
        cache.store("exact prompt", "the answer")
        frozen_time["t"] += 59.0
        assert cache.lookup("exact prompt") is not None
    finally:
        cache.close()


def test_purge_expired(embedder, frozen_time):
    cache = SemanticCache(embedder=embedder, ttl_seconds=60.0)
    try:
        cache.store("exact prompt", "old answer")
        frozen_time["t"] += 120.0
        removed = cache.purge_expired()
        assert removed == 1
        assert cache._store.count() == 0
    finally:
        cache.close()


def test_lru_eviction(embedder, frozen_time):
    cache = SemanticCache(embedder=embedder, max_size=2)
    try:
        cache.store("exact prompt", "r1")  # t=1000
        frozen_time["t"] += 10
        cache.store("paraphrase of exact prompt", "r2")  # t=1010
        # Touch the first entry strictly later so the second becomes LRU
        frozen_time["t"] += 10
        cache.lookup("exact prompt")  # touched at t=1020
        frozen_time["t"] += 10
        cache.store("something completely unrelated", "r3")  # t=1030, evicts one
        prompts = {e.prompt for e in cache._store.all()}
        assert cache._store.count() == 2
        # "paraphrase..." was least recently used -> evicted
        assert "paraphrase of exact prompt" not in prompts
        assert "exact prompt" in prompts
        assert "something completely unrelated" in prompts
    finally:
        cache.close()


def test_stats_math(embedder):
    cache = SemanticCache(
        embedder=embedder,
        similarity_threshold=0.90,
        input_price_per_1k=0.0015,
        output_price_per_1k=0.002,
    )
    try:
        cache.store("exact prompt", "the answer", prompt_tokens=100, completion_tokens=50)
        cache.lookup("something completely unrelated")  # miss
        cache.lookup("paraphrase of exact prompt")  # hit (0.95)
        cache.lookup("exact prompt")  # hit (1.0)
        s = cache.stats()
        assert s["hits"] == 2
        assert s["misses"] == 1
        assert s["hit_rate"] == pytest.approx(round(2 / 3, 4))
        assert s["prompt_tokens_saved"] == 200
        assert s["completion_tokens_saved"] == 100
        assert s["total_tokens_saved"] == 300
        # (200/1000)*0.0015 + (100/1000)*0.002 = 0.0003 + 0.0002
        assert s["estimated_cost_saved_usd"] == pytest.approx(0.0005)
        assert s["entries"] == 1
    finally:
        cache.close()


def test_stats_cost_none_without_prices(cache):
    cache.store("exact prompt", "the answer", prompt_tokens=10, completion_tokens=5)
    cache.lookup("exact prompt")
    s = cache.stats()
    assert s["estimated_cost_saved_usd"] is None
    assert s["total_tokens_saved"] == 15


def test_hit_updates_use_count_and_recency(cache, frozen_time):
    cache.store("exact prompt", "the answer")
    frozen_time["t"] += 5
    cache.lookup("exact prompt")
    entry = cache._store.all()[0]
    assert entry.use_count == 1
    assert entry.last_used_at == frozen_time["t"]
