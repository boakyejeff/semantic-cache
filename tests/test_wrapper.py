"""Tests for the cached_call wrapper (LLM fully mocked; no network)."""

from semantic_cache.wrapper import cached_call


def test_miss_calls_llm_and_stores(cache):
    calls = []

    def fake_llm(prompt):
        calls.append(prompt)
        return f"answer to: {prompt}"

    result = cached_call(cache, "exact prompt", fake_llm, prompt_tokens=10, completion_tokens=5)
    assert result.cache_hit is False
    assert result.response == "answer to: exact prompt"
    assert calls == ["exact prompt"]
    assert cache._store.count() == 1


def test_hit_skips_llm(cache):
    calls = []

    def fake_llm(prompt):
        calls.append(prompt)
        return "fresh answer"

    cache.store("exact prompt", "cached answer")
    result = cached_call(cache, "paraphrase of exact prompt", fake_llm)
    assert result.cache_hit is True
    assert result.similarity is not None and result.similarity > 0.90
    assert result.response == "cached answer"
    assert calls == []  # LLM never invoked on a hit


def test_second_call_hits_after_first_stores(cache):
    calls = []

    def fake_llm(prompt):
        calls.append(prompt)
        return "computed once"

    first = cached_call(cache, "exact prompt", fake_llm)
    second = cached_call(cache, "exact prompt", fake_llm)
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert second.response == "computed once"
    assert len(calls) == 1
