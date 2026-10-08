"""Convenience wrapper: check the cache before calling the LLM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from .cache import SemanticCache


@dataclass
class CallResult:
    response: str
    cache_hit: bool
    similarity: Optional[float] = None


def cached_call(
    cache: SemanticCache,
    prompt: str,
    llm_fn: Callable[[str], str],
    model: Optional[str] = None,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> CallResult:
    """Run ``llm_fn(prompt)`` with a semantic-cache fast path.

    On a cache hit the LLM is never called. On a miss the response is stored
    (with the supplied token counts and model label) before being returned.
    """
    hit = cache.lookup(prompt)
    if hit is not None:
        return CallResult(response=hit.response, cache_hit=True, similarity=hit.similarity)
    response = llm_fn(prompt)
    cache.store(
        prompt,
        response,
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
    return CallResult(response=response, cache_hit=False)
