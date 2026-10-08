"""Offline demo: semantic cache with a mocked LLM (no network, no API keys).

Run:  python examples/demo.py   (from the project root)

Shows:
  1. cache misses on first-seen prompts (mock LLM is "called"),
  2. a HIT on an exact repeat,
  3. a HIT on a paraphrased repeat,
  4. a MISS on an unrelated prompt,
  5. stats: hits, misses, hit rate, tokens and dollars "saved".

Note on the threshold: this demo runs on the offline HashingEmbedder
fallback (sentence-transformers is not installed here), which only sees
character-level overlap, so paraphrases score ~0.65-0.72 and the demo uses
threshold 0.60. With the sentence-transformer backend you would use ~0.90.
Thresholds do not transfer between embedders -- tune per backend (see README).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from semantic_cache import SemanticCache, cached_call, is_sentence_transformers_available


class MockLLM:
    """Stands in for a real LLM API: deterministic, offline, countable."""

    def __init__(self):
        self.calls = 0
        self.answers = {
            "france": "Paris is the capital of France.",
            "password": "To reset your password: open Settings > Security > Reset password, then follow the emailed link.",
            "planet": "Jupiter is the largest planet in our solar system.",
        }

    def __call__(self, prompt: str) -> str:
        self.calls += 1
        low = prompt.lower()
        if "france" in low or "capital" in low:
            return self.answers["france"]
        if "password" in low:
            return self.answers["password"]
        if "planet" in low:
            return self.answers["planet"]
        return "I don't have a canned answer for that (mock LLM)."


def rough_tokens(text: str) -> int:
    # NOT a real tokenizer: ~1.3 tokens/word heuristic, documented as rough.
    return max(1, int(len(text.split()) * 1.3))


def main() -> None:
    backend = "sentence-transformers" if is_sentence_transformers_available() else "HashingEmbedder (offline fallback)"
    print(f"Embedder backend: {backend}\n")

    cache = SemanticCache(
        similarity_threshold=0.60,  # tuned for the hashing fallback; see note above
        max_size=100,
        ttl_seconds=3600,
        input_price_per_1k=0.0015,   # example GPT-4o-mini-ish pricing
        output_price_per_1k=0.006,
    )
    llm = MockLLM()

    def ask(prompt: str) -> None:
        result = cached_call(
            cache,
            prompt,
            llm,
            model="mock-llm-1",
            prompt_tokens=rough_tokens(prompt),
            completion_tokens=rough_tokens("placeholder"),
        )
        tag = "HIT " if result.cache_hit else "MISS"
        sim = f" (similarity={result.similarity:.3f})" if result.similarity else ""
        print(f"[{tag}]{sim} Q: {prompt}")
        print(f"       A: {result.response}\n")

    # Seed the cache (all misses -> mock LLM "called")
    ask("What is the capital of France?")
    ask("How do I reset my password?")
    ask("What is the largest planet?")

    # Exact repeat -> hit
    ask("What is the capital of France?")

    # Paraphrased repeat -> hit (semantic, not string, match)
    ask("What's the capital city of France?")

    # Unrelated -> miss
    ask("Explain quantum entanglement simply.")

    print(f"Mock LLM was actually called {llm.calls} time(s).")
    print("\nCache stats:")
    for key, value in cache.stats().items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
