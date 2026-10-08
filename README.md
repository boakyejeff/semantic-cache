# Semantic Cache for LLM Responses

A drop-in semantic caching layer: store LLM prompt → response pairs with
embeddings in SQLite, and serve the cached response when a new prompt is
*semantically* similar enough — cutting cost and latency on repeated or
paraphrased queries. Exact-string caches miss paraphrases; this one doesn't.

## Why

LLM APIs charge per token and take seconds per call. In real products the
same question gets asked many ways ("What's the capital of France?" vs
"What's the capital city of France?"). A semantic cache answers the second
from the first: one embedding lookup instead of another API call.

## Install

```bash
pip install -r requirements.txt
```

For production-quality embeddings (recommended), also install:

```bash
pip install sentence-transformers
```

Without it, the cache falls back to a deterministic offline hashing embedder
(see Limitations). Everything — tests and demo — runs with zero network.

## Quickstart

```python
from semantic_cache import SemanticCache, cached_call

cache = SemanticCache(
    similarity_threshold=0.90,   # cosine similarity required for a hit
    ttl_seconds=3600,            # entries expire after 1 hour (optional)
    max_size=10_000,             # LRU-evict beyond this many entries (optional)
    input_price_per_1k=0.0015,   # for the estimated-savings stat (optional)
    output_price_per_1k=0.006,
)

def call_llm(prompt: str) -> str:
    return my_llm_client.complete(prompt)  # your real API call here

result = cached_call(cache, "What is the capital of France?", call_llm)
print(result.response, "| cache hit:", result.cache_hit)

print(cache.stats())
# {'hits': 1, 'misses': 0, 'hit_rate': 1.0, 'prompt_tokens_saved': ...,
#  'total_tokens_saved': ..., 'estimated_cost_saved_usd': ..., 'entries': 1}
```

Lower-level API if you prefer manual control:

```python
hit = cache.lookup("paraphrased question here")
if hit:
    print(hit.response, hit.similarity, hit.matched_prompt)
else:
    cache.store(prompt, response, model="gpt-4o-mini",
                prompt_tokens=42, completion_tokens=18)
```

Run the offline demo (mock LLM, no network):

```bash
python examples/demo.py
```

Run the test suite:

```bash
pytest tests/ -q
```

## How similarity-threshold tuning works

The threshold is the single most important setting, and there is no universal
right value — it depends on your embedder and your tolerance for wrong hits.

Tuning procedure:

1. Collect 50–200 real prompt pairs from your logs, labeled as
   *paraphrase* (same intent) or *not* (different intent).
2. Embed both prompts in each pair; record the cosine similarity.
3. Plot the two score distributions (or just eyeball the overlap).
4. Pick the threshold that maximizes F1 on this labeled set — the point that
   best separates "same intent" from "different intent".
5. Re-tune whenever you change the embedder. **Thresholds do not transfer
   between embedders**: the hashing fallback used in the offline demo needs
   ~0.60, while `all-MiniLM-L6-v2` typically wants ~0.85–0.95 on the same data.

A useful rule of thumb: start strict (high threshold, fewer but safer hits),
measure the false-hit rate on live traffic, then relax gradually. A wrong
cached answer is worse than a cache miss.

## API reference

- `SemanticCache(embedder=None, store_path=":memory:", similarity_threshold=0.90, ttl_seconds=None, max_size=None, input_price_per_1k=None, output_price_per_1k=None)`
  - `.lookup(prompt) -> CacheHit | None`
  - `.store(prompt, response, model=None, prompt_tokens=0, completion_tokens=0) -> int`
  - `.stats() -> dict` — hits, misses, hit_rate, tokens saved, estimated USD saved, entry count
  - `.purge_expired() -> int`, `.clear()`, `.close()`
- `cached_call(cache, prompt, llm_fn, model=None, prompt_tokens=0, completion_tokens=0) -> CallResult`
- Embedders: `SentenceTransformerEmbedder`, `HashingEmbedder`, `default_embedder()`
- Storage: `SQLiteStore` (embeddings as float32 blobs; linear scan — O(n) lookup)

## Limitations (read before production use)

- **Embedding quality determines everything.** The cache is only as smart as
  the vectors. The bundled `HashingEmbedder` fallback captures surface
  character overlap, not meaning ("not good" looks like "good"). Use a real
  sentence-embedding model for anything beyond a demo.
- **Not a substitute for evaluation.** A semantic hit can be wrong. Track the
  false-hit rate on your traffic, keep thresholds strict, and never cache
  responses where a wrong answer is costly (medical, legal, financial advice)
  without human review.
- **No ANN index.** Lookup scans all stored embeddings: O(n). Fine for
  thousands of entries; use FAISS/ScaNN/pgvector past that.
- **Token counts are caller-supplied.** The cache trusts the `prompt_tokens`
  / `completion_tokens` you pass; the demo uses a rough heuristic, not a real
  tokenizer. Savings figures are estimates, not invoices.
- **SQLite concurrency.** Fine for a single process; use WAL mode or a server
  DB for multi-writer deployments.
- **Prompt privacy.** Cached prompts persist on disk — don't cache PII or
  secrets unless your threat model allows it.

## Project layout

```
semantic_cache/      # the library
  embeddings.py      # pluggable embedders (neural + offline hashing fallback)
  store.py           # SQLite vector store
  cache.py           # SemanticCache: lookup/store/TTL/LRU/stats
  wrapper.py         # cached_call() helper
examples/demo.py     # offline demo with a mocked LLM
tests/               # pytest suite (24 tests, all offline)
```

## License

MIT — see LICENSE.
