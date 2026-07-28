# Fusion RAG Pipeline

## Why not just a single query?

Even with hybrid search, a single query phrasing may miss relevant chunks. The same concept can be expressed many ways, and the best-matching chunks may use different terminology:

- Query: *"How does Dynamo handle write conflicts?"*
- Relevant chunk: *"Dynamo uses vector clocks to track causal dependencies between object versions..."*

A rewritten variant like *"What is Dynamo's conflict resolution mechanism for concurrent writes?"* retrieves that chunk far more effectively. RAG-Fusion runs multiple phrasings in parallel and merges results with RRF.

---

## Flow

```
query
  │
  ▼
QueryRewriter (single LLM call → structured output)
  → standalone query       ← original intent preserved
  → variant 1
  → variant 2  ... variant n
  │
  ├─→ BM25(standalone)      ──── 1 BM25 list ─────┐
  ├─→ dense(standalone)     ──┐                   │
  ├─→ dense(variant 1)      ──┤  n+1 dense        │
  ├─→ dense(variant 2)      ──┤  lists, all       │
  └─→ dense(variant n)      ──┘  parallel         │
                              │                   │
                              └───────────────────┘
                                        │
                                  Outer RRF merge
                                  (n+2 rank lists)
                                        │
                                 top_k candidates
                                        │
                                 [Re-ranker]  ← optional, scores vs standalone
                                        │
                                    top_k chunks
```

---

## Why outer RRF?

Each search returns chunks in ranked order. The outer RRF treats each result list as a rank list — no score normalisation needed across result sets. A chunk that ranks highly across multiple query variants is genuinely relevant from multiple semantic angles, and the RRF score reflects that directly.

**No threshold is applied at the outer level.** This is intentional: the RRF score range scales with the number of rank lists (`more lists → higher possible scores`), so an absolute threshold would need retuning whenever `n_queries` changes. Sorting by score and slicing `[:top_k]` is threshold-free and always correct.

---

## Sub-query inflation

Each sub-query fetches `top_k × n_total_queries` results with per-query thresholds disabled:

```python
inflated = params.top_k * n_total_queries
dense = replace(params, top_k=inflated, cosine_threshold=None, rrf_score_threshold=None)
```

This ensures the outer RRF pool is large enough to surface the best chunks after deduplication. Per-query threshold filtering would discard good candidates before the outer RRF evaluates them across all rank lists.

---

## Fusion + Hybrid: BM25 once, not per variant

### The naive approach (and why it fails)

Running full hybrid search (BM25 + dense) for every query variant seems like it would give the best of both worlds. In practice it hurts:

1. **BM25 runs redundantly.** Variants are semantic reformulations — different words, same intent. BM25 is term-based and produces overlapping keyword-matching results across variants regardless of their semantic differences.

2. **Consistent BM25 nominations inflate outer RRF scores artificially.** A chunk that matches keywords in all variants accumulates a high outer RRF score not because it's broadly semantically relevant, but because BM25 kept finding the same keywords. This crowds out better chunks.

3. **Empirically worse.** Chunk relevance and answer faithfulness both dropped vs dense-only fusion in testing.

### The principled approach

**BM25 runs exactly once** on the standalone query. It contributes one rank list to the outer RRF — anchoring keyword recall for the original intent — without polluting the semantic signal from dense variants. Its contribution is directly measurable: if BM25 consistently co-nominates the top chunks alongside dense, it's adding value; if it doesn't, its single rank list has limited influence.

---

## Query contextualization in multi-turn chat

In the conversational agent path, the LangGraph agent manages history and contextualizes queries **before** calling the `rag_search` tool. By the time `FusionRAGPipeline.search_async` receives the query, it is already a self-contained, history-resolved question. `QueryRewriter` then generates semantic variants of this already-contextualized query — it does not need access to conversation history.

For direct pipeline use (notebooks, scripts), queries referencing prior context won't be automatically resolved. Pass the full self-contained question.

---

## Re-ranker (optional)

After the outer RRF produces a final top_k, an optional cross-encoder re-ranker re-scores those candidates jointly — reading query and document together rather than independently.

**Why apply re-ranking after the outer RRF and not per sub-query?**

Sub-queries are intermediate — they feed the outer RRF pool, not the final answer. Re-ranking them individually would discard good candidates before fusion evaluates them across all rank lists. The right place is after fusion has synthesised all signals.

The re-ranker scores against the **standalone query** (the original-intent query), not the variants.

```python
from rag.hybrid.pipeline import HybridRAGPipeline
from rag.fusion.pipeline import FusionRAGPipeline
from rag.rerankers.cross_encoder import CrossEncoderReranker

pipeline = HybridRAGPipeline(
    reranker=CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2")
)
fusion = FusionRAGPipeline(pipeline, n_queries=3)
# re-ranker is shared via pipeline.reranker — applied automatically in search_async()
```

---

## Design decisions

| Decision | Reason |
|----------|--------|
| RRF over score fusion | Result lists from different queries are not on a comparable scale |
| No threshold on outer RRF | Score range scales with `n_queries`; sort + slice is always correct |
| BM25 once on standalone | Variants are semantic reformulations; BM25 should not run on rephrased queries |
| Sub-query thresholds disabled | Let outer RRF decide quality; per-query filtering loses good candidates early |
| Re-rank after outer RRF, not per sub-query | Sub-queries are intermediate; re-ranking them prematurely loses candidates |
| Re-rank against standalone query | Standalone captures original intent; variants are reformulations |
| Degraded mode on LLM failure | Returns `(query, [])` variants → runs as single-query dense search, never hard-fails |

---

## Usage

```python
from rag.hybrid.pipeline import HybridRAGPipeline
from rag.fusion.pipeline import FusionRAGPipeline
from rag.rerankers.cross_encoder import CrossEncoderReranker
from rag.vectorstores.base import SearchParams

# Dense fusion only (semantic query diversification)
pipeline = HybridRAGPipeline()
fusion   = FusionRAGPipeline(pipeline, n_queries=3)

standalone, chunks = await fusion.search_async(
    "How does Dynamo resolve write conflicts?",
    params=SearchParams(top_k=5, use_hybrid=False),
)

# Fusion + hybrid (BM25 once on standalone + dense on all variants)
pipeline = HybridRAGPipeline(
    reranker=CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2")
)
fusion = FusionRAGPipeline(pipeline, n_queries=3)

standalone, chunks = await fusion.search_async(
    "How does Dynamo resolve write conflicts?",
    params=SearchParams(top_k=5, use_hybrid=True),
)
# chunks[i].metadata["rrf_score"]    — outer RRF score
# chunks[i].metadata["rerank_score"] — cross-encoder score (if re-ranker set)
```
