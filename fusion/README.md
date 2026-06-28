# Fusion RAG Pipeline

## Why not just a single query?

Even with hybrid search, a single query phrasing may miss relevant chunks. The same concept can be expressed in many ways, and the best-matching chunks may use different terminology than the query:

- Query: *"How does Dynamo handle write conflicts?"*
- Relevant chunk: *"Dynamo uses vector clocks to track causal dependencies between object versions..."*

A rewritten variant like *"What is Dynamo's conflict resolution mechanism for concurrent writes?"* retrieves that chunk far more effectively. Fusion RAG runs multiple phrasings in parallel and merges results with RRF.

---

## Flow

```
query
  │
  ▼
QueryRewriter (single LLM call)
  → standalone query  ← coreferences resolved if conversation history present
  → variant 1
  → variant 2
  → variant n
  │
  ├─→ search(standalone) ──┐
  ├─→ search(variant 1)   ─┤  all in parallel
  ├─→ search(variant 2)   ─┤
  └─→ search(variant n)   ─┘
                            │
                      Outer RRF merge
                            │
                     top_k candidates
                            │
                      [Re-ranker]   ← optional, scores against standalone query
                            │
                        top_k chunks
```

---

## Why outer RRF?

Each search returns chunks in ranked order. The outer RRF treats each result list as a rank list — no score normalisation needed across result sets. A chunk that ranks highly across multiple query variants is genuinely relevant from multiple semantic angles, and the RRF score reflects that directly.

**No threshold is applied at the outer level.** This is intentional: the RRF score range scales with the number of rank lists (`more lists → higher possible scores`), so an absolute threshold would need retuning whenever `n_queries` changes. Sorting by score and slicing `[:top_k]` is threshold-free and always correct.

---

## Sub-query inflation

Each sub-query fetches `top_k × n_total_queries` results with thresholds disabled:

```python
sub_params = replace(params,
    top_k=params.top_k * n_total_queries,
    cosine_threshold=None,
    rrf_score_threshold=None,
)
```

This ensures the outer RRF pool is large enough to surface the best chunks after deduplication. Per-query threshold filtering would discard good candidates before the outer RRF gets to evaluate them across all lists.

---

## Fusion + Hybrid: BM25 once, not per variant

### The naive approach (and why it fails)

Running full hybrid search (BM25 + dense) for every query variant seems like it would give the best of both worlds. In practice it hurts:

1. **BM25 runs redundantly.** Variants are semantic reformulations — they mean the same thing with different words. BM25 is term-based and returns overlapping keyword-matching results across variants regardless of semantic difference.

2. **Consistent BM25 nominations inflate outer RRF scores artificially.** A chunk that matches keywords in all variants accumulates a high outer RRF score not because it's broadly semantically relevant, but because BM25 found the same keywords repeatedly. This crowds out better chunks.

3. **Empirically worse.** Chunk relevance and faithfulness both dropped vs dense-only fusion in testing.

### The principled approach

```
query
  │
  ▼
QueryRewriter
  → standalone query
  → variant 1 ... variant n
  │
  ├─→ BM25(standalone)  ───────────────────┐  ← 1 BM25 list
  ├─→ dense(standalone) ──┐                │
  ├─→ dense(variant 1)  ──┤  n+1 dense     │
  ├─→ dense(variant 2)  ──┤  lists,        │
  └─→ dense(variant n)  ──┘  all parallel  │
                             │              │
                             └──────────────┘
                                    │
                              Outer RRF merge
                              (n+2 rank lists)
                                    │
                             top_k candidates
                                    │
                              [Re-ranker]   ← optional
                                    │
                                top_k chunks
```

**BM25 runs exactly once** on the standalone query. It contributes one rank list to the outer RRF — anchoring keyword recall for the original intent — without polluting the semantic signal from dense variants. Its contribution is directly measurable in the final RRF scores: if BM25 consistently co-nominates the top chunks alongside dense, it's adding value; if it doesn't, its single rank list has limited influence.

---

## Multi-turn / Conversational queries

Follow-up questions that reference prior context (*"What are its limitations?"*) are contextualized and rewritten in a **single LLM call**:

- Without history → LLM produces `n` semantic variants of the query
- With history → LLM resolves the follow-up into a standalone query **and** produces `n` variants

Merging contextualization and rewriting into one call saves a round-trip and gives the model full conversation context to produce better variants from the start.

```python
from rag.conversation.history import ConversationHistory

history = ConversationHistory()

chunks = await fusion.search_async(
    "What are its limitations?",         # ambiguous follow-up
    params=SearchParams(top_k=5, use_hybrid=True),
    history=history,
)
history.add("What are its limitations?", answer)
```

---

## Re-ranker (optional)

After the outer RRF produces a sorted top_k list, an optional re-ranker can re-score those candidates with a cross-encoder — scoring query and document jointly rather than independently.

**Why apply re-ranking here and not per sub-query?**

Sub-queries are intermediate — they feed the outer RRF pool, not the final answer. Re-ranking them individually would discard good candidates before the fusion step gets to evaluate them across all rank lists. The right place is after the outer RRF has already synthesised all signals into a final top_k.

The re-ranker scores against the **standalone query** (the contextualized, original-intent query), not the variants, so it judges relevance to what the user actually asked.

```python
from rag.hybrid.pipeline import HybridRAGPipeline
from rag.fusion.pipeline import FusionRAGPipeline
from rag.rerankers.cross_encoder import CrossEncoderReranker

pipeline = HybridRAGPipeline(
    reranker=CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2")
)
fusion = FusionRAGPipeline(pipeline, n_queries=3)
# re-ranker is shared via pipeline.reranker — applied automatically in fusion.search_async()
```

---

## Design decisions summary

| Decision | Reason |
|----------|--------|
| RRF over score fusion | Result lists from different queries are not on a comparable scale |
| No threshold on outer RRF | Score range scales with `n_queries`; sort+slice is always correct |
| BM25 once on standalone | Variants are semantic reformulations; BM25 should not run on rephrased queries |
| Sub-query thresholds disabled | Let outer RRF decide quality; per-query filtering loses good candidates early |
| Contextualize + rewrite in one LLM call | Saves one round-trip; model produces better variants with full conversation context |
| Re-rank after outer RRF, not per sub-query | Sub-queries are intermediate; re-ranking them loses candidates before fusion evaluates across all lists |
| Re-rank against standalone query | Standalone captures original intent; variants are reformulations, not the target relevance axis |

---

## Usage

```python
from rag.hybrid.pipeline import HybridRAGPipeline
from rag.fusion.pipeline import FusionRAGPipeline
from rag.rerankers.cross_encoder import CrossEncoderReranker
from rag.vectorstores.base import SearchParams

# Without re-ranker
pipeline = HybridRAGPipeline()
fusion   = FusionRAGPipeline(pipeline, n_queries=3)

# With local cross-encoder re-ranker (shared via pipeline.reranker)
pipeline = HybridRAGPipeline(
    reranker=CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2")
)
fusion = FusionRAGPipeline(pipeline, n_queries=3)

# Dense fusion (semantic diversification only)
chunks = await fusion.search_async(
    "How does Dynamo resolve write conflicts?",
    params=SearchParams(top_k=5, use_hybrid=False),
)

# Fusion + hybrid (BM25 once on standalone + dense on all variants) + re-rank
chunks = await fusion.search_async(
    "How does Dynamo resolve write conflicts?",
    params=SearchParams(top_k=5, use_hybrid=True),
)
```
