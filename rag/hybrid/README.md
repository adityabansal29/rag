# Hybrid RAG Pipeline

## Why not just dense vector search?

Dense search retrieves by cosine similarity in embedding space. It captures semantic meaning well — "automobile" and "car" are close — but has blind spots:

- **Rare terms**: model names, identifiers, and technical jargon that wasn't well-represented in training data embed poorly
- **Exact keyword recall**: a query for "hinted handoff" may semantically match vague "availability" chunks instead of the specific section that uses those exact words
- **Short, specific queries**: a 3-word query embedding may not capture intent as well as direct term matching

BM25 covers these gaps but requires exact term overlap and has no notion of meaning. Hybrid search combines both.

---

## Flow

```
Document
  │
  ▼
[Parser]       → unstructured or docling
  │
  ▼
[Chunker]      → merge small / split large text chunks (100–512 tokens)
  │              non-text chunks (table, image, code) preserved as-is
  ▼
[LLM Enricher] → describe images · summarize tables/code · summarize parent sections
  │              (async batched, semaphore-bounded concurrency)
  ▼
[Embedder]     → OpenAI text-embedding-3-small (default)
  │
  ▼
[Vector Store] → Chroma / Pinecone / Qdrant


query
  │
  ├─→ BM25 (keyword ranking over full corpus)  ──┐
  │                                              ├─→ RRF merge → top_k candidates
  └─→ Dense (cosine similarity via embedding)  ──┘
                                                         │
                                                  [Re-ranker]   ← optional
                                                         │
                                                     top_k chunks
```

---

## Parent-Document Retrieval

Documents are chunked into a parent-child hierarchy:

```
Parent (section heading / page)
├── Child 1  (text paragraph)
├── Child 2  (table → LLM summary)
└── Child 3  (image → LLM description)
```

**Only children are embedded.** This keeps embeddings focused on specific content rather than verbose section context, improving retrieval precision.

**At generation time**, each child carries `parent_summary` in its metadata — the LLM-generated summary of its parent section, produced during enrichment. The `generate_answer` helper injects this as `[Section context: ...]` above the chunk content, giving the model both the specific retrieved passage and its broader section context without polluting the embedding space.

---

## Why RRF for merging?

BM25 scores and cosine similarity scores are on completely different scales — you can't add or average them directly. RRF sidesteps this by working purely on **rank position**, not score value:

```
rrf_score(chunk) = Σ  1 / (k + rank_in_list)   for each list the chunk appears in
```

With `k=60`, a chunk ranked 1st contributes `1/61 ≈ 0.016` per list. A chunk that ranks highly in both BM25 and dense lists accumulates a high combined score regardless of the underlying score scales. Chunks absent from a list contribute 0.

---

## Score thresholds

Two independent thresholds exist because the two modes produce incomparable scores:

| Param | Applies to | Range | Default |
|-------|-----------|-------|---------|
| `cosine_threshold` | Dense-only search | 0 (unrelated) → 1 (identical) | 0.6 |
| `rrf_score_threshold` | Hybrid search (RRF output) | 0 (absent from both) → ~0.033 (rank-1 in both) | 0.02 |

In hybrid mode, `cosine_threshold` is not applied — the RRF score already reflects both signals.

---

## Vector store notes

| Store | BM25 implementation | Notes |
|-------|-------------------|-------|
| Chroma | In-memory `BM25Retriever` over all stored docs | Full corpus fetched per query — fine for small corpora |
| Qdrant | Native sparse index (fastembed BM25) | Requires `enable_hybrid=True` at collection creation |
| Pinecone | Native sparse-dense with `BM25Encoder` | Requires `dotproduct` metric and `enable_hybrid=True` |

Chroma fetches the full corpus on every hybrid search because it has no native sparse index. For large corpora, Qdrant or Pinecone are more efficient.

---

## Re-ranker (optional)

BM25 and dense search both rank by a single score per document independently — they don't consider the query and candidate document *together*. A cross-encoder re-ranker fixes this.

A cross-encoder reads the query and each candidate document jointly and produces a relevance score that captures their interaction. It's more expensive (one inference per candidate) but significantly more accurate. Since it only runs on the top candidates already retrieved, the cost is bounded.

```
BM25 + Dense → RRF → top_k candidates → [CrossEncoder] → re-scored, re-ordered top_k
```

Two implementations:

| Class | Backend | Cost | Notes |
|-------|---------|------|-------|
| `CrossEncoderReranker` | sentence-transformers (local) | free, GPU optional | `ms-marco-MiniLM-L-6-v2` default |
| `CohereReranker` | Cohere API | per-call | `rerank-english-v3.0` default |

The re-ranker is injected at construction and applied transparently after every `search_async()` call. Pass `reranker=None` (default) to skip it. Both re-rankers add a `rerank_score` field to each returned document's metadata.

---

## Usage

```python
from rag.hybrid.pipeline import HybridRAGPipeline, generate_answer
from rag.rerankers.cross_encoder import CrossEncoderReranker
from rag.vectorstores.base import SearchParams

# Without re-ranker
pipeline = HybridRAGPipeline()

# With local cross-encoder re-ranker
pipeline = HybridRAGPipeline(
    reranker=CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2")
)

# Ingest
await pipeline.run_async("paper.pdf", parser="unstructured")

# Dense-only search
_, chunks = await pipeline.search_async(
    "How does hinted handoff work?",
    params=SearchParams(top_k=5),
)

# Hybrid search (BM25 + dense + RRF, then re-rank if set)
_, chunks = await pipeline.search_async(
    "How does hinted handoff work?",
    params=SearchParams(top_k=5, use_hybrid=True),
)

# Generate answer (parent_summary injected into context automatically)
answer = await generate_answer(
    pipeline.enricher.llm,
    "How does hinted handoff work?",
    chunks,
)
```
