# RAG Pipeline

A modular Retrieval-Augmented Generation (RAG) pipeline with pluggable parsers, embedders, LLM enrichers, and vector stores. Supports hybrid search (BM25 + dense), fusion RAG (query rewriting + parallel search + RRF), and multi-turn conversational queries.

## Architecture

### Ingestion

```
Document
   │
   ▼
[Parser]         → unstructured or docling
   │
   ▼
[Chunker]        → merge small / split large text chunks (100–512 tokens)
   │              non-text chunks (table, image, code, diagram) preserved as-is
   ▼
[LLM Enricher]   → describe images, summarize tables/code, summarize parent sections
   │
   ▼
[Embedder]       → OpenAI text-embedding-3-small (default) or custom
   │
   ▼
[Vector Store]   → Chroma (local) · Pinecone (serverless) · Qdrant
```

### Retrieval — Hybrid

```
query
  │
  ▼
[HybridRAGPipeline.search()]
  ├── (optional) QueryContextualizer  → resolves follow-up references using history
  ├── BM25 retrieval  ──┐
  └── Dense retrieval ──┴─→ RRF merge → top_k chunks
```

### Retrieval — Fusion

```
query
  │
  ▼
[FusionRAGPipeline.search()]
  ├── (optional) QueryRewriter + contextualize → standalone query + n variants (1 LLM call)
  ├── Parallel HybridRAGPipeline.search() per variant
  └── RRF merge across all result lists → top_k chunks
```

### Chunk hierarchy

Each document section becomes a **parent chunk** with one or more **child chunks**. Only children are embedded and stored. The parent's section heading is prepended to each child's embedding content for context.

```
Parent (section heading)
├── Child 1  (text)
├── Child 2  (table → LLM summary)
└── Child 3  (image → LLM description)
```

## Modules

| Module | Description |
|--------|-------------|
| `hybrid/pipeline.py` | `HybridRAGPipeline` — ingestion + hybrid search + answer generation |
| `fusion/pipeline.py` | `FusionRAGPipeline` — wraps hybrid pipeline with query rewriting and RRF fusion |
| `fusion/query_rewriter.py` | `QueryRewriter` — rewrites query into n variants; also contextualizes follow-ups |
| `conversation/history.py` | `ConversationHistory` — stores prior Q&A turns |
| `conversation/contextualizer.py` | `QueryContextualizer` — rewrites follow-up questions as standalone queries |
| `evaluators/llm_evaluator.py` | `LLMEvaluator` — scores chunk relevance and answer faithfulness |
| `parsers/` | `UnstructuredParser`, `DoclingParser` |
| `chunkers/chunker.py` | Token-aware merge/split of text chunks |
| `enrichers/enricher.py` | `LLMEnricher` — async LLM enrichment of non-text chunks |
| `embedders/` | `OpenAIEmbedder` (default), `BaseEmbedder` for custom |
| `vectorstores/` | `ChromaVectorStore`, `PineconeVectorStore`, `QdrantVectorStore` |
| `llm/` | `OpenAILLMClient`, `AnthropicLLMClient`, `GeminiLLMClient` via LangChain |
| `utils/rrf.py` | `rrf_fuse()` — generic Reciprocal Rank Fusion |

## Chunk types

| Type | Embedding content |
|------|-------------------|
| `TEXT` | Raw paragraph text |
| `TABLE` | LLM summary of the table |
| `IMAGE` | LLM description of the image |
| `CODE` | LLM explanation of the code block |
| `DIAGRAM` | LLM description of the diagram |

## Vector stores

| Store | Hybrid search | Notes |
|-------|--------------|-------|
| Chroma | DIY — BM25Retriever + custom RRF | Local persistent or remote HTTP |
| Pinecone | Native sparse-dense + alpha weighting | Requires `dotproduct` metric index |
| Qdrant | Native — fastembed BM25 + built-in RRF | In-memory, local path, or remote |

## Setup

```bash
# Install with uv
uv sync

# Editable install (for import resolution in notebooks/IDE)
uv pip install -e ..

# Set environment variables
cp .env.example .env   # add OPENAI_API_KEY (and others as needed)
```

## Quick start

```python
from rag import HybridRAGPipeline
from rag.vectorstores.base import SearchParams

pipeline = HybridRAGPipeline()

# Index a document
pipeline.run("./paper.pdf", parser="unstructured")

# Search
results = pipeline.search("How does hinted handoff work?", params=SearchParams(top_k=5))
for doc in results:
    print(doc.metadata["score"], doc.page_content)

# Generate an answer
answer = pipeline.generate_answer("How does hinted handoff work?", results)
print(answer)
```

## Search params

```python
from rag.vectorstores.base import SearchParams

params = SearchParams(
    top_k=5,
    cosine_threshold=0.6,      # 0 = unrelated, 1 = identical; applies to dense-only search
    rrf_score_threshold=0.02,  # 0 = absent from both lists, ~0.033 = rank-1 in both; applies to hybrid
    use_hybrid=True,
    metadata_filters={"chunk_type": "table"},
)
results = pipeline.search("your query", params=params)
```

## Fusion RAG

Rewrites the query into n variants and searches in parallel, then merges all result lists with RRF for higher recall.

```python
from rag import HybridRAGPipeline, FusionRAGPipeline
from rag.vectorstores.base import SearchParams

pipeline = HybridRAGPipeline()
fusion   = FusionRAGPipeline(pipeline, n_queries=3)

results = fusion.search(
    "How does Dynamo resolve conflicts?",
    params=SearchParams(top_k=5, use_hybrid=True),
)
```

## Multi-turn / conversational queries

When follow-up questions contain references to prior context ("What are its limitations?"), pass a `ConversationHistory` to resolve them into standalone queries before retrieval.

For `HybridRAGPipeline`, contextualization is a separate LLM call. For `FusionRAGPipeline`, contextualization and query rewriting are merged into a single LLM call.

```python
from rag import HybridRAGPipeline, ConversationHistory
from rag.vectorstores.base import SearchParams

pipeline = HybridRAGPipeline()
history  = ConversationHistory()

questions = [
    "How does Dynamo handle write conflicts?",
    "What are its limitations?",           # resolved using history
    "How does hinted handoff solve that?", # chained follow-up
]

for question in questions:
    chunks = pipeline.search(
        question,
        params=SearchParams(top_k=4, use_hybrid=True),
        history=history,
    )
    answer = pipeline.generate_answer(question, chunks)
    print(answer)
    history.add(question, answer)
```

## Evaluation

Reference-free evaluation scoring chunk relevance and answer faithfulness in parallel.

```python
from rag.evaluators.llm_evaluator import LLMEvaluator

evaluator = LLMEvaluator(pipeline.enricher.llm)  # reuses pipeline's LLM client

chunks = pipeline.search("your question", params=SearchParams(top_k=5))
answer = pipeline.generate_answer("your question", chunks)
result = evaluator.evaluate_sync("your question", chunks, answer)
result.print_summary()
```

`EvalResult` fields: `avg_chunk_relevance` (0–1), `faithfulness` (0–1), `unsupported_claims`, per-chunk `ChunkScore`.

## Swapping components

```python
from rag import HybridRAGPipeline
from rag.embedders.openai_embedder import OpenAIEmbedder
from rag.vectorstores.qdrant_store import QdrantVectorStore
from rag.vectorstores.pinecone_store import PineconeVectorStore

# Qdrant with hybrid search
pipeline = HybridRAGPipeline(
    vectorstore=QdrantVectorStore(
        collection_name="rag",
        dimension=1536,
        path="./qdrant_db",
        enable_hybrid=True,
    )
)

# Pinecone with hybrid search
pipeline = HybridRAGPipeline(
    vectorstore=PineconeVectorStore(
        api_key="...",
        index_name="rag-pipeline",
        enable_hybrid=True,   # requires dotproduct metric — use a new index
    )
)
```

### Pinecone alpha (hybrid weight)

```python
results = pipeline.vectorstore.search(
    query_vector=vector,
    query_text="query",
    params=params,
    alpha=0.75,   # 1.0 = fully dense, 0.0 = fully sparse
)
```

## Custom embedder

```python
from rag.embedders.base import BaseEmbedder

class MyEmbedder(BaseEmbedder):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...
    def embed_query(self, text: str) -> list[float]:
        ...

pipeline = HybridRAGPipeline(embedder=MyEmbedder())
```

## LLM clients

All LLM calls use LangChain with `with_structured_output` for typed Pydantic responses. Switch provider via `build_llm_client`:

```python
from rag.llm.base import build_llm_client

llm = build_llm_client("anthropic", model="claude-opus-4-7")
llm = build_llm_client("gemini",    model="gemini-2.0-flash")
llm = build_llm_client("openai",    model="gpt-4o")  # default

pipeline = HybridRAGPipeline(llm_model="gpt-4o", llm_concurrency=10)
```

## Dependencies

| Category | Packages |
|----------|----------|
| Parsing | `unstructured[all-docs]`, `docling` |
| Chunking | `tiktoken` |
| Embedding | `openai`, `fastembed` |
| Vector stores | `chromadb`, `pinecone`, `qdrant-client` |
| Hybrid search | `langchain-community`, `pinecone-text`, `rank-bm25` |
| LLM | `langchain-openai`, `langchain-anthropic`, `langchain-google-genai` |
| Framework | `langchain-core`, `pydantic`, `python-dotenv` |
