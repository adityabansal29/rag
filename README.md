# RAG Pipeline

A modular Retrieval-Augmented Generation (RAG) pipeline with pluggable parsers, embedders, LLM enrichers, and vector stores. Supports hybrid search (BM25 + dense) across Chroma, Pinecone, and Qdrant.

## Architecture

```
Document
   │
   ▼
[Parser]         → unstructured or docling
   │
   ▼
[Chunker]        → merge small / split large text chunks (100–512 tokens)
   │              non-text chunks (table, image, code) preserved as-is
   ▼
[LLM Enricher]   → describe images, summarize tables/code, summarize parent sections
   │
   ▼
[Embedder]       → OpenAI text-embedding-3-small (default) or custom
   │
   ▼
[Vector Store]   → Chroma (local) · Pinecone (serverless) · Qdrant
```

### Chunk hierarchy

Each document section becomes a **parent chunk** with one or more **child chunks**. Only children are embedded and stored. The parent's section heading is prepended to each child's embedding content for context.

```
Parent (section heading)
├── Child 1  (text)
├── Child 2  (table → LLM summary)
└── Child 3  (image → LLM description)
```

## Chunk types

| Type | Description |
|------|-------------|
| `TEXT` | Regular paragraph text |
| `TABLE` | Tabular data, LLM-summarised before embedding |
| `IMAGE` | Base64 image, LLM-described before embedding |
| `CODE` | Code block, LLM-explained before embedding |
| `DIAGRAM` | Diagram/figure, LLM-described before embedding |

## Vector stores

| Store | Hybrid search | Notes |
|-------|--------------|-------|
| Chroma | DIY — BM25Retriever + RRF | Local persistent or remote HTTP |
| Pinecone | Native sparse-dense + alpha weighting | Requires `dotproduct` metric |
| Qdrant | Native — fastembed BM25 + built-in RRF | In-memory, local, or remote |

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
from rag.pipeline import RAGPipeline

pipeline = RAGPipeline()

# Index a document
pipeline.run("./paper.pdf", parser="unstructured")

# Search
results = pipeline.search("How does hinted handoff work?")
for doc in results:
    print(doc.metadata["score"], doc.page_content)
```

## Search params

```python
from rag.vectorstores.base import SearchParams

params = SearchParams(
    top_k=5,
    cosine_threshold=0.6,      # 0 = unrelated, 1 = identical
    rrf_score_threshold=0.02,  # 0 = absent from both lists, ~0.033 = rank-1 in both
    use_hybrid=True,
    metadata_filters={"chunk_type": "text"},
)
results = pipeline.search("your query", params=params)
```

`cosine_threshold` applies to dense-only search; `rrf_score_threshold` applies when `use_hybrid=True`.

## Parsers

```python
# Unstructured (default) — fast, broad format support
pipeline.run("file.pdf", parser="unstructured")

# Docling — better table and layout extraction
pipeline.run("file.pdf", parser="docling")
```

## Swapping components

```python
from rag.pipeline import RAGPipeline
from rag.embedders.openai_embedder import OpenAIEmbedder
from rag.vectorstores.qdrant_store import QdrantVectorStore
from rag.vectorstores.pinecone_store import PineconeVectorStore

# Qdrant with hybrid search
pipeline = RAGPipeline(
    vectorstore=QdrantVectorStore(
        collection_name="rag",
        dimension=1536,
        path="./qdrant_db",
        enable_hybrid=True,
    )
)

# Pinecone with hybrid search
pipeline = RAGPipeline(
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

pipeline = RAGPipeline(embedder=MyEmbedder())
```

## LLM enrichment

Supports OpenAI, Anthropic, and Gemini for enriching non-text chunks. Configured via `llm_model` on `RAGPipeline`:

```python
pipeline = RAGPipeline(llm_model="gpt-4o", llm_concurrency=10)
```

## Dependencies

| Category | Packages |
|----------|----------|
| Parsing | `unstructured[all-docs]`, `docling` |
| Chunking | `tiktoken` |
| Embedding | `openai`, `fastembed` |
| Vector stores | `chromadb`, `pinecone`, `qdrant-client` |
| Hybrid search | `langchain-community`, `pinecone-text`, `rank-bm25` |
| LLM | `openai`, `anthropic`, `google-genai` |
| Framework | `langchain-core`, `python-dotenv` |
