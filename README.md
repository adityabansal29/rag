# RAG Pipeline

A full-stack RAG system with document ingestion, hybrid/fusion search, and a conversational agent UI.

## Structure

```
rag/          # Python library — parsers, chunkers, embedders, vector stores, pipelines
backend/      # FastAPI API + SQS worker + LangGraph agent
frontend/     # Next.js UI — upload, job tracking, chat
```

## Architecture

### Ingestion (S3 → SQS → Worker → Vector Store)

```
Upload (browser)
  → S3 presigned PUT
  → POST /notify  → DynamoDB job (queued) + SQS message
  → SQS worker    → download → pipeline → DynamoDB step progress
```

Pipeline steps:

```
[Parser]       → unstructured or docling
[Chunker]      → merge small / split large text chunks (100–512 tokens)
               non-text chunks (table, image, code, diagram) preserved as-is
[LLM Enricher] → describe images, summarize tables/code, summarize parent sections
[Embedder]     → OpenAI text-embedding-3-small
[Vector Store] → Chroma (local) · Pinecone · Qdrant
```

### Retrieval — Hybrid

```
query
  ├── BM25 retrieval  ──┐
  └── Dense retrieval ──┴─→ RRF merge → [CrossEncoder re-ranker] → top_k chunks
```

### Retrieval — Fusion

```
query
  → QueryRewriter  → standalone + n variants (1 LLM call)
  ├── BM25(standalone)  ─────────────────────────┐
  ├── dense(standalone) ──┐                      │
  ├── dense(variant 1)   ─┤  all parallel        │
  └── dense(variant n)   ─┘                      │
                           └────────────── outer RRF merge → [re-ranker] → top_k
```

Fusion is used for chat. BM25 runs once on the standalone query; dense runs per variant to maximize semantic coverage without polluting BM25 signal.

### Chat

LangGraph ReAct agent with a `rag_search` tool (FusionRAG). Conversation history is managed via a LangGraph checkpointer (DynamoDB in production, SQLite in dev) keyed by `thread_id`. The client stores `thread_id` in `sessionStorage` — stateless client, stateful server.

### Chunk hierarchy

```
Parent (section heading)
├── Child 1  (text)
├── Child 2  (table → LLM summary)
└── Child 3  (image → LLM description)
```

Only children are embedded. The parent's section title is prepended to each child's embedding content. Parent summaries are stored and injected at generation time, not into embeddings.

## Setup

```bash
# Python (from repo root)
uv sync
uv pip install -e .

# Frontend
cd frontend && npm install
```

Copy `.env.example` to `.env` and fill in:

```
OPENAI_API_KEY=
AWS_REGION=
S3_BUCKET=
SQS_QUEUE_URL=
DYNAMO_TABLE=rag-jobs
LANGGRAPH_CHECKPOINTER=dynamodb        # or sqlite for dev
CHAT_CHECKPOINTS_TABLE=rag-chat-checkpoints
CHAT_WRITES_TABLE=rag-chat-writes
```

Create DynamoDB tables:

```bash
python backend/scripts/setup_dynamo.py
```

## Running

```bash
bash run.sh api      # FastAPI on :8000
bash run.sh worker   # SQS listener
bash run.sh ui       # Next.js on :3000

bash run.sh upload ./paper.pdf   # presign → PUT S3 → notify API
```

## Modules

| Module | Description |
|--------|-------------|
| `rag/hybrid/pipeline.py` | `HybridRAGPipeline` — ingestion + hybrid search + answer generation |
| `rag/fusion/pipeline.py` | `FusionRAGPipeline` — query rewriting + parallel hybrid + outer RRF |
| `rag/fusion/query_rewriter.py` | `QueryRewriter` — rewrites query into n semantic variants |
| `rag/evaluators/llm_evaluator.py` | `LLMEvaluator` — chunk relevance + answer faithfulness scoring |
| `rag/parsers/` | `UnstructuredParser`, `DoclingParser` |
| `rag/chunkers/chunker.py` | Token-aware merge/split |
| `rag/enrichers/enricher.py` | `LLMEnricher` — async LLM enrichment + `build_embedding_content` |
| `rag/embedders/` | `OpenAIEmbedder`, `BaseEmbedder` |
| `rag/vectorstores/` | `ChromaVectorStore`, `PineconeVectorStore`, `QdrantVectorStore` |
| `rag/rerankers/` | `CrossEncoderReranker` (local), `CohereReranker` (API) |
| `rag/llm/` | `OpenAILLMClient`, `AnthropicLLMClient`, `GeminiLLMClient` |
| `rag/utils/rrf.py` | `rrf_fuse()` — Reciprocal Rank Fusion |
| `backend/api/app.py` | FastAPI lifespan — LangGraph agent init |
| `backend/api/routes.py` | API endpoints: presign, notify, jobs, chat, health |
| `backend/api/state.py` | Shared AWS clients + pipeline + agent ref |
| `backend/worker/sqs_listener.py` | SQS polling loop |
| `backend/worker/job_tracker.py` | DynamoDB step progress tracker |
| `backend/scripts/setup_dynamo.py` | Create DynamoDB tables |

## Vector stores

| Store | Hybrid search | Notes |
|-------|--------------|-------|
| Chroma | In-memory BM25Retriever + RRF | Full corpus per query — fine for small corpora |
| Pinecone | Native sparse-dense + alpha weighting | Requires `dotproduct` metric index |
| Qdrant | Native fastembed BM25 + built-in RRF | In-memory, local path, or remote |

## LLM clients

```python
from rag.llm.base import build_llm_client

llm = build_llm_client("anthropic", model="claude-opus-4-8")
llm = build_llm_client("gemini",    model="gemini-2.0-flash")
llm = build_llm_client("openai",    model="gpt-4o")  # default
```
