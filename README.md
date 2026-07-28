# Production RAG System

A full-stack document QA system implementing advanced RAG patterns end-to-end: **hybrid BM25 + dense retrieval with RRF**, **RAG-Fusion with LLM query rewriting**, **cross-encoder re-ranking**, **parent-document retrieval with hierarchical chunking**, and a **multi-turn conversational agent** with persistent memory via LangGraph.

---

## System Overview

```
                         ┌──────────────────────────────────────┐
                         │             Browser                  │
                         │    Upload · Jobs · Chat              │
                         └────────────┬────────────┬────────────┘
                                      │            │
          ─── INGESTION ──────────────┘            └── CHAT ───
                                                                               
  Browser → S3 presigned PUT                   POST /chat                      
         → POST /notify                              │                         
         → DynamoDB (queued) + SQS            LangGraph ReAct agent            
                     │                               │                         
                SQS Worker                     rag_search tool                 
                     │                               │                         
   parse → chunk → LLM-enrich → embed         FusionRAGPipeline               
                     │                               │                         
           Chroma / Pinecone / Qdrant        QueryRewriter  (1 LLM call)       
                     ▲                               │                         
                     │              BM25(standalone) + dense(all variants)    
                     │                               │                         
                     └───────────────────── Outer RRF merge                   
                                                     │                         
                                            [CrossEncoder re-rank]             
                                                     │                         
                                            top_k chunks → LLM answer         
```

---

## Advanced RAG Techniques

| Technique | Where |
|-----------|-------|
| **Hybrid retrieval** — BM25 + dense vectors fused with Reciprocal Rank Fusion | `rag/hybrid/` |
| **RAG-Fusion** — query rewriting + parallel multi-query search + outer RRF | `rag/fusion/` |
| **Cross-encoder re-ranking** — sentence-transformers (local) or Cohere API | `rag/rerankers/` |
| **Parent-document retrieval** — children indexed for precision, parent context injected at generation | `rag/chunkers/`, `rag/enrichers/` |
| **Hierarchical chunking** — merge small / split large text (100–512 tokens); non-text preserved | `rag/chunkers/` |
| **Contextual chunk enrichment** — LLM captioning for images, tables, and code blocks | `rag/enrichers/` |
| **LangGraph ReAct agent** — tool-calling loop with thread-persistent conversation memory | `backend/api/` |
| **Multi-vector store** — Chroma, Pinecone, Qdrant via swappable interface | `rag/vectorstores/` |
| **LLM-as-evaluator** — chunk relevance + answer faithfulness scoring | `rag/evaluators/` |
| **Multi-provider LLM** — OpenAI, Anthropic, Gemini via unified client interface | `rag/llm/` |
| **Async SQS-driven ingestion** — decoupled upload → process pipeline | `backend/worker/` |

---

## Stack

**Backend** Python 3.11 · FastAPI · LangGraph · LangChain · asyncio  
**Ingestion** AWS S3 · SQS · DynamoDB  
**Embeddings** OpenAI `text-embedding-3-small`  
**Vector stores** Chroma (default) · Pinecone · Qdrant  
**Re-rankers** `ms-marco-MiniLM-L-6-v2` (local) · Cohere `rerank-english-v3.0` (API)  
**LLM** OpenAI GPT-4o · Anthropic Claude · Google Gemini  
**Frontend** Next.js 16 · TypeScript · Tailwind CSS  

---

## Project Layout

```
rag/          # core library — parsers, chunkers, enrichers, embedders, pipelines
backend/      # FastAPI + SQS worker + LangGraph agent
frontend/     # Next.js UI — upload, job tracking, chat
tests/        # pytest suite
```

---

## Setup

**Prerequisites:** Python 3.12+, Node.js 18+, [uv](https://docs.astral.sh/uv/getting-started/installation/), AWS account, OpenAI API key.

```bash
uv sync && uv pip install -e .
cd frontend && npm install
```

Copy `.env.example` to `.env`:

```
OPENAI_API_KEY=sk-...

AWS_REGION=us-east-1
S3_BUCKET=your-bucket
SQS_QUEUE_URL=https://sqs.us-east-1.amazonaws.com/123/your-queue
DYNAMO_TABLE=rag-jobs

LANGGRAPH_CHECKPOINTER=sqlite        # or dynamodb for prod
CHECKPOINTER_DB=./chat_history.db    # sqlite only
CHAT_CHECKPOINTS_TABLE=rag-chat-checkpoints
CHAT_WRITES_TABLE=rag-chat-writes

PRESIGNED_EXPIRY_SECONDS=900         # optional, default 15 min
```

```bash
python backend/scripts/setup_dynamo.py   # creates rag-jobs, rag-chat-checkpoints, rag-chat-writes
```

## Run

```bash
bash run.sh api       # FastAPI on :8000
bash run.sh worker    # SQS polling loop
bash run.sh ui        # Next.js on :3000

bash run.sh upload ./paper.pdf   # upload a document from the CLI
```

---

## Retrieval

See [`rag/hybrid/README.md`](rag/hybrid/README.md) for hybrid BM25 + dense search, RRF merging, and re-ranking.

See [`rag/fusion/README.md`](rag/fusion/README.md) for query rewriting, multi-query fusion, and why BM25 runs exactly once.

---

## Conversational Agent

A LangGraph ReAct agent exposes `rag_search` (a LangChain `@tool` wrapping `FusionRAGPipeline`) and loops until it has enough context to answer. Conversation history is maintained by a LangGraph checkpointer — DynamoDB in production, SQLite in dev — keyed by `thread_id`. The browser stores `thread_id` in `sessionStorage`: stateless client, stateful server.
