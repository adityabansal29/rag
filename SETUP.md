# Setup Guide

## Prerequisites

- Python 3.12+
- Node.js 18+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (Python package manager)
- AWS account with credentials configured (`aws configure` or env vars)
- OpenAI API key

---

## 1. AWS Infrastructure

You need three AWS resources. Create them in the same region.

### S3 Bucket

Create a bucket for document uploads. Note the bucket name.

### SQS Queue

Create a standard queue (not FIFO). Note the queue URL.

### DynamoDB Tables

Run the setup script after completing step 2 (Python environment):

```bash
python backend/scripts/setup_dynamo.py
```

This creates three tables (idempotent — safe to re-run):
- `rag-jobs` — ingestion job tracking
- `rag-chat-checkpoints` — LangGraph conversation checkpoints
- `rag-chat-writes` — LangGraph checkpoint writes

---

## 2. Python Environment

```bash
uv sync
uv pip install -e .
```

This installs all dependencies and the `rag` library in editable mode.

---

## 3. Environment Variables

Copy and fill in `.env` at the project root:

```bash
cp .env.example .env
```

Required variables:

```env
# OpenAI
OPENAI_API_KEY=sk-...

# AWS
AWS_REGION=us-east-1
S3_BUCKET=your-bucket-name
SQS_QUEUE_URL=https://sqs.us-east-1.amazonaws.com/123456789/your-queue

# DynamoDB
DYNAMO_TABLE=rag-jobs

# LangGraph checkpointer
# Use "sqlite" for local dev (no AWS needed), "dynamodb" for production
LANGGRAPH_CHECKPOINTER=dynamodb
CHAT_CHECKPOINTS_TABLE=rag-chat-checkpoints
CHAT_WRITES_TABLE=rag-chat-writes
```

Optional:

```env
PRESIGNED_EXPIRY_SECONDS=900   # S3 presigned URL TTL (default: 15 min)
CHECKPOINTER_DB=./chat_history.db  # SQLite path (only used when LANGGRAPH_CHECKPOINTER=sqlite)
```

---

## 4. Create DynamoDB Tables

```bash
source .venv/bin/activate
python backend/scripts/setup_dynamo.py
```

---

## 5. Frontend

```bash
cd frontend
npm install
```

---

## 6. Run

Each of these runs in a separate terminal.

```bash
# Terminal 1 — API server
bash run.sh api

# Terminal 2 — SQS worker (picks up ingestion jobs)
bash run.sh worker

# Terminal 3 — Frontend
bash run.sh ui
```

Open [http://localhost:3000](http://localhost:3000).

---

## 7. Test an Upload

```bash
bash run.sh upload ./path/to/document.pdf
```

This presigns an S3 URL, uploads the file, and notifies the API to enqueue the job. Watch the worker terminal for progress, or open `/jobs` in the UI.

---

## Vector Store

The default vector store is **Chroma** (local, no extra setup). To switch:

**Qdrant (local)**
```python
from rag.vectorstores.qdrant_store import QdrantVectorStore
pipeline = HybridRAGPipeline(vectorstore=QdrantVectorStore(path="./qdrant_db", enable_hybrid=True))
```

**Pinecone**
```python
from rag.vectorstores.pinecone_store import PineconeVectorStore
pipeline = HybridRAGPipeline(vectorstore=PineconeVectorStore(api_key="...", index_name="rag", enable_hybrid=True))
```

Pinecone requires a `dotproduct` metric index for hybrid search.

---

## Troubleshooting

**`OPENAI_API_KEY` not set** — check `.env` is in the project root and `load_dotenv()` runs before any imports that need it.

**SQS messages not processing** — confirm the worker is running and `SQS_QUEUE_URL` matches exactly (including region).

**DynamoDB access denied** — your AWS credentials need `dynamodb:PutItem`, `dynamodb:UpdateItem`, `dynamodb:GetItem`, `dynamodb:Scan` on the three tables.

**Port 8000 already in use** — find and kill the old process: `lsof -ti:8000 | xargs kill`
