"""Shared mutable state initialised during app lifespan."""
import json
import os

from dotenv import load_dotenv
from langchain_core.tools import tool

from backend.config import AWS_REGION, BUCKET, QUEUE_URL, s3, sqs, dynamo, build_vectorstore
from rag.hybrid.pipeline import HybridRAGPipeline
from rag.fusion.pipeline import FusionRAGPipeline
from rag.rerankers.cross_encoder import CrossEncoderReranker
from rag.vectorstores.base import SearchParams

load_dotenv(override=True)

PRESIGNED_EXPIRY = int(os.getenv("PRESIGNED_EXPIRY_SECONDS", "900"))
DYNAMO_TABLE     = os.getenv("DYNAMO_TABLE", "rag-jobs")

CHECKPOINTER_BACKEND   = os.getenv("LANGGRAPH_CHECKPOINTER", "sqlite")
CHECKPOINTER_DB        = os.getenv("CHECKPOINTER_DB", "./chat_history.db")
CHAT_CHECKPOINTS_TABLE = os.getenv("CHAT_CHECKPOINTS_TABLE", "rag-chat-checkpoints")
CHAT_WRITES_TABLE      = os.getenv("CHAT_WRITES_TABLE", "rag-chat-writes")

_hybrid  = HybridRAGPipeline(vectorstore=build_vectorstore(), reranker=CrossEncoderReranker())
searcher = FusionRAGPipeline(_hybrid, n_queries=2)

@tool
async def rag_search(query: str) -> str:
    """Search the knowledge base using fusion RAG and return relevant chunks with scores."""
    _, chunks = await searcher.search_async(query, params=SearchParams(top_k=4, use_hybrid=True))
    if not chunks:
        return "No relevant information found."
    results = []
    for doc in chunks:
        score = doc.metadata.get("rrf_score") or doc.metadata.get("score", 0)
        results.append({
            "score":   score,
            "type":    doc.metadata.get("chunk_type", "text"),
            "page":    doc.metadata.get("page", ""),
            "source":  doc.metadata.get("source", ""),
            "content": doc.page_content,
        })
    return json.dumps(results)


# set during lifespan
agent: object | None = None
