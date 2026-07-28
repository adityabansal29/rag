"""Shared mutable state initialised during app lifespan."""
import os

from dotenv import load_dotenv

from backend.config import AWS_REGION, BUCKET, QUEUE_URL, s3, sqs, dynamo, build_vectorstore
from rag.hybrid.pipeline import HybridRAGPipeline
from rag.fusion.pipeline import FusionRAGPipeline
from rag.rerankers.cross_encoder import CrossEncoderReranker

load_dotenv(override=True)

PRESIGNED_EXPIRY = int(os.getenv("PRESIGNED_EXPIRY_SECONDS", "900"))
DYNAMO_TABLE     = os.getenv("DYNAMO_TABLE", "rag-jobs")

CHECKPOINTER_BACKEND   = os.getenv("LANGGRAPH_CHECKPOINTER", "sqlite")
CHECKPOINTER_DB        = os.getenv("CHECKPOINTER_DB", "./chat_history.db")
CHAT_CHECKPOINTS_TABLE = os.getenv("CHAT_CHECKPOINTS_TABLE", "rag-chat-checkpoints")
CHAT_WRITES_TABLE      = os.getenv("CHAT_WRITES_TABLE", "rag-chat-writes")

_hybrid  = HybridRAGPipeline(vectorstore=build_vectorstore(), reranker=CrossEncoderReranker())
searcher = FusionRAGPipeline(_hybrid, n_queries=2)

# set during lifespan
agent: object | None = None
