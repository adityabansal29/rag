"""Shared mutable state initialised during app lifespan."""
import os

import boto3
from botocore.config import Config
from dotenv import load_dotenv

from rag.hybrid.pipeline import HybridRAGPipeline
from rag.fusion.pipeline import FusionRAGPipeline
from rag.rerankers.cross_encoder import CrossEncoderReranker
from rag.vectorstores.base import BaseVectorStore

load_dotenv(override=True)

AWS_REGION       = os.getenv("AWS_REGION", "us-east-1")
BUCKET           = os.environ["S3_BUCKET"]
QUEUE_URL        = os.environ["SQS_QUEUE_URL"]
PRESIGNED_EXPIRY = int(os.getenv("PRESIGNED_EXPIRY_SECONDS", "900"))
DYNAMO_TABLE     = os.getenv("DYNAMO_TABLE", "rag-jobs")

CHECKPOINTER_BACKEND   = os.getenv("LANGGRAPH_CHECKPOINTER", "sqlite")
CHECKPOINTER_DB        = os.getenv("CHECKPOINTER_DB", "./chat_history.db")
CHAT_CHECKPOINTS_TABLE = os.getenv("CHAT_CHECKPOINTS_TABLE", "rag-chat-checkpoints")
CHAT_WRITES_TABLE      = os.getenv("CHAT_WRITES_TABLE", "rag-chat-writes")

s3 = boto3.client(
    "s3",
    region_name=AWS_REGION,
    endpoint_url=f"https://s3.{AWS_REGION}.amazonaws.com",
    config=Config(signature_version="s3v4"),
)
sqs    = boto3.client("sqs",      region_name=AWS_REGION)
dynamo = boto3.client("dynamodb", region_name=AWS_REGION)

def _build_vectorstore() -> BaseVectorStore:
    api_key = os.getenv("PINECONE_API_KEY")
    if api_key:
        from rag.vectorstores.pinecone_store import PineconeVectorStore
        return PineconeVectorStore(
            api_key=api_key,
            index_name=os.getenv("PINECONE_INDEX_NAME", "rag-pipeline"),
        )
    from rag.vectorstores.chroma_store import ChromaVectorStore
    return ChromaVectorStore()

_hybrid  = HybridRAGPipeline(vectorstore=_build_vectorstore(), reranker=CrossEncoderReranker())
searcher = FusionRAGPipeline(_hybrid, n_queries=2)

# set during lifespan
agent: object | None = None
