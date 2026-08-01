"""Shared AWS config and vectorstore factory — single source of truth for both API and worker."""
import os

import boto3
from botocore.config import Config
from dotenv import load_dotenv

from rag.vectorstores.base import BaseVectorStore

load_dotenv(override=True)

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

_missing = [v for v in ("S3_BUCKET", "SQS_QUEUE_URL") if not os.getenv(v)]
if _missing:
    raise RuntimeError(
        f"Missing required environment variables: {', '.join(_missing)}. "
        "Set them in .env or the environment before starting."
    )

BUCKET    = os.environ["S3_BUCKET"]
QUEUE_URL = os.environ["SQS_QUEUE_URL"]

s3 = boto3.client(
    "s3",
    region_name=AWS_REGION,
    endpoint_url=f"https://s3.{AWS_REGION}.amazonaws.com",
    config=Config(signature_version="s3v4"),
)
sqs    = boto3.client("sqs",      region_name=AWS_REGION)
dynamo = boto3.client("dynamodb", region_name=AWS_REGION)


def build_vectorstore() -> BaseVectorStore:
    if os.getenv("PINECONE_API_KEY"):
        from rag.vectorstores.pinecone_store import PineconeVectorStore
        return PineconeVectorStore(
            api_key=os.environ["PINECONE_API_KEY"],
            index_name=os.getenv("PINECONE_INDEX_NAME", "rag-pipeline"),
            region=os.getenv("PINECONE_REGION", "us-east-1"),
            enable_hybrid=True,
        )
    if os.getenv("QDRANT_URL"):
        from rag.vectorstores.qdrant_store import QdrantVectorStore
        return QdrantVectorStore(
            host=os.environ["QDRANT_URL"],
            port=int(os.getenv("QDRANT_PORT", "6333")),
            collection_name=os.getenv("QDRANT_COLLECTION", "rag"),
            enable_hybrid=True,
        )
    from rag.vectorstores.chroma_store import ChromaVectorStore
    return ChromaVectorStore()
