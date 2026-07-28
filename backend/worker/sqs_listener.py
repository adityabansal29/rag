import asyncio
import json
import os
import tempfile

import boto3
from dotenv import load_dotenv

from rag.hybrid.pipeline import HybridRAGPipeline
from backend.worker.job_tracker import JobTracker

load_dotenv(override=True)

BUCKET     = os.environ["S3_BUCKET"]
QUEUE_URL  = os.environ["SQS_QUEUE_URL"]
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

s3  = boto3.client("s3",  region_name=AWS_REGION)
sqs = boto3.client("sqs", region_name=AWS_REGION)


def _build_vectorstore():
    api_key = os.getenv("PINECONE_API_KEY")
    if api_key:
        from rag.vectorstores.pinecone_store import PineconeVectorStore
        return PineconeVectorStore(
            api_key=api_key,
            index_name=os.getenv("PINECONE_INDEX_NAME", "rag-pipeline"),
        )
    from rag.vectorstores.chroma_store import ChromaVectorStore
    return ChromaVectorStore()


pipeline = HybridRAGPipeline(vectorstore=_build_vectorstore())


def _process(msg: dict) -> None:
    body     = json.loads(msg["Body"])
    key      = body["s3_key"]
    job_id   = body.get("job_id", key)
    filename = body.get("filename", key.split("/")[-1])

    tracker = JobTracker(job_id=job_id, filename=filename, s3_key=key)
    tracker.start()

    suffix = "_" + filename
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        s3.download_fileobj(BUCKET, key, tmp)
        tmp_path = tmp.name

    try:
        asyncio.run(pipeline.run_async(tmp_path, on_step=tracker.on_step))
        tracker.finish()
        print(f"[worker] indexed: {key}")
    except Exception as e:
        tracker.fail(str(e))
        raise
    finally:
        os.unlink(tmp_path)


def listen() -> None:
    print(f"[worker] polling {QUEUE_URL}")
    while True:
        resp = sqs.receive_message(
            QueueUrl=QUEUE_URL,
            MaxNumberOfMessages=1,
            WaitTimeSeconds=20,
        )
        for msg in resp.get("Messages", []):
            try:
                _process(msg)
                sqs.delete_message(
                    QueueUrl=QUEUE_URL,
                    ReceiptHandle=msg["ReceiptHandle"],
                )
            except Exception as e:
                print(f"[worker] error processing {msg['MessageId']}: {e}")


if __name__ == "__main__":
    listen()
