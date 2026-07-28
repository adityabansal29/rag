import asyncio
import json
import os
import signal
import tempfile

from dotenv import load_dotenv

from backend.config import BUCKET, QUEUE_URL, s3, sqs, build_vectorstore
from backend.worker.job_tracker import JobTracker
from rag.hybrid.pipeline import HybridRAGPipeline

load_dotenv(override=True)

pipeline = HybridRAGPipeline(vectorstore=build_vectorstore())


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
    shutdown = False

    def _handle_signal(signum, frame):
        nonlocal shutdown
        print(f"[worker] signal {signum} received — draining current message then exiting")
        shutdown = True

    signal.signal(signal.SIGINT,  _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    print(f"[worker] polling {QUEUE_URL}")
    while not shutdown:
        resp = sqs.receive_message(
            QueueUrl=QUEUE_URL,
            MaxNumberOfMessages=1,
            WaitTimeSeconds=20,
        )
        for msg in resp.get("Messages", []):
            try:
                _process(msg)
            except Exception as e:
                print(f"[worker] error processing {msg['MessageId']}: {e}")
            finally:
                # Always delete — on failure, SQS would re-deliver indefinitely otherwise.
                # Permanent failures should be caught by a DLQ, not re-delivery loops.
                sqs.delete_message(
                    QueueUrl=QUEUE_URL,
                    ReceiptHandle=msg["ReceiptHandle"],
                )


if __name__ == "__main__":
    listen()
