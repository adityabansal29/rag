import asyncio
import json
import logging
import os
import signal
import tempfile

logger = logging.getLogger(__name__)

from dotenv import load_dotenv

from backend.config import BUCKET, QUEUE_URL, s3, sqs, build_vectorstore
from backend.worker.job_tracker import JobTracker
from rag.hybrid.pipeline import HybridRAGPipeline

load_dotenv(override=True)

pipeline = HybridRAGPipeline(vectorstore=build_vectorstore())


async def _process(msg: dict) -> None:
    body     = json.loads(msg["Body"])
    key      = body["s3_key"]
    job_id   = body.get("job_id", key)
    filename = body.get("filename", key.split("/")[-1])

    tracker = JobTracker(job_id=job_id, filename=filename, s3_key=key)
    tracker.start()

    suffix = "_" + filename
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        await asyncio.to_thread(s3.download_fileobj, BUCKET, key, tmp)
        tmp_path = tmp.name

    try:
        await pipeline.run_async(tmp_path, on_step=tracker.on_step)
        tracker.finish()
        logger.info("[worker] indexed: %s", key)
    except Exception as e:
        tracker.fail(str(e))
        raise
    finally:
        os.unlink(tmp_path)


async def listen() -> None:
    shutdown = False

    def _handle_signal(signum, frame):
        nonlocal shutdown
        logger.info("[worker] signal %s received — draining current message then exiting", signum)
        shutdown = True

    signal.signal(signal.SIGINT,  _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    logger.info("[worker] polling %s", QUEUE_URL)
    while not shutdown:
        resp = await asyncio.to_thread(
            sqs.receive_message,
            QueueUrl=QUEUE_URL,
            MaxNumberOfMessages=1,
            WaitTimeSeconds=20,
        )
        for msg in resp.get("Messages", []):
            try:
                await _process(msg)
            except Exception as e:
                logger.error("[worker] error processing %s: %s", msg['MessageId'], e)
            finally:
                # Always delete — on failure, SQS would re-deliver indefinitely otherwise.
                await asyncio.to_thread(
                    sqs.delete_message,
                    QueueUrl=QUEUE_URL,
                    ReceiptHandle=msg["ReceiptHandle"],
                )


if __name__ == "__main__":
    from backend.logging_config import configure_logging
    configure_logging()
    asyncio.run(listen())
