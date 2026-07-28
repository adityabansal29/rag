"""RED tests for Task 3: SQS infinite retry and graceful shutdown."""
import sys
import importlib
import pytest
from unittest.mock import MagicMock, patch


def _import_listener_with_mocks():
    """Import sqs_listener with all heavy side-effects patched."""
    mocks = {
        "boto3": MagicMock(),
        "dotenv": MagicMock(),
        "backend.worker.job_tracker": MagicMock(),
        "rag.hybrid.pipeline": MagicMock(),
        "rag.vectorstores.chroma_store": MagicMock(),
        "rag.vectorstores.pinecone_store": MagicMock(),
    }
    env_overrides = {"S3_BUCKET": "test-bucket", "SQS_QUEUE_URL": "https://sqs.test/q"}
    # Remove cached module so it re-imports with our patches
    for key in list(sys.modules.keys()):
        if "sqs_listener" in key:
            del sys.modules[key]

    with patch.dict("sys.modules", mocks), patch.dict("os.environ", env_overrides):
        import backend.worker.sqs_listener as listener
        return listener


class TestSQSRetry:
    def test_listen_deletes_failed_message(self):
        """listen() must call delete_message even when _process raises."""
        listener = _import_listener_with_mocks()

        sqs_mock = MagicMock()
        msg = {
            "Body": '{"s3_key": "x", "job_id": "j", "filename": "x.pdf"}',
            "MessageId": "m1",
            "ReceiptHandle": "rh-1",
        }
        sqs_mock.receive_message.side_effect = [
            {"Messages": [msg]},
            StopIteration("stop"),
        ]

        with patch.object(listener, "sqs", sqs_mock), \
             patch.object(listener, "QUEUE_URL", "q"), \
             patch.object(listener, "_process", side_effect=RuntimeError("bad")):
            try:
                listener.listen()
            except StopIteration:
                pass

        sqs_mock.delete_message.assert_called_once_with(
            QueueUrl="q",
            ReceiptHandle="rh-1",
        )

    def test_listen_deletes_successful_message(self):
        """listen() deletes the message on success too."""
        listener = _import_listener_with_mocks()

        sqs_mock = MagicMock()
        msg = {
            "Body": '{"s3_key": "x", "job_id": "j", "filename": "x.pdf"}',
            "MessageId": "m1",
            "ReceiptHandle": "rh-ok",
        }
        sqs_mock.receive_message.side_effect = [
            {"Messages": [msg]},
            StopIteration("stop"),
        ]

        with patch.object(listener, "sqs", sqs_mock), \
             patch.object(listener, "QUEUE_URL", "q"), \
             patch.object(listener, "_process", return_value=None):
            try:
                listener.listen()
            except StopIteration:
                pass

        sqs_mock.delete_message.assert_called_once_with(
            QueueUrl="q",
            ReceiptHandle="rh-ok",
        )
