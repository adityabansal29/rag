"""Tests for SQS worker: message deletion behavior."""
import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import sys
import os as _os


# Import once with all network-touching code patched
_mocks = {
    "boto3": MagicMock(),
    "dotenv": MagicMock(),
    "backend.worker.job_tracker": MagicMock(),
    "rag.hybrid.pipeline": MagicMock(),
    "backend.config": MagicMock(
        BUCKET="bucket", QUEUE_URL="q",
        s3=MagicMock(), sqs=MagicMock(),
        build_vectorstore=MagicMock(return_value=MagicMock()),
    ),
}
for _k in list(sys.modules):
    if "backend.worker.sqs_listener" in _k:
        del sys.modules[_k]

with patch.dict("sys.modules", _mocks), patch.dict(_os.environ, {"S3_BUCKET": "b", "SQS_QUEUE_URL": "q"}):
    import backend.worker.sqs_listener as _listener


class _StopLoop(Exception):
    """Sentinel to break the infinite listen() loop in tests."""


class TestSQSRetry:
    """Test that listen() deletes messages regardless of _process outcome."""

    def _run_one_cycle(self, msgs, process_side_effect):
        """Run listen() for exactly one receive+process+delete cycle, then stop."""
        sqs_mock = MagicMock()
        call_count = [0]

        def receive(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return {"Messages": msgs}
            raise _StopLoop()

        sqs_mock.receive_message.side_effect = receive

        async def run():
            with patch.object(_listener, "sqs", sqs_mock), \
                 patch.object(_listener, "QUEUE_URL", "q"), \
                 patch.object(_listener, "_process", side_effect=process_side_effect):
                try:
                    await _listener.listen()
                except _StopLoop:
                    pass

        asyncio.run(run())
        return sqs_mock

    def test_listen_deletes_failed_message(self):
        msg = {
            "Body": '{"s3_key": "x", "job_id": "j", "filename": "x.pdf"}',
            "MessageId": "m1",
            "ReceiptHandle": "rh-1",
        }
        sqs_mock = self._run_one_cycle([msg], process_side_effect=RuntimeError("bad"))
        sqs_mock.delete_message.assert_called_once_with(QueueUrl="q", ReceiptHandle="rh-1")

    def test_listen_deletes_successful_message(self):
        msg = {
            "Body": '{"s3_key": "x", "job_id": "j", "filename": "x.pdf"}',
            "MessageId": "m1",
            "ReceiptHandle": "rh-ok",
        }
        async def noop(_msg):
            pass

        sqs_mock = self._run_one_cycle([msg], process_side_effect=noop)
        sqs_mock.delete_message.assert_called_once_with(QueueUrl="q", ReceiptHandle="rh-ok")
