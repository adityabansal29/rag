"""RED tests for Task 5: worker must be fully async — no asyncio.run() inside _process."""
import ast
import pathlib
import sys
import pytest
from unittest.mock import patch, MagicMock, AsyncMock


def test_no_asyncio_run_in_process():
    """_process must not call asyncio.run() — creates a new event loop per message."""
    src = pathlib.Path("backend/worker/sqs_listener.py").read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_process":
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    func = child.func
                    # Check for asyncio.run(...)
                    if isinstance(func, ast.Attribute) and func.attr == "run":
                        if isinstance(func.value, ast.Name) and func.value.id == "asyncio":
                            pytest.fail("_process() calls asyncio.run() — must use await instead")


def test_process_is_async():
    """_process must be defined as async def."""
    src = pathlib.Path("backend/worker/sqs_listener.py").read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "_process":
            return  # found it
    pytest.fail("_process is not defined as async def")


def test_listen_is_async():
    """listen() must be async so a single event loop covers the worker lifetime."""
    src = pathlib.Path("backend/worker/sqs_listener.py").read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "listen":
            return
    pytest.fail("listen() is not defined as async def")


def test_process_awaits_pipeline(tmp_path):
    """_process must await pipeline.run_async (not asyncio.run it)."""
    import sys
    for k in list(sys.modules):
        if "sqs_listener" in k:
            del sys.modules[k]

    mock_pipeline = MagicMock()
    mock_pipeline.run_async = AsyncMock(return_value=[])
    mock_tracker = MagicMock()
    mock_tracker.on_step = MagicMock()

    env = {"S3_BUCKET": "b", "SQS_QUEUE_URL": "q"}
    mocks = {
        "boto3": MagicMock(),
        "dotenv": MagicMock(),
        "backend.worker.job_tracker": MagicMock(),
        "rag.hybrid.pipeline": MagicMock(),
        "backend.config": MagicMock(
            BUCKET="b", QUEUE_URL="q",
            s3=MagicMock(), sqs=MagicMock(),
            build_vectorstore=MagicMock(return_value=MagicMock()),
        ),
    }

    import asyncio
    with patch.dict("sys.modules", mocks), patch.dict("os.environ", env):
        import backend.worker.sqs_listener as listener
        listener.pipeline = mock_pipeline

        # Create a real temp file
        tmp_file = tmp_path / "test.pdf"
        tmp_file.write_bytes(b"pdf")

        msg = {
            "Body": '{"s3_key": "uploads/x.pdf", "job_id": "j1", "filename": "x.pdf"}',
            "MessageId": "m1",
            "ReceiptHandle": "rh",
        }

        with patch.object(listener, "s3") as s3_mock, \
             patch("backend.worker.sqs_listener.JobTracker") as jt_cls, \
             patch("tempfile.NamedTemporaryFile") as ntf_mock, \
             patch("os.unlink"):
            jt_cls.return_value = mock_tracker
            # Simulate NamedTemporaryFile context
            ntf_mock.return_value.__enter__ = MagicMock(return_value=MagicMock(name=str(tmp_file)))
            ntf_mock.return_value.__exit__ = MagicMock(return_value=False)
            ntf_mock.return_value.__enter__.return_value.name = str(tmp_file)

            asyncio.run(listener._process(msg))

        mock_pipeline.run_async.assert_awaited_once()
