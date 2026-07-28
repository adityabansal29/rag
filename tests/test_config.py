"""Tests for Task 4: shared config module and env var validation."""
import os
import sys
import pytest
from unittest.mock import patch, MagicMock


def _fresh_import(env_overrides: dict, clear: bool = False):
    """Import backend.config fresh with controlled env vars.
    Mocks load_dotenv (no .env file) and boto3 clients.
    """
    for k in list(sys.modules):
        if k in ("backend.config", "backend"):
            del sys.modules[k]

    base_env = {} if clear else {k: v for k, v in os.environ.items()}
    base_env.update(env_overrides)

    with patch.dict("os.environ", base_env, clear=True), \
         patch("dotenv.load_dotenv"), \
         patch("boto3.client", return_value=MagicMock()), \
         patch("botocore.config.Config"):
        import backend.config as cfg
        return cfg


class TestEnvVarValidation:
    def test_missing_s3_bucket_raises_clear_error(self):
        """Missing S3_BUCKET must raise RuntimeError with a helpful message, not KeyError."""
        for k in list(sys.modules):
            if "backend.config" in k:
                del sys.modules[k]
        with pytest.raises(RuntimeError, match="S3_BUCKET"):
            _fresh_import({"SQS_QUEUE_URL": "https://sqs.test/q"}, clear=True)

    def test_missing_queue_url_raises_clear_error(self):
        """Missing SQS_QUEUE_URL must raise RuntimeError with a helpful message."""
        with pytest.raises(RuntimeError, match="SQS_QUEUE_URL"):
            _fresh_import({"S3_BUCKET": "bucket"}, clear=True)

    def test_valid_env_provides_bucket_and_queue(self):
        """With both vars set, the module exposes BUCKET and QUEUE_URL."""
        cfg = _fresh_import(
            {"S3_BUCKET": "my-bucket", "SQS_QUEUE_URL": "https://sqs.test/q"},
            clear=True,
        )
        assert cfg.BUCKET == "my-bucket"
        assert cfg.QUEUE_URL == "https://sqs.test/q"


class TestBuildVectorstore:
    def test_build_vectorstore_exported(self):
        """`build_vectorstore` must be importable from backend.config."""
        cfg = _fresh_import(
            {"S3_BUCKET": "b", "SQS_QUEUE_URL": "q"},
            clear=True,
        )
        assert callable(getattr(cfg, "build_vectorstore", None)), \
            "backend.config must export build_vectorstore()"

    def test_single_definition_no_duplicate_in_state_or_worker(self):
        """state.py and sqs_listener.py must NOT define _build_vectorstore themselves."""
        import ast, pathlib
        for path in ("backend/api/state.py", "backend/worker/sqs_listener.py"):
            src = pathlib.Path(path).read_text()
            tree = ast.parse(src)
            fn_names = [
                node.name for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef)
            ]
            assert "_build_vectorstore" not in fn_names, \
                f"{path} still defines _build_vectorstore — should import from backend.config"
            assert "build_vectorstore" not in fn_names, \
                f"{path} still defines build_vectorstore — should import from backend.config"
