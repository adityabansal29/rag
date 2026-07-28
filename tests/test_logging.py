"""Tests for Task 12: production code must use logging, not print()."""
import ast
import pathlib
import pytest


PRODUCTION_MODULES = [
    "rag/fusion/pipeline.py",
    "rag/fusion/query_rewriter.py",
    "rag/enrichers/enricher.py",
    "rag/rerankers/cross_encoder.py",
    "rag/rerankers/cohere.py",
    "rag/vectorstores/chroma_store.py",
    "rag/vectorstores/pinecone_store.py",
    "backend/worker/sqs_listener.py",
]


def _has_bare_print(path: str) -> list[int]:
    src = pathlib.Path(path).read_text()
    tree = ast.parse(src)
    lines = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == "print":
            lines.append(node.lineno)
    return lines


@pytest.mark.parametrize("module_path", PRODUCTION_MODULES)
def test_no_bare_print(module_path):
    """Production modules must not use bare print() — use logging.getLogger instead."""
    offending = _has_bare_print(module_path)
    assert not offending, (
        f"{module_path} still has print() on lines {offending}. "
        "Use logging.getLogger(__name__).debug/info/warning instead."
    )
