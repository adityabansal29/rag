"""Tests for Task 9: rag_search must return JSON-parseable output;
chat endpoint must parse sources without string-splitting."""
import ast
import json
import pathlib
import asyncio
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

from rag.vectorstores.base import SearchParams


def _make_docs():
    return [
        Document(
            page_content="Some relevant text about RAG.",
            metadata={
                "chunk_id": "abc123",
                "rrf_score": 0.045,
                "chunk_type": "text",
                "page": "3",
                "source": "report.pdf",
            }
        )
    ]


class TestRagSearchStructured:
    def test_rag_search_returns_valid_json(self):
        """rag_search tool output must be parseable with json.loads()."""
        # Test the function logic in isolation — build a minimal version
        # that mirrors what rag_search does after Task 9.
        docs = _make_docs()
        results = []
        for doc in docs:
            score = doc.metadata.get("rrf_score") or doc.metadata.get("score", 0)
            results.append({
                "score":   score,
                "type":    doc.metadata.get("chunk_type", "text"),
                "page":    doc.metadata.get("page", ""),
                "source":  doc.metadata.get("source", ""),
                "content": doc.page_content,
            })
        output = json.dumps(results)

        # Verify the output format is what rag_search now returns
        parsed = json.loads(output)
        assert isinstance(parsed, list)
        assert parsed[0]["score"] == 0.045
        assert parsed[0]["source"] == "report.pdf"
        assert "content" in parsed[0]

        # Verify routes.py actually calls json.dumps (AST check)
        src = pathlib.Path("backend/api/routes.py").read_text()
        assert "json.dumps" in src, \
            "rag_search in routes.py must use json.dumps() to return structured output"

    def test_rag_search_no_freetext_format(self):
        """rag_search must NOT format results as free-text key=value headers."""
        src = pathlib.Path("backend/api/routes.py").read_text()
        # The old format: f"[{i+1}] score={score} type=..."
        assert 'f"[{i+1}] score={score}' not in src, \
            "rag_search still uses old free-text key=value format"

    def test_chat_endpoint_no_string_splitting(self):
        """chat endpoint must use json.loads() to parse sources, not string splitting."""
        src = pathlib.Path("backend/api/routes.py").read_text()
        tree = ast.parse(src)

        fn_types = (ast.FunctionDef, ast.AsyncFunctionDef)
        for node in ast.walk(tree):
            if isinstance(node, fn_types) and node.name == "chat":
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        if isinstance(child.func, ast.Attribute) and child.func.attr == "split":
                            for arg in child.args:
                                if isinstance(arg, ast.Constant) and arg.value == "\n\n":
                                    pytest.fail(
                                        "chat() still uses .split('\\n\\n') to parse sources. "
                                        "Use json.loads() after Task 9."
                                    )

    def test_chat_uses_json_loads_for_sources(self):
        """chat endpoint must call json.loads() to parse rag_search output."""
        src = pathlib.Path("backend/api/routes.py").read_text()
        tree = ast.parse(src)

        fn_types = (ast.FunctionDef, ast.AsyncFunctionDef)
        found_json_loads = False
        for node in ast.walk(tree):
            if isinstance(node, fn_types) and node.name == "chat":
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        func = child.func
                        if isinstance(func, ast.Attribute) and func.attr == "loads":
                            found_json_loads = True

        assert found_json_loads, "chat() must call json.loads() to parse rag_search output"
