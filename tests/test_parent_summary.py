"""Tests for Task 8: parent retrieved_content must be stored in child metadata."""
import pytest
from langchain_core.documents import Document

from rag.models import Chunk, ChunkType
from rag.hybrid.pipeline import chunks_to_langchain_docs


def _make_parent(parent_summary: str | None, n_children: int = 2) -> Chunk:
    parent = Chunk(
        id="parent-1",
        parent_id=None,
        chunk_type=ChunkType.TEXT,
        metadata={"source": "test.pdf", "section_title": "Intro"},
        raw_content="Intro heading",
        retrieved_content=parent_summary,
    )
    for i in range(n_children):
        child = Chunk(
            id=f"child-{i}",
            parent_id="parent-1",
            chunk_type=ChunkType.TEXT,
            metadata={"source": "test.pdf", "page": 1},
            raw_content=f"child text {i}",
            embedding_content=f"Section: Intro\nContent: child text {i}",
        )
        parent.add_child(child)
    return parent


def test_child_docs_have_parent_summary():
    """Each child LangChain doc must carry parent_summary in its metadata."""
    parent = _make_parent("This is a rich section summary from the LLM.")
    docs = chunks_to_langchain_docs([parent])

    assert len(docs) == 2
    for doc in docs:
        assert "parent_summary" in doc.metadata, \
            "child doc metadata must have parent_summary key"
        assert doc.metadata["parent_summary"] == "This is a rich section summary from the LLM."


def test_child_docs_empty_parent_summary_when_none():
    """If parent has no retrieved_content, parent_summary should be empty string."""
    parent = _make_parent(parent_summary=None)
    docs = chunks_to_langchain_docs([parent])

    for doc in docs:
        assert doc.metadata.get("parent_summary", "MISSING") == "", \
            "parent_summary must be '' (not missing, not None) when parent has no summary"


def test_generate_answer_includes_parent_context():
    """generate_answer_async must include parent_summary in context if present."""
    import asyncio
    from unittest.mock import AsyncMock, MagicMock
    from rag.hybrid.pipeline import HybridRAGPipeline

    pipeline = HybridRAGPipeline.__new__(HybridRAGPipeline)
    mock_llm = MagicMock()
    captured_content = []

    async def fake_call_text(system_prompt, content):
        captured_content.append(content)
        return "answer"

    mock_llm.call_text = fake_call_text
    pipeline.enricher = MagicMock()
    pipeline.enricher.llm = mock_llm

    docs = [
        Document(
            page_content="child content about topic X",
            metadata={"parent_summary": "Parent section covers topics X, Y, Z in depth."},
        )
    ]

    asyncio.run(pipeline.generate_answer_async("What is topic X?", docs))

    assert captured_content, "call_text was never called"
    context_sent = captured_content[0]
    assert "Parent section covers topics X, Y, Z in depth." in context_sent, \
        "generate_answer_async must include parent_summary in context"
