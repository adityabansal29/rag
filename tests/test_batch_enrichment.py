"""Tests for Task 14: enricher must process children in bounded batches."""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock
from rag.enrichers.enricher import LLMEnricher
from rag.llm.base import BaseLLMClient
from rag.models import Chunk, ChunkType


def _make_parent(n_children: int) -> Chunk:
    parent = Chunk(
        id="parent-1",
        parent_id=None,
        raw_content="section text",
        chunk_type=ChunkType.TEXT,
        metadata={},
    )
    for i in range(n_children):
        child = Chunk(
            id=f"child-{i}",
            parent_id="parent-1",
            raw_content=f"content {i}",
            chunk_type=ChunkType.TEXT,
            metadata={},
        )
        parent.children.append(child)
    return parent


def _make_mock_llm(call_text_fn=None):
    """Return a minimal BaseLLMClient mock."""
    mock = MagicMock(spec=BaseLLMClient)
    if call_text_fn is None:
        mock.call_text = AsyncMock(return_value="summary")
    else:
        mock.call_text = call_text_fn
    return mock


class TestBatchEnrichment:
    def test_all_children_enriched(self):
        """enrich_all must enrich every child, regardless of batch size."""
        enricher = LLMEnricher(provider=_make_mock_llm(), max_concurrency=3)

        parent = _make_parent(10)
        asyncio.run(enricher.enrich_all([parent]))

        for child in parent.children:
            assert child.retrieved_content is not None, \
                f"child {child.id} was not enriched — possible silent drop in batching"

    def test_peak_concurrency_bounded(self):
        """Peak concurrent LLM calls must not exceed max_concurrency."""
        max_conc = 3
        peak = 0
        current = 0

        async def fake_call_text(system_prompt, content, response_model=None):
            nonlocal peak, current
            current += 1
            peak = max(peak, current)
            await asyncio.sleep(0.001)
            current -= 1
            return "enriched"

        enricher = LLMEnricher(provider=_make_mock_llm(fake_call_text), max_concurrency=max_conc)

        parent = _make_parent(0)
        # non-text children so LLM is actually called
        for i in range(20):
            child = Chunk(id=f"c{i}", parent_id="p0", raw_content=f"table {i}", chunk_type=ChunkType.TABLE, metadata={})
            parent.children.append(child)

        asyncio.run(enricher.enrich_all([parent]))

        assert peak <= max_conc, \
            f"peak concurrent LLM calls was {peak}, expected <= {max_conc} (max_concurrency)"
