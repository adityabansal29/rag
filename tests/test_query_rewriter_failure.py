"""Tests for Task 10: fusion degradation must be observable when rewriter fails."""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.documents import Document

from rag.fusion.query_rewriter import QueryRewriter
from rag.fusion.pipeline import FusionRAGPipeline
from rag.vectorstores.base import SearchParams


class TestQueryRewriterFailure:
    def test_rewriter_failure_signals_degraded(self):
        """When rewrite() fails, returned docs must carry fusion_degraded=True metadata."""
        mock_llm = MagicMock()
        mock_llm.call_text = AsyncMock(side_effect=RuntimeError("LLM down"))

        rewriter = QueryRewriter(mock_llm)

        # Make a minimal FusionRAGPipeline
        pipeline = FusionRAGPipeline.__new__(FusionRAGPipeline)
        pipeline.n_queries = 2
        pipeline.rewriter = rewriter

        inner = MagicMock()
        inner.reranker = None

        doc = Document(page_content="result", metadata={"chunk_id": "x"})

        async def fake_dense(query, params=None):
            return [doc]

        async def fake_bm25(query, params=None):
            return [doc]

        inner.dense_search_async = fake_dense
        inner.bm25_search_async = fake_bm25
        pipeline.pipeline = inner

        params = SearchParams(top_k=2, use_hybrid=False)
        _, docs = asyncio.run(pipeline.search_async("what is X?", params))

        assert docs, "search must return results even when rewriter fails"
        for doc in docs:
            assert doc.metadata.get("fusion_degraded") is True, \
                "docs must carry fusion_degraded=True when rewriter failed"

    def test_rewriter_success_no_degraded_flag(self):
        """When rewrite() succeeds, docs must NOT carry fusion_degraded."""
        mock_llm = MagicMock()

        async def good_rewrite(system_prompt, content, response_model=None):
            m = MagicMock()
            m.queries = ["variant A", "variant B"]
            return m

        mock_llm.call_text = good_rewrite

        rewriter = QueryRewriter(mock_llm)
        pipeline = FusionRAGPipeline.__new__(FusionRAGPipeline)
        pipeline.n_queries = 2
        pipeline.rewriter = rewriter

        inner = MagicMock()
        inner.reranker = None

        doc = Document(page_content="result", metadata={"chunk_id": "x"})

        async def fake_dense(query, params=None):
            return [doc]

        inner.dense_search_async = fake_dense
        pipeline.pipeline = inner

        params = SearchParams(top_k=2, use_hybrid=False)
        _, docs = asyncio.run(pipeline.search_async("what is X?", params))

        for doc in docs:
            assert doc.metadata.get("fusion_degraded") is not True, \
                "docs must NOT carry fusion_degraded when rewriter succeeded"
