"""
RED tests for Task 1: vector store score threshold consistency.

Issue #3: ChromaVectorStore.bm25_search ignores rrf_score_threshold.
Issue #4: PineconeVectorStore.search applies cosine_threshold to dotproduct scores.
"""
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

from rag.vectorstores.base import SearchParams


# ── Issue #3: BM25 threshold filtering in ChromaVectorStore ──────────────────

class TestChromaBM25Threshold:
    def _make_store(self):
        """Return a ChromaVectorStore with a fake collection of 5 docs."""
        from rag.vectorstores.chroma_store import ChromaVectorStore
        store = ChromaVectorStore.__new__(ChromaVectorStore)
        store._bm25_cache = None

        docs = [
            Document(page_content=f"doc {i}", metadata={"chunk_id": f"id{i}"})
            for i in range(5)
        ]

        # Fake BM25Retriever that returns all docs ranked in order
        fake_retriever = MagicMock()
        fake_retriever.invoke.return_value = docs
        fake_retriever.k = 5

        # Patch _get_bm25_retriever to return our fake
        store._get_bm25_retriever = MagicMock(return_value=fake_retriever)

        # Patch collection.count to return non-zero
        store.collection = MagicMock()
        store.collection.count.return_value = 5

        return store, docs

    def test_bm25_no_threshold_returns_all(self):
        store, docs = self._make_store()
        params = SearchParams(top_k=5, rrf_score_threshold=None)
        results = store.bm25_search("query", params)
        assert len(results) == 5

    def test_bm25_rrf_threshold_filters_low_ranked(self):
        """RRF score for rank 0 = 1/60 ≈ 0.0167; rank 4 = 1/64 ≈ 0.0156.
        With threshold of 0.0165, rank 0 passes, ranks 1-4 are filtered."""
        store, docs = self._make_store()
        # threshold just below rank-0 RRF score (1/60 ≈ 0.01667)
        params = SearchParams(top_k=5, rrf_score_threshold=0.0165)
        results = store.bm25_search("query", params)
        # Only rank-0 doc (rrf=1/60=0.01667) passes; ranks 1-4 (rrf≤1/61) don't
        assert len(results) == 1

    def test_bm25_zero_threshold_returns_all(self):
        store, docs = self._make_store()
        params = SearchParams(top_k=5, rrf_score_threshold=0.0)
        results = store.bm25_search("query", params)
        assert len(results) == 5


# ── Issue #4: Pinecone hybrid should not apply cosine_threshold ───────────────

class TestPineconeHybridThreshold:
    def _make_store(self, enable_hybrid: bool):
        from rag.vectorstores.pinecone_store import PineconeVectorStore
        store = PineconeVectorStore.__new__(PineconeVectorStore)
        store.namespace = "test"
        store.enable_hybrid = enable_hybrid

        # Fake index returning matches with dotproduct scores > 1 (valid for dotproduct)
        fake_matches = [
            {"id": "id0", "score": 2.5, "metadata": {"chunk_id": "id0", "text": "doc 0"}},
            {"id": "id1", "score": 0.3, "metadata": {"chunk_id": "id1", "text": "doc 1"}},
        ]
        store.index = MagicMock()
        store.index.query.return_value = {"matches": fake_matches}

        if enable_hybrid:
            store._sparse_encoder = MagicMock()
            store._sparse_encoder.encode_queries.return_value = {"indices": [0], "values": [1.0]}

        return store

    def test_dense_cosine_threshold_filters(self):
        """In dense mode, cosine_threshold should filter low scores."""
        store = self._make_store(enable_hybrid=False)
        # score 0.3 is below threshold 0.6 → only id0 passes
        params = SearchParams(top_k=2, cosine_threshold=0.6, use_hybrid=False)
        results = store.search([0.1] * 1536, query_text="q", params=params)
        assert len(results) == 1
        assert results[0].metadata["chunk_id"] == "id0"

    def test_hybrid_dotproduct_ignores_cosine_threshold(self):
        """In hybrid mode (dotproduct metric), cosine_threshold must NOT filter.
        Score 2.5 and 0.3 are both valid dotproduct scores; threshold is meaningless."""
        store = self._make_store(enable_hybrid=True)
        # cosine_threshold=0.6 would wrongly drop score=0.3 in dense mode
        params = SearchParams(top_k=2, cosine_threshold=0.6, use_hybrid=True)
        results = store.search([0.1] * 1536, query_text="q", params=params)
        # Both results must pass — dotproduct 0.3 is not "below" a cosine threshold
        assert len(results) == 2
