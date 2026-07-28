import asyncio
import logging
from dataclasses import replace

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

from rag.hybrid.pipeline import HybridRAGPipeline
from rag.vectorstores.base import SearchParams
from rag.fusion.query_rewriter import QueryRewriter
from rag.utils.rrf import rrf_fuse


def _fusion_search_params(params: SearchParams, n_queries: int, use_hybrid: bool) -> tuple[SearchParams, SearchParams | None]:
    """Return (dense_params, bm25_params) tuned for fusion retrieval.

    Fusion inflates top_k so the RRF merge has enough candidates across N queries.
    Score thresholds are cleared here — fusion's outer RRF does its own ranking,
    so per-query filtering would silently drop valid candidates before the merge.
    """
    inflated = params.top_k * n_queries
    dense = replace(params, top_k=inflated, cosine_threshold=None, rrf_score_threshold=None, use_hybrid=False)
    if use_hybrid:
        bm25 = replace(params, top_k=inflated, rrf_score_threshold=None, use_hybrid=False)
        return dense, bm25
    return dense, None


class FusionRAGPipeline:
    """
    Fusion RAG: rewrites the query into n variations, searches in parallel,
    then merges all results with RRF to produce the final ranked chunks.

    Flow:
        query
          → LLM rewrites into n queries (+ original = n+1 total)
          → parallel search per query (higher top_k, no score filtering)
          → RRF fusion across all result lists
          → top_k final chunks
          → (optional) generate answer + evaluate
    """

    def __init__(
        self,
        pipeline: HybridRAGPipeline,
        n_queries: int = 3,
    ):
        self.pipeline  = pipeline
        self.n_queries = n_queries
        self.rewriter  = QueryRewriter(pipeline.enricher.llm)

    async def search_async(
        self,
        query: str,
        params: SearchParams | None = None,
    ) -> tuple[str, list[Document]]:
        params = params or SearchParams()
        standalone, variants = await self.rewriter.rewrite(query, self.n_queries)
        fusion_degraded = not variants  # rewriter returned no variants (LLM failure or empty response)
        if standalone != query:
            logger.debug("[contextualize] '%s' → '%s'", query[:60], standalone[:60])
        all_queries = [standalone] + variants
        if fusion_degraded:
            logger.warning("[fusion] no variants returned — running as single-query dense search")

        logger.debug("[fusion] rewritten queries (%d total): %s", len(all_queries), all_queries)

        dense_params, bm25_params = _fusion_search_params(params, len(all_queries), params.use_hybrid)

        if bm25_params is not None:
            # BM25 once on standalone + dense on every variant — merged in outer RRF
            bm25_results, *dense_results = await asyncio.gather(
                self.pipeline.bm25_search_async(standalone, bm25_params),
                *[self.pipeline.dense_search_async(q, dense_params) for q in all_queries],
            )
            labeled = [("bm25:standalone", bm25_results)] + [
                (f"dense:{q[:40]}", docs) for q, docs in zip(all_queries, dense_results)
            ]
        else:
            dense_results = await asyncio.gather(*[
                self.pipeline.dense_search_async(q, dense_params) for q in all_queries
            ])
            labeled = [(f"dense:{q[:40]}", docs) for q, docs in zip(all_queries, dense_results)]

        # build doc lookup and ranked id lists
        doc_lookup: dict[str, Document] = {}
        rank_lists: list[list[str]] = []

        for label, docs in labeled:
            ids = []
            for doc in docs:
                chunk_id = doc.metadata["chunk_id"]
                doc_lookup[chunk_id] = doc
                ids.append(chunk_id)
            rank_lists.append(ids)
            logger.debug("[fusion] '%s' → %d chunks", label, len(docs))

        # RRF across all query result lists
        rrf_scores = rrf_fuse(rank_lists)
        # when re-ranking, pass a larger pool but cap it to avoid overwhelming the reranker
        candidate_count = min(len(rrf_scores), params.top_k * 5) if self.pipeline.reranker is not None else params.top_k
        top_ids = sorted(rrf_scores, key=lambda id_: rrf_scores[id_], reverse=True)[:candidate_count]

        logger.info("[fusion] RRF merged → top %d of %d unique chunks", candidate_count, len(rrf_scores))
        if logger.isEnabledFor(logging.DEBUG):
            rows = "\n".join(f"  {id_:<40} {rrf_scores[id_]:>10.6f}" for id_ in top_ids)
            logger.debug("[fusion] RRF scores:\n%s", rows)

        extra = {"fusion_degraded": True} if fusion_degraded else {}
        final_docs = [
            Document(
                page_content=doc_lookup[id_].page_content,
                metadata={**doc_lookup[id_].metadata, "rrf_score": round(rrf_scores[id_], 6), **extra},
            )
            for id_ in top_ids
        ]

        if self.pipeline.reranker is not None and final_docs:
            final_docs = self.pipeline.reranker.rerank(standalone, final_docs, top_k=params.top_k)

        return standalone, final_docs

    def search(
        self,
        query: str,
        params: SearchParams | None = None,
    ) -> tuple[str, list[Document]]:
        return asyncio.run(self.search_async(query, params))

    async def generate_answer_async(self, query: str, chunks: list[Document]) -> str:
        return await self.pipeline.generate_answer_async(query, chunks)

    def generate_answer(self, query: str, chunks: list[Document]) -> str:
        return self.pipeline.generate_answer(query, chunks)
