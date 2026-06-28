import asyncio
from dataclasses import replace

from langchain_core.documents import Document

from rag.hybrid.pipeline import HybridRAGPipeline
from rag.vectorstores.base import SearchParams
from rag.fusion.query_rewriter import QueryRewriter
from rag.utils.rrf import rrf_fuse
from rag.conversation.history import ConversationHistory


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
        history: ConversationHistory | None = None,
    ) -> list[Document]:
        params = params or SearchParams()
        # Single LLM call: contextualize (if history) + rewrite variants together
        standalone, variants = await self.rewriter.rewrite(query, self.n_queries, history)
        if standalone != query:
            print(f"\n  [contextualize] '{query[:60]}' → '{standalone[:60]}'")
        all_queries = [standalone] + variants

        print(f"\n  [fusion] rewritten queries ({len(all_queries)} total):")
        for i, q in enumerate(all_queries):
            label = "original" if i == 0 else f"variant {i}"
            print(f"    [{label}] {q}")

        inflated_top_k = params.top_k * len(all_queries)

        if params.use_hybrid:
            # BM25 once on standalone + dense on every variant — merged in outer RRF
            bm25_params   = replace(params, top_k=inflated_top_k, rrf_score_threshold=None, use_hybrid=False)
            dense_params  = replace(params, top_k=inflated_top_k, cosine_threshold=None, rrf_score_threshold=None, use_hybrid=False)

            bm25_results, *dense_results = await asyncio.gather(
                self.pipeline.bm25_search_async(standalone, bm25_params),
                *[self.pipeline.dense_search_async(q, dense_params) for q in all_queries],
            )
            labeled = [("bm25:standalone", bm25_results)] + [
                (f"dense:{q[:40]}", docs) for q, docs in zip(all_queries, dense_results)
            ]
        else:
            dense_params = replace(params, top_k=inflated_top_k, cosine_threshold=None, rrf_score_threshold=None)
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
            print(f"  [fusion] '{label}' → {len(docs)} chunks")

        # RRF across all query result lists
        rrf_scores = rrf_fuse(rank_lists)
        # when re-ranking, pass all RRF candidates so the reranker does the final slicing
        candidate_count = len(rrf_scores) if self.pipeline.reranker is not None else params.top_k
        top_ids = sorted(rrf_scores, key=lambda id_: rrf_scores[id_], reverse=True)[:candidate_count]

        print(f"\n  [fusion] RRF merged → top {candidate_count} of {len(rrf_scores)} unique chunks")
        print(f"  {'Chunk ID':<40} {'RRF Score':>10}")
        print(f"  {'-'*40} {'-'*10}")
        for id_ in top_ids:
            print(f"  {id_:<40} {rrf_scores[id_]:>10.6f}")
        print()

        final_docs = [
            Document(
                page_content=doc_lookup[id_].page_content,
                metadata={**doc_lookup[id_].metadata, "rrf_score": round(rrf_scores[id_], 6)},
            )
            for id_ in top_ids
            if id_ in doc_lookup
        ]

        if self.pipeline.reranker is not None and final_docs:
            final_docs = self.pipeline.reranker.rerank(standalone, final_docs, top_k=params.top_k)

        return final_docs

    def search(
        self,
        query: str,
        params: SearchParams | None = None,
        history: ConversationHistory | None = None,
    ) -> list[Document]:
        return asyncio.run(self.search_async(query, params, history))
