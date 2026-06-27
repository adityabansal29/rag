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
        params: SearchParams = None,
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

        # fetch more per sub-query; disable thresholds so RRF decides quality
        sub_params = replace(
            params,
            top_k=params.top_k * len(all_queries),
            cosine_threshold=None,
            rrf_score_threshold=None,
        )

        results_per_query: list[list[Document]] = await asyncio.gather(*[
            self.pipeline.search_async(q, sub_params)
            for q in all_queries
        ])

        # build doc lookup and ranked id lists
        doc_lookup: dict[str, Document] = {}
        rank_lists: list[list[str]] = []

        for q, docs in zip(all_queries, results_per_query):
            ids = []
            for doc in docs:
                chunk_id = doc.metadata["chunk_id"]
                doc_lookup[chunk_id] = doc
                ids.append(chunk_id)
            rank_lists.append(ids)
            print(f"  [fusion] '{q[:60]}' → {len(docs)} chunks")

        # RRF across all query result lists
        rrf_scores = rrf_fuse(rank_lists)
        top_ids = sorted(rrf_scores, key=lambda id_: rrf_scores[id_], reverse=True)[:params.top_k]

        print(f"\n  [fusion] RRF merged → top {params.top_k} of {len(rrf_scores)} unique chunks")
        print(f"  {'Chunk ID':<40} {'RRF Score':>10}")
        print(f"  {'-'*40} {'-'*10}")
        for id_ in top_ids:
            print(f"  {id_:<40} {rrf_scores[id_]:>10.6f}")
        print()

        return [
            Document(
                page_content=doc_lookup[id_].page_content,
                metadata={**doc_lookup[id_].metadata, "rrf_score": round(rrf_scores[id_], 6)},
            )
            for id_ in top_ids
            if id_ in doc_lookup
        ]

    def search(
        self,
        query: str,
        params: SearchParams = None,
        history: ConversationHistory | None = None,
    ) -> list[Document]:
        return asyncio.run(self.search_async(query, params, history))
