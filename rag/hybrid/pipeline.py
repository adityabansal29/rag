import asyncio
import logging
import os
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

from langchain_core.documents import Document

from rag.models import Chunk, ChunkType
from rag.parsers.base import BaseParser
from rag.parsers.unstructured_parser import UnstructuredParser
from rag.parsers.docling_parser import DoclingParser
from rag.chunkers.chunker import process_all_parents
from rag.enrichers.enricher import LLMEnricher, build_embedding_content
from rag.embedders.base import BaseEmbedder
from rag.embedders.openai_embedder import OpenAIEmbedder
from rag.vectorstores.base import BaseVectorStore, SearchParams
from rag.vectorstores.chroma_store import ChromaVectorStore
from rag.rerankers.base import BaseReranker


async def generate_answer(llm, query: str, chunks: list[Document]) -> str:
    parts = []
    for i, doc in enumerate(chunks):
        block = f"[{i+1}] {doc.page_content}"
        summary = doc.metadata.get("parent_summary", "")
        if summary:
            block = f"[{i+1}] [Section context: {summary}]\n{doc.page_content}"
        parts.append(block)
    context = "\n\n".join(parts)
    return await llm.call_text(
        system_prompt=(
            "You are a helpful assistant. Answer the question using only the provided context. "
            "Be concise and accurate. If the context is insufficient, say so."
        ),
        content=f"Context:\n{context}\n\nQuestion: {query}",
    )


def chunks_to_langchain_docs(parent_chunks: list[Chunk]) -> list[Document]:
    docs: list[Document] = []

    for parent in parent_chunks:
        parent_summary = parent.retrieved_content or ""
        for child in parent.children:
            if not child.embedding_content:
                continue

            docs.append(Document(
                page_content=child.embedding_content,
                metadata={
                    "chunk_id":        child.id,
                    "parent_id":       child.parent_id,
                    "chunk_type":      child.chunk_type.value,
                    "parent_summary":  parent_summary,
                    **({"raw_content": str(child.raw_content or "")} if child.chunk_type != ChunkType.TEXT else {}),
                    **{k: v for k, v in child.metadata.items()
                       if isinstance(v, (str, int, float, bool))},
                },
            ))

    return docs


class HybridRAGPipeline:
    """
    Full RAG ingestion + hybrid search pipeline.

    Flow:
        parse → chunk (merge/split) → enrich (LLM) → build embedding text
        → LangChain docs → embed → upsert to vector store
    """

    def __init__(
        self,
        embedder: BaseEmbedder | None = None,
        vectorstore: BaseVectorStore | None = None,
        llm_model: str | None = None,
        llm_concurrency: int = 10,
        reranker: BaseReranker | None = None,
    ):
        resolved_model      = llm_model or os.getenv("LLM_MODEL", "gpt-4o")
        self.embedder       = embedder    or OpenAIEmbedder()
        self.vectorstore    = vectorstore or ChromaVectorStore()
        self.reranker       = reranker
        self.enricher       = LLMEnricher(
            model=resolved_model,
            max_concurrency=llm_concurrency,
        )

    def _get_parser(self, parser: str) -> BaseParser:
        if parser == "unstructured":
            return UnstructuredParser()
        if parser == "docling":
            return DoclingParser()
        raise ValueError(f"Unknown parser: {parser}. Choose 'unstructured' or 'docling'.")

    async def run_async(
        self,
        file_path: str,
        parser: str = "unstructured",
        on_step: Callable[[str, str, dict], None] | None = None,
    ) -> list[Chunk]:
        def _step(name: str, metadata: dict) -> None:
            if on_step:
                on_step(name, "done", metadata)

        logger.info("[1/6] Parsing %s with %s", Path(file_path).name, parser)
        raw_parents = self._get_parser(parser).parse(file_path)
        logger.info("      → %d sections found", len(raw_parents))
        type_counts: dict[str, int] = Counter(
            child.chunk_type.value
            for p in raw_parents for child in p.children
        )
        _step("parsing", {"sections_found": len(raw_parents), "element_types": type_counts})

        logger.info("[2/6] Merging/splitting text chunks")
        chunks_before = sum(len(p.children) for p in raw_parents)
        parent_chunks = process_all_parents(raw_parents)
        chunks_after  = sum(len(p.children) for p in parent_chunks)
        logger.info("      → %d child chunks after merge/split", chunks_after)
        _step("chunking", {"chunks_before": chunks_before, "chunks_after": chunks_after})

        logger.info("[3/6] Enriching non-text chunks with LLM")
        await self.enricher.enrich_all(parent_chunks)
        non_text  = sum(1 for p in parent_chunks for c in p.children if c.chunk_type.value != "text")
        summaries = sum(1 for p in parent_chunks if p.retrieved_content)
        logger.info("      → enrichment complete")
        _step("enriching", {
            "total_children": chunks_after,
            "non_text_enriched": non_text,
            "parent_summaries": summaries,
        })

        logger.info("[4/6] Building embedding content")
        build_embedding_content(parent_chunks)

        logger.info("[5/6] Creating LangChain documents")
        docs = chunks_to_langchain_docs(parent_chunks)
        logger.info("      → %d documents created", len(docs))
        _step("embedding", {"documents_created": len(docs)})

        logger.info("[6/6] Embedding and storing")
        texts   = [doc.page_content for doc in docs]
        vectors = self.embedder.embed_documents(texts)
        self.vectorstore.upsert(docs, vectors)
        logger.info("      → %d documents stored", len(docs))
        _step("storing", {"documents_stored": len(docs)})

        return parent_chunks

    async def search_async(
        self,
        query: str,
        params: SearchParams | None = None,
    ) -> tuple[str, list[Document]]:
        params = params or SearchParams()
        mode = "hybrid" if params.use_hybrid else "dense"
        logger.debug("[search] query='%s' mode=%s top_k=%d", query[:80], mode, params.top_k)
        query_vector = self.embedder.embed_query(query)

        # fetch a larger pool when re-ranking so the reranker can make a real selection
        retrieve_params = replace(params, top_k=params.top_k * 3) if self.reranker is not None else params
        results = self.vectorstore.search(
            query_vector=query_vector,
            query_text=query,
            params=retrieve_params,
        )
        logger.debug("[search] → %d chunks returned", len(results))

        if self.reranker is not None and results:
            results = self.reranker.rerank(query, results, top_k=params.top_k)

        return query, results

    async def bm25_search_async(
        self,
        query: str,
        params: SearchParams | None = None,
    ) -> list[Document]:
        params = params or SearchParams()
        logger.debug("[bm25] query='%s' top_k=%d", query[:80], params.top_k)
        results = self.vectorstore.bm25_search(query, params)
        logger.debug("[bm25] → %d chunks returned", len(results))
        return results

    async def dense_search_async(
        self,
        query: str,
        params: SearchParams | None = None,
    ) -> list[Document]:
        params = params or SearchParams()
        logger.debug("[dense] query='%s' top_k=%d", query[:80], params.top_k)
        query_vector = self.embedder.embed_query(query)
        results = self.vectorstore.search(
            query_vector=query_vector,
            query_text=query,
            params=params,
        )
        logger.debug("[dense] → %d chunks returned", len(results))
        return results


