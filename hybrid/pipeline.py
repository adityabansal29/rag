import asyncio
from pathlib import Path

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


def chunks_to_langchain_docs(parent_chunks: list[Chunk]) -> list[Document]:
    docs: list[Document] = []

    for parent in parent_chunks:
        for child in parent.children:
            if not child.embedding_content:
                continue

            docs.append(Document(
                page_content=child.embedding_content,
                metadata={
                    "chunk_id":        child.id,
                    "parent_id":       child.parent_id,
                    "chunk_type":      child.chunk_type.value,
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
        llm_model: str = "gpt-4o",
        llm_concurrency: int = 10,
    ):
        self.embedder    = embedder    or OpenAIEmbedder()
        self.vectorstore = vectorstore or ChromaVectorStore()
        self.llm_model   = llm_model
        self.enricher    = LLMEnricher(
            model=llm_model,
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
    ) -> list[Chunk]:
        print(f"\n{'='*60}")
        print(f"[1/6] Parsing {Path(file_path).name} with {parser}...")
        print(f"{'='*60}\n")
        raw_parents = self._get_parser(parser).parse(file_path)
        print(f"\n      → {len(raw_parents)} sections found")
        BaseParser.print_chunks(raw_parents)

        print(f"\n{'='*60}")
        print(f"[2/6] Merging/splitting text chunks...")
        print(f"{'='*60}\n")
        parent_chunks = process_all_parents(raw_parents)
        total_children = sum(len(p.children) for p in parent_chunks)
        print(f"\n      → {total_children} child chunks after merge/split")

        print(f"\n{'='*60}")
        print(f"[3/6] Enriching non-text chunks with LLM...")
        print(f"{'='*60}\n")
        await self.enricher.enrich_all(parent_chunks)
        print("\n      → enrichment complete")

        print(f"\n{'='*60}")
        print(f"[4/6] Building embedding content...")
        print(f"{'='*60}\n")
        build_embedding_content(parent_chunks)
        print("      → embedding content built")

        print(f"\n{'='*60}")
        print(f"[5/6] Creating LangChain documents...")
        print(f"{'='*60}\n")
        docs = chunks_to_langchain_docs(parent_chunks)
        print(f"      → {len(docs)} documents created")

        print(f"\n{'='*60}")
        print(f"[6/6] Embedding and storing...")
        print(f"{'='*60}\n")
        texts   = [doc.page_content for doc in docs]
        vectors = self.embedder.embed_documents(texts)
        self.vectorstore.upsert(docs, vectors)
        print(f"      → {len(docs)} documents stored\n")

        return parent_chunks

    def run(self, file_path: str, parser: str = "unstructured") -> list[Chunk]:
        return asyncio.run(self.run_async(file_path, parser))

    def search(self, query: str, params: SearchParams = None) -> list[Document]:
        params = params or SearchParams()
        mode = "hybrid" if params.use_hybrid else "dense"
        print(f"\n  [search] query='{query[:80]}'  mode={mode}  top_k={params.top_k}")
        query_vector = self.embedder.embed_query(query)
        results = self.vectorstore.search(
            query_vector=query_vector,
            query_text=query,
            params=params,
        )
        print(f"  [search] → {len(results)} chunks returned")
        return results

    async def generate_answer_async(self, query: str, chunks: list[Document]) -> str:
        print(f"\n  [generate] query='{query[:80]}'  context_chunks={len(chunks)}")
        context = "\n\n".join(f"[{i+1}] {doc.page_content}" for i, doc in enumerate(chunks))
        answer = await self.enricher.llm.call_text(
            system_prompt=(
                "You are a helpful assistant. Answer the question using only the provided context. "
                "Be concise and accurate. If the context is insufficient, say so."
            ),
            content=f"Context:\n{context}\n\nQuestion: {query}",
        )
        print(f"  [generate] → {answer[:120]!r}")
        return answer

    def generate_answer(self, query: str, chunks: list[Document]) -> str:
        return asyncio.run(self.generate_answer_async(query, chunks))
