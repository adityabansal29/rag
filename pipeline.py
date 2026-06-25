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
    """
    Convert child chunks to LangChain Documents.
    Uses embedding_content as page_content.
    Attaches full metadata including parent info.
    """
    docs: list[Document] = []

    for parent in parent_chunks:
        for child in parent.children:
            if not child.embedding_content:
                continue

            docs.append(Document(
                page_content=child.embedding_content,
                metadata={
                    # identifiers
                    "chunk_id":        child.id,
                    "parent_id":       child.parent_id,

                    # content
                    "chunk_type":      child.chunk_type.value,
                    **({"raw_content": str(child.raw_content or "")} if child.chunk_type != ChunkType.TEXT else {}),

                    # source metadata
                    **{k: v for k, v in child.metadata.items()
                       if isinstance(v, (str, int, float, bool))},  # chroma/pinecone safe types only
                },
            ))

    return docs


class RAGPipeline:
    """
    Full RAG ingestion pipeline.

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
        """
        Full async pipeline run.
        Returns parent chunks with all fields populated.
        """
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

    def run(
        self,
        file_path: str,
        parser: str = "unstructured",
    ) -> list[Chunk]:
        """Sync wrapper around run_async."""
        return asyncio.run(self.run_async(file_path, parser))

    def search(
        self,
        query: str,
        params: SearchParams = None,
    ) -> list[Document]:
        """Search the vector store with a text query."""
        query_vector = self.embedder.embed_query(query)
        return self.vectorstore.search(
            query_vector=query_vector,
            query_text=query,
            params=params or SearchParams(),
        )
