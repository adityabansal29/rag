import chromadb
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from rag.vectorstores.base import BaseVectorStore, SearchParams


class ChromaVectorStore(BaseVectorStore):
    """
    Chroma vector store.
    Uses persistent local storage by default.
    Pass host/port for remote Chroma server.
    Hybrid search: BM25Retriever (langchain-community) + DIY RRF over all stored docs.
    """

    def __init__(
        self,
        collection_name: str = "rag",
        persist_directory: str = "./chroma_db",
        host: str | None = None,
        port: int | None = None,
    ):
        if host and port:
            self.client = chromadb.HttpClient(host=host, port=port)
        else:
            self.client = chromadb.PersistentClient(path=persist_directory)

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(
        self,
        documents: list[Document],
        vectors: list[list[float]],
    ) -> None:
        ids       = [doc.metadata["chunk_id"] for doc in documents]
        texts     = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=vectors,
            metadatas=metadatas,
        )

    def search(
        self,
        query_vector: list[float],
        query_text: str | None = None,
        params: SearchParams = None,
    ) -> list[Document]:
        params = params or SearchParams()

        if params.use_hybrid and query_text:
            return self._hybrid_search(query_vector, query_text, params)

        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=params.top_k,
            where=params.metadata_filters,
        )
        docs = []
        for i, text in enumerate(results["documents"][0]):
            similarity = 1 - results["distances"][0][i]
            if params.cosine_threshold is not None and similarity < params.cosine_threshold:
                continue
            docs.append(Document(
                page_content=text,
                metadata={**results["metadatas"][0][i], "score": round(similarity, 4)},
            ))
        return docs

    def _hybrid_search(
        self,
        query_vector: list[float],
        query_text: str,
        params: SearchParams,
    ) -> list[Document]:
        # fetch all stored docs for BM25 corpus
        all_docs = self.collection.get(
            where=params.metadata_filters,
            include=["documents", "metadatas"],
        )
        if not all_docs["ids"]:
            return []

        langchain_docs = [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(all_docs["documents"], all_docs["metadatas"])
        ]
        doc_lookup = {
            meta["chunk_id"]: (text, meta)
            for text, meta in zip(all_docs["documents"], all_docs["metadatas"])
        }

        # BM25 search
        bm25 = BM25Retriever.from_documents(langchain_docs, k=params.top_k * 2)
        bm25_results = bm25.invoke(query_text)
        bm25_rank = {
            doc.metadata["chunk_id"]: rank
            for rank, doc in enumerate(bm25_results)
        }

        # dense search
        n = min(params.top_k * 2, len(all_docs["ids"]))
        dense_results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=n,
            where=params.metadata_filters,
        )
        dense_ids = dense_results["ids"][0]
        dense_distances = dense_results["distances"][0]
        dense_rank = {id_: rank for rank, id_ in enumerate(dense_ids)}
        dense_similarity = {id_: 1 - dist for id_, dist in zip(dense_ids, dense_distances)}

        # RRF merge (k=60 is standard constant)
        k = 60
        all_ids = set(dense_rank) | set(bm25_rank)
        rrf_scores = {
            id_: (1 / (k + dense_rank[id_]) if id_ in dense_rank else 0)
                + (1 / (k + bm25_rank[id_]) if id_ in bm25_rank else 0)
            for id_ in all_ids
        }

        top_ids = sorted(rrf_scores, key=lambda id_: rrf_scores[id_], reverse=True)[:params.top_k]

        docs = []
        for id_ in top_ids:
            if id_ not in doc_lookup:
                continue
            text, metadata = doc_lookup[id_]
            if params.rrf_score_threshold is not None and rrf_scores[id_] < params.rrf_score_threshold:
                continue
            docs.append(Document(
                page_content=text,
                metadata={**metadata, "score": round(rrf_scores[id_], 6)},
            ))
        return docs

    def delete(self, ids: list[str]) -> None:
        self.collection.delete(ids=ids)
