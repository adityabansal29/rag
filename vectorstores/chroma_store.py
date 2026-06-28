import chromadb
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from rag.vectorstores.base import BaseVectorStore, SearchParams
from rag.utils.rrf import rrf_fuse


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
        params: SearchParams | None = None,
    ) -> list[Document]:
        params = params or SearchParams()

        if params.use_hybrid and query_text:
            return self._hybrid_search(query_vector, query_text, params)

        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=params.top_k,
            where=params.metadata_filters,
        )
        print(f"\n  Chroma Dense Results  threshold={params.cosine_threshold}")
        print(f"  {'#':<5} {'Chunk ID':<40} {'Score':>8} {'Status':>10}")
        print(f"  {'-'*5} {'-'*40} {'-'*8} {'-'*10}")
        docs = []
        for i, text in enumerate(results["documents"][0]):
            similarity = 1 - results["distances"][0][i]
            chunk_id = results["metadatas"][0][i].get("chunk_id", "?")
            kept = params.cosine_threshold is None or similarity >= params.cosine_threshold
            status = "kept" if kept else "filtered"
            print(f"  {i:<5} {str(chunk_id):<40} {similarity:>8.4f} {status:>10}")
            if not kept:
                continue
            docs.append(Document(
                page_content=text,
                metadata={**results["metadatas"][0][i], "score": round(similarity, 4)},
            ))
        print()
        return docs

    def bm25_search(
        self,
        query_text: str,
        params: SearchParams | None = None,
    ) -> list[Document]:
        params = params or SearchParams()
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
        bm25 = BM25Retriever.from_documents(langchain_docs, k=params.top_k)
        results = bm25.invoke(query_text)

        print(f"\n  BM25 Rankings  corpus={len(all_docs['ids'])}")
        print(f"  {'#':<5} {'Chunk ID':<40} {'RRF Score':>10}")
        print(f"  {'-'*5} {'-'*40} {'-'*10}")
        for rank, doc in enumerate(results):
            print(f"  {rank:<5} {str(doc.metadata.get('chunk_id','?')):<40} {1/(60+rank):>10.6f}")
        print()

        return results

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
        print(f"\n  [chroma hybrid] corpus={len(all_docs['ids'])} docs  top_k={params.top_k}  rrf_threshold={params.rrf_score_threshold}")

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

        print(f"\n  {'BM25 Rankings':}")
        print(f"  {'#':<5} {'Chunk ID':<40} {'RRF Score':>10}")
        print(f"  {'-'*5} {'-'*40} {'-'*10}")
        for id_, rank in sorted(bm25_rank.items(), key=lambda x: x[1]):
            rrf = 1 / (60 + rank)
            print(f"  {rank:<5} {id_:<40} {rrf:>10.6f}")

        # dense search
        n = min(params.top_k * 2, len(all_docs["ids"]))
        dense_results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=n,
            where=params.metadata_filters,
        )
        dense_ids = dense_results["ids"][0]
        dense_rank = {id_: rank for rank, id_ in enumerate(dense_ids)}

        print(f"\n  {'Vector Rankings':}")
        print(f"  {'#':<5} {'Chunk ID':<40} {'RRF Score':>10}")
        print(f"  {'-'*5} {'-'*40} {'-'*10}")
        for id_, rank in sorted(dense_rank.items(), key=lambda x: x[1]):
            rrf = 1 / (60 + rank)
            print(f"  {rank:<5} {id_:<40} {rrf:>10.6f}")

        # RRF merge
        dense_ids_ranked = [id_ for id_, _ in sorted(dense_rank.items(), key=lambda x: x[1])]
        bm25_ids_ranked  = [id_ for id_, _ in sorted(bm25_rank.items(),  key=lambda x: x[1])]
        rrf_scores = rrf_fuse([dense_ids_ranked, bm25_ids_ranked])

        top_ids = sorted(rrf_scores, key=lambda id_: rrf_scores[id_], reverse=True)[:params.top_k]

        print(f"\n  {'RRF Merged (top {params.top_k})':}")
        print(f"  {'Chunk ID':<40} {'RRF Score':>10} {'Status':>10}")
        print(f"  {'-'*40} {'-'*10} {'-'*10}")
        for id_ in top_ids:
            status = "kept" if (params.rrf_score_threshold is None or rrf_scores[id_] >= params.rrf_score_threshold) else "filtered"
            print(f"  {id_:<40} {rrf_scores[id_]:>10.6f} {status:>10}")
        print()

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
