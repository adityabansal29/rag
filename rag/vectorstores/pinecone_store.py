from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder
from langchain_core.documents import Document

from rag.vectorstores.base import BaseVectorStore, SearchParams


class PineconeVectorStore(BaseVectorStore):
    """
    Pinecone serverless vector store.
    Hybrid search: native sparse-dense with alpha weighting via pinecone-text BM25Encoder.
    Note: enable_hybrid=True requires metric="dotproduct" — create a new index if switching.
    """

    def __init__(
        self,
        api_key: str,
        index_name: str = "rag-pipeline",
        dimension: int = 1536,
        cloud: str = "aws",
        region: str = "us-east-1",
        namespace: str = "rag",
        enable_hybrid: bool = False,
    ):
        self.pc = Pinecone(api_key=api_key)
        self.namespace = namespace
        self.enable_hybrid = enable_hybrid

        metric = "dotproduct" if enable_hybrid else "cosine"

        existing = [i.name for i in self.pc.list_indexes()]
        if index_name not in existing:
            self.pc.create_index(
                name=index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(cloud=cloud, region=region),
            )

        self.index = self.pc.Index(index_name)

        if enable_hybrid:
            self._sparse_encoder = BM25Encoder().default()

    def upsert(
        self,
        documents: list[Document],
        vectors: list[list[float]],
    ) -> None:
        texts = [doc.page_content for doc in documents]

        sparse_vecs = None
        if self.enable_hybrid:
            sparse_vecs = self._sparse_encoder.encode_documents(texts)

        records = []
        for i, (doc, vector) in enumerate(zip(documents, vectors)):
            record = {
                "id":     doc.metadata["chunk_id"],
                "values": vector,
                "metadata": {
                    **doc.metadata,
                    "text": doc.page_content,
                },
            }
            if sparse_vecs:
                record["sparse_values"] = sparse_vecs[i]
            records.append(record)

        batch_size = 100
        for i in range(0, len(records), batch_size):
            self.index.upsert(
                vectors=records[i: i + batch_size],
                namespace=self.namespace,
            )

    def search(
        self,
        query_vector: list[float],
        query_text: str | None = None,
        params: SearchParams | None = None,
        alpha: float = 0.75,
    ) -> list[Document]:
        params = params or SearchParams()

        dense_vec = query_vector
        sparse_vector = None
        if self.enable_hybrid and params.use_hybrid and query_text:
            raw_sparse = self._sparse_encoder.encode_queries(query_text)
            dense_vec = [v * alpha for v in query_vector]
            sparse_vector = {
                "indices": raw_sparse["indices"],
                "values":  [v * (1 - alpha) for v in raw_sparse["values"]],
            }

        query_kwargs = dict(
            vector=dense_vec,
            top_k=params.top_k,
            filter=params.metadata_filters,
            include_metadata=True,
            namespace=self.namespace,
        )
        if sparse_vector:
            query_kwargs["sparse_vector"] = sparse_vector

        results = self.index.query(**query_kwargs)

        is_hybrid = sparse_vector is not None
        label = f"Pinecone {'Hybrid' if is_hybrid else 'Dense'} Results (alpha={alpha if is_hybrid else 'n/a'})"
        print(f"\n  {label}")
        print(f"  {'#':<5} {'Chunk ID':<40} {'Score':>10} {'Status':>10}")
        print(f"  {'-'*5} {'-'*40} {'-'*10} {'-'*10}")
        for i, match in enumerate(results["matches"]):
            status = "kept" if (params.cosine_threshold is None or match["score"] >= params.cosine_threshold) else "filtered"
            print(f"  {i:<5} {match['id']:<40} {match['score']:>10.6f} {status:>10}")
        print()

        docs = []
        for match in results["matches"]:
            if params.cosine_threshold is not None and match["score"] < params.cosine_threshold:
                continue
            metadata = dict(match["metadata"])
            text = metadata.pop("text", "")
            docs.append(Document(
                page_content=text,
                metadata={**metadata, "score": round(match["score"], 4)},
            ))
        return docs

    def bm25_search(
        self,
        query_text: str,
        params: SearchParams | None = None,
    ) -> list[Document]:
        if not self.enable_hybrid:
            raise NotImplementedError("PineconeVectorStore requires enable_hybrid=True for BM25 search.")
        params = params or SearchParams()
        raw_sparse = self._sparse_encoder.encode_queries(query_text)
        sparse_vector = {
            "indices": raw_sparse["indices"],
            "values":  raw_sparse["values"],
        }
        zero_dense = [0.0] * self.index.describe_index_stats()["dimension"]

        results = self.index.query(
            vector=zero_dense,
            sparse_vector=sparse_vector,
            top_k=params.top_k,
            filter=params.metadata_filters,
            include_metadata=True,
            namespace=self.namespace,
        )

        print(f"\n  Pinecone BM25 Results")
        print(f"  {'#':<5} {'Chunk ID':<40} {'Score':>10}")
        print(f"  {'-'*5} {'-'*40} {'-'*10}")
        for i, match in enumerate(results["matches"]):
            print(f"  {i:<5} {match['id']:<40} {match['score']:>10.6f}")
        print()

        docs = []
        for match in results["matches"]:
            metadata = dict(match["metadata"])
            text = metadata.pop("text", "")
            docs.append(Document(
                page_content=text,
                metadata={**metadata, "score": round(match["score"], 4)},
            ))
        return docs

    def delete(self, ids: list[str]) -> None:
        self.index.delete(ids=ids, namespace=self.namespace)
