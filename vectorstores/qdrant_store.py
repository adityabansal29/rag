import uuid

from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance,
    PointStruct, Filter, FieldCondition, MatchValue,
    SparseVector, SparseVectorParams, SparseIndexParams,
    Prefetch, FusionQuery, Fusion,
)
from langchain_core.documents import Document

from rag.vectorstores.base import BaseVectorStore, SearchParams


class QdrantVectorStore(BaseVectorStore):
    """
    Qdrant vector store.
    Hybrid search: native sparse (fastembed BM25) + dense with built-in RRF fusion.
    Note: enable_hybrid=True creates a different collection schema — use a new collection name.
    """

    DENSE_VEC  = "dense"
    SPARSE_VEC = "sparse"

    def __init__(
        self,
        collection_name: str = "rag",
        dimension: int = 1536,
        host: str | None = None,
        port: int = 6333,
        path: str | None = None,
        in_memory: bool = False,
        enable_hybrid: bool = False,
    ):
        if in_memory:
            self.client = QdrantClient(":memory:")
        elif path:
            self.client = QdrantClient(path=path)
        elif host:
            self.client = QdrantClient(host=host, port=port)
        else:
            self.client = QdrantClient(":memory:")

        self.collection_name = collection_name
        self.dimension = dimension
        self.enable_hybrid = enable_hybrid

        existing = [c.name for c in self.client.get_collections().collections]
        if collection_name not in existing:
            if enable_hybrid:
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config={
                        self.DENSE_VEC: VectorParams(size=dimension, distance=Distance.COSINE),
                    },
                    sparse_vectors_config={
                        self.SPARSE_VEC: SparseVectorParams(
                            index=SparseIndexParams(on_disk=False)
                        )
                    },
                )
            else:
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
                )

        if enable_hybrid:
            self._sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

    def upsert(
        self,
        documents: list[Document],
        vectors: list[list[float]],
    ) -> None:
        texts = [doc.page_content for doc in documents]

        sparse_embeddings = None
        if self.enable_hybrid:
            sparse_embeddings = list(self._sparse_model.embed(texts))

        points = []
        for i, (doc, dense_vec) in enumerate(zip(documents, vectors)):
            if self.enable_hybrid:
                sp = sparse_embeddings[i]
                vector = {
                    self.DENSE_VEC: dense_vec,
                    self.SPARSE_VEC: SparseVector(
                        indices=sp.indices.tolist(),
                        values=sp.values.tolist(),
                    ),
                }
            else:
                vector = dense_vec

            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={**doc.metadata, "text": doc.page_content},
            ))

        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(
        self,
        query_vector: list[float],
        query_text: str | None = None,
        params: SearchParams = None,
    ) -> list[Document]:
        params = params or SearchParams()
        qdrant_filter = self._build_filter(params.metadata_filters)

        if self.enable_hybrid and params.use_hybrid and query_text:
            return self._hybrid_search(query_vector, query_text, params, qdrant_filter)

        if self.enable_hybrid:
            # dense-only search on hybrid collection
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=(self.DENSE_VEC, query_vector),
                limit=params.top_k,
                query_filter=qdrant_filter,
                score_threshold=params.cosine_threshold,
            )
        else:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=params.top_k,
                query_filter=qdrant_filter,
                score_threshold=params.cosine_threshold,
            )

        return self._hits_to_docs(results)

    def _hybrid_search(
        self,
        query_vector: list[float],
        query_text: str,
        params: SearchParams,
        qdrant_filter,
    ) -> list[Document]:
        sparse_query = list(self._sparse_model.query_embed(query_text))[0]

        results = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                Prefetch(
                    query=query_vector,
                    using=self.DENSE_VEC,
                    limit=params.top_k * 2,
                    filter=qdrant_filter,
                ),
                Prefetch(
                    query=SparseVector(
                        indices=sparse_query.indices.tolist(),
                        values=sparse_query.values.tolist(),
                    ),
                    using=self.SPARSE_VEC,
                    limit=params.top_k * 2,
                    filter=qdrant_filter,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=params.top_k,
            score_threshold=params.rrf_score_threshold,
            with_payload=True,
        )

        return self._hits_to_docs(results.points)

    def _hits_to_docs(self, hits) -> list[Document]:
        docs = []
        for hit in hits:
            payload = dict(hit.payload)
            text = payload.pop("text", "")
            docs.append(Document(
                page_content=text,
                metadata={**payload, "score": round(hit.score, 4)},
            ))
        return docs

    @staticmethod
    def _build_filter(metadata_filters: dict | None) -> Filter | None:
        if not metadata_filters:
            return None
        return Filter(must=[
            FieldCondition(key=k, match=MatchValue(value=v))
            for k, v in metadata_filters.items()
        ])

    def delete(self, ids: list[str]) -> None:
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(must=[
                FieldCondition(key="chunk_id", match=MatchValue(value=id_))
                for id_ in ids
            ]),
        )
