from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance,
    PointStruct, Filter, FieldCondition, MatchValue
)
from langchain_core.documents import Document
import uuid

from rag.vectorstores.base import BaseVectorStore


class QdrantVectorStore(BaseVectorStore):
    """
    Qdrant vector store.
    Supports local in-memory, local persistent, or remote Qdrant server.
    """

    def __init__(
        self,
        collection_name: str = "rag",
        dimension: int = 1536,
        host: str | None = None,
        port: int = 6333,
        path: str | None = None,          # local persistent path
        in_memory: bool = False,
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

        # create collection if not exists
        existing = [c.name for c in self.client.get_collections().collections]
        if collection_name not in existing:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=dimension,
                    distance=Distance.COSINE,
                ),
            )

    def upsert(
        self,
        documents: list[Document],
        vectors: list[list[float]],
    ) -> None:
        points = []
        for doc, vector in zip(documents, vectors):
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    **doc.metadata,
                    "text": doc.page_content,
                },
            ))

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        filter: dict | None = None,
    ) -> list[Document]:
        qdrant_filter = None
        if filter:
            # convert simple key=value filter to Qdrant filter
            conditions = [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filter.items()
            ]
            qdrant_filter = Filter(must=conditions)

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            query_filter=qdrant_filter,
        )

        docs = []
        for hit in results:
            payload = dict(hit.payload)
            text = payload.pop("text", "")
            docs.append(Document(
                page_content=text,
                metadata=payload,
            ))
        return docs

    def delete(self, ids: list[str]) -> None:
        from qdrant_client.models import PointIdsList
        # Qdrant delete by payload filter on chunk_id
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="chunk_id",
                        match=MatchValue(value=id_)
                    )
                    for id_ in ids
                ]
            ),
        )
