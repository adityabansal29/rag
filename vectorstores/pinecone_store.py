from pinecone import Pinecone, ServerlessSpec
from langchain_core.documents import Document

from rag.vectorstores.base import BaseVectorStore


class PineconeVectorStore(BaseVectorStore):
    """
    Pinecone serverless vector store.
    Creates index if it doesn't exist.
    """

    def __init__(
        self,
        api_key: str,
        index_name: str = "rag-pipeline",
        dimension: int = 1536,           # text-embedding-3-small dimension
        cloud: str = "aws",
        region: str = "us-east-1",
        namespace: str = "default",
    ):
        self.pc = Pinecone(api_key=api_key)
        self.namespace = namespace

        # create index if not exists
        existing = [i.name for i in self.pc.list_indexes()]
        if index_name not in existing:
            self.pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud=cloud, region=region),
            )

        self.index = self.pc.Index(index_name)

    def upsert(
        self,
        documents: list[Document],
        vectors: list[list[float]],
    ) -> None:
        records = []
        for doc, vector in zip(documents, vectors):
            records.append({
                "id":     doc.metadata["chunk_id"],
                "values": vector,
                "metadata": {
                    **doc.metadata,
                    "text": doc.page_content,   # store text in metadata for retrieval
                },
            })

        # upsert in batches of 100
        batch_size = 100
        for i in range(0, len(records), batch_size):
            self.index.upsert(
                vectors=records[i: i + batch_size],
                namespace=self.namespace,
            )

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        filter: dict | None = None,
    ) -> list[Document]:
        results = self.index.query(
            vector=query_vector,
            top_k=top_k,
            filter=filter,
            include_metadata=True,
            namespace=self.namespace,
        )

        docs = []
        for match in results["matches"]:
            metadata = dict(match["metadata"])
            text = metadata.pop("text", "")
            docs.append(Document(
                page_content=text,
                metadata=metadata,
            ))
        return docs

    def delete(self, ids: list[str]) -> None:
        self.index.delete(ids=ids, namespace=self.namespace)
