import chromadb
from langchain_core.documents import Document

from rag.vectorstores.base import BaseVectorStore


class ChromaVectorStore(BaseVectorStore):
    """
    Chroma vector store.
    Uses persistent local storage by default.
    Pass host/port for remote Chroma server.
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
        ids        = [doc.metadata["chunk_id"] for doc in documents]
        texts      = [doc.page_content for doc in documents]
        metadatas  = [doc.metadata for doc in documents]

        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=vectors,
            metadatas=metadatas,
        )

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        filter: dict | None = None,
    ) -> list[Document]:
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where=filter,
        )

        docs = []
        for i, text in enumerate(results["documents"][0]):
            docs.append(Document(
                page_content=text,
                metadata=results["metadatas"][0][i],
            ))
        return docs

    def delete(self, ids: list[str]) -> None:
        self.collection.delete(ids=ids)
