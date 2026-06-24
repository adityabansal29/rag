from abc import ABC, abstractmethod

from langchain_core.documents import Document


class BaseVectorStore(ABC):
    """
    Pluggable vector store interface.
    Implement to swap between Chroma, Pinecone, Qdrant etc.
    """

    @abstractmethod
    def upsert(
        self,
        documents: list[Document],
        vectors: list[list[float]],
    ) -> None:
        """Store documents with their precomputed vectors."""
        ...

    @abstractmethod
    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        filter: dict | None = None,
    ) -> list[Document]:
        """Search by vector. Returns top_k matching documents."""
        ...

    @abstractmethod
    def delete(self, ids: list[str]) -> None:
        """Delete documents by id."""
        ...
