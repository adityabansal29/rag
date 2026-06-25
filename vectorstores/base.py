from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from langchain_core.documents import Document


@dataclass
class SearchParams:
    top_k: int = 5
    cosine_threshold: float = 0.6             # cosine similarity: 0 = unrelated, 1 = identical
    rrf_score_threshold: float = 0.02         # RRF fusion score: 0 = not in either list, ~0.033 = rank-1 in both
    metadata_filters: dict | None = None
    use_hybrid: bool = False


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
        query_text: str | None = None,
        params: SearchParams = None,
    ) -> list[Document]:
        """Search by vector. Returns documents above score_threshold up to top_k."""
        ...

    @abstractmethod
    def delete(self, ids: list[str]) -> None:
        """Delete documents by id."""
        ...
