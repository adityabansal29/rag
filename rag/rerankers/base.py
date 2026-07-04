from abc import ABC, abstractmethod

from langchain_core.documents import Document


class BaseReranker(ABC):
    @abstractmethod
    def rerank(self, query: str, documents: list[Document], top_k: int) -> list[Document]:
        """Re-score and return the top_k most relevant documents for query."""
        ...
