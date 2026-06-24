from abc import ABC, abstractmethod


class BaseEmbedder(ABC):
    """
    Pluggable embedding interface.
    Implement this to swap between OpenAI, HuggingFace, Cohere, Voyage etc.
    """

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents. Returns list of vectors."""
        ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string. Returns a vector."""
        ...
