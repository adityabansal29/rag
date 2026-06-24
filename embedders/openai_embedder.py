from openai import OpenAI

from rag.embedders.base import BaseEmbedder


class OpenAIEmbedder(BaseEmbedder):
    """
    OpenAI text-embedding-3-small (default) or text-embedding-3-large.
    Swap model_name to use a different OpenAI embedding model.
    """

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        batch_size: int = 100,
    ):
        self.client = OpenAI()
        self.model_name = model_name
        self.batch_size = batch_size

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed in batches to avoid API limits."""
        all_vectors: list[list[float]] = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i: i + self.batch_size]
            response = self.client.embeddings.create(
                input=batch,
                model=self.model_name,
            )
            # response.data is sorted by index
            vectors = [item.embedding for item in response.data]
            all_vectors.extend(vectors)

        return all_vectors

    def embed_query(self, text: str) -> list[float]:
        response = self.client.embeddings.create(
            input=[text],
            model=self.model_name,
        )
        return response.data[0].embedding
