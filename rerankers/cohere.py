from langchain_core.documents import Document

from rag.rerankers.base import BaseReranker


class CohereReranker(BaseReranker):
    """
    Cohere API re-ranker using the rerank endpoint.
    Requires a COHERE_API_KEY environment variable or explicit api_key.

    Default model: rerank-english-v3.0 (Cohere's best for English retrieval).
    """

    def __init__(self, api_key: str | None = None, model: str = "rerank-english-v3.0"):
        try:
            import cohere
        except ImportError as e:
            raise ImportError(
                "cohere is required for CohereReranker. "
                "Install it with: uv add cohere"
            ) from e
        import os
        self.client = cohere.Client(api_key or os.environ["COHERE_API_KEY"])
        self.model = model

    def rerank(self, query: str, documents: list[Document], top_k: int) -> list[Document]:
        if not documents:
            return []

        response = self.client.rerank(
            model=self.model,
            query=query,
            documents=[doc.page_content for doc in documents],
            top_n=top_k,
        )

        print(f"\n  [rerank] cohere '{self.model}' scored {len(documents)} candidates → top {top_k}")
        print(f"  {'#':<5} {'Score':>10} {'Chunk ID':<40}")
        print(f"  {'-'*5} {'-'*10} {'-'*40}")

        reranked = []
        for i, result in enumerate(response.results):
            doc = documents[result.index]
            score = result.relevance_score
            print(f"  {i:<5} {score:>10.4f} {doc.metadata.get('chunk_id', ''):<40}")
            reranked.append(Document(
                page_content=doc.page_content,
                metadata={**doc.metadata, "rerank_score": round(score, 6)},
            ))
        print()

        return reranked
