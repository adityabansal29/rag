from langchain_core.documents import Document

from rag.rerankers.base import BaseReranker


class CrossEncoderReranker(BaseReranker):
    """
    Local cross-encoder re-ranker using sentence-transformers.
    Scores each (query, document) pair jointly — captures richer relevance signals
    than bi-encoder cosine similarity alone.

    Default model: cross-encoder/ms-marco-MiniLM-L-6-v2 (fast, strong on passage retrieval).
    Swap for cross-encoder/ms-marco-electra-base for higher accuracy at ~3× cost.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required for CrossEncoderReranker. "
                "Install it with: uv add sentence-transformers"
            ) from e
        self.model = CrossEncoder(model_name)
        self.model_name = model_name

    def rerank(self, query: str, documents: list[Document], top_k: int) -> list[Document]:
        if not documents:
            return []

        pairs = [(query, doc.page_content) for doc in documents]
        scores = self.model.predict(pairs)

        scored = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)

        print(f"\n  [rerank] cross-encoder '{self.model_name}' scored {len(documents)} candidates → top {top_k}")
        print(f"  {'#':<5} {'Score':>10} {'Chunk ID':<40}")
        print(f"  {'-'*5} {'-'*10} {'-'*40}")
        for i, (score, doc) in enumerate(scored[:top_k]):
            print(f"  {i:<5} {float(score):>10.4f} {doc.metadata.get('chunk_id', ''):<40}")
        print()

        return [
            Document(
                page_content=doc.page_content,
                metadata={**doc.metadata, "rerank_score": round(float(score), 6)},
            )
            for score, doc in scored[:top_k]
        ]
