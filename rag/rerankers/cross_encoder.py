import logging

from langchain_core.documents import Document

from rag.rerankers.base import BaseReranker

logger = logging.getLogger(__name__)


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

        logger.info("[rerank] cross-encoder '%s' scored %d candidates → top %d", self.model_name, len(documents), top_k)
        if logger.isEnabledFor(logging.DEBUG):
            rows = "\n".join(f"  {i:<5} {float(s):>10.4f} {d.metadata.get('chunk_id', ''):<40}"
                             for i, (s, d) in enumerate(scored[:top_k]))
            logger.debug("[rerank] scores:\n%s", rows)

        return [
            Document(
                page_content=doc.page_content,
                metadata={**doc.metadata, "rerank_score": round(float(score), 6)},
            )
            for score, doc in scored[:top_k]
        ]
