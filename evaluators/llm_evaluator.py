import asyncio
from langchain_core.documents import Document
from pydantic import BaseModel

from rag.llm.base import BaseLLMClient, build_llm_client
from rag.evaluators.base import ChunkScore, EvalResult


class ChunkRelevanceResponse(BaseModel):
    chunks: list[ChunkScore]
    avg_relevance: float


class FaithfulnessResponse(BaseModel):
    faithfulness: float
    unsupported_claims: list[str] = []
    reason: str


_CHUNK_RELEVANCE_SYSTEM = """You are a RAG evaluation expert. Score each retrieved chunk's relevance to the user query.
Relevance scale: 1.0 = chunk directly and fully answers the query, 0.5 = partially relevant, 0.0 = completely unrelated."""

_FAITHFULNESS_SYSTEM = """You are a RAG evaluation expert. Check whether every factual claim in the generated answer is supported by the retrieved context chunks.
Faithfulness scale: 1.0 = every claim is grounded in the context, 0.0 = answer is entirely hallucinated."""


def _format_chunks(chunks: list[Document]) -> str:
    lines = []
    for i, doc in enumerate(chunks):
        chunk_id = doc.metadata.get("chunk_id", f"chunk_{i}")
        lines.append(f"[{i+1}] chunk_id={chunk_id}\n{doc.page_content[:500]}")
    return "\n\n".join(lines)


class LLMEvaluator:
    """
    Reference-free RAG evaluator.
    Scores chunk relevance and answer faithfulness in parallel via LLM.
    Accepts any BaseLLMClient or a provider string (same pattern as LLMEnricher).
    """

    def __init__(self, provider: str | BaseLLMClient = "openai", model: str | None = None):
        if isinstance(provider, BaseLLMClient):
            self.llm = provider
        else:
            self.llm = build_llm_client(provider, model)

    async def evaluate(
        self,
        query: str,
        chunks: list[Document],
        answer: str,
    ) -> EvalResult:
        chunk_resp, faith_resp = await asyncio.gather(
            self._score_chunk_relevance(query, chunks),
            self._score_faithfulness(query, chunks, answer),
        )

        return EvalResult(
            query=query,
            answer=answer,
            chunk_scores=chunk_resp.chunks,
            avg_chunk_relevance=chunk_resp.avg_relevance,
            faithfulness=faith_resp.faithfulness,
            unsupported_claims=faith_resp.unsupported_claims,
            faithfulness_reason=faith_resp.reason,
        )

    def evaluate_sync(
        self,
        query: str,
        chunks: list[Document],
        answer: str,
    ) -> EvalResult:
        return asyncio.run(self.evaluate(query, chunks, answer))

    async def _score_chunk_relevance(self, query: str, chunks: list[Document]) -> ChunkRelevanceResponse:
        user_msg = f"Query: {query}\n\nChunks:\n{_format_chunks(chunks)}"
        return await self.llm.call_text(
            system_prompt=_CHUNK_RELEVANCE_SYSTEM,
            content=user_msg,
            response_model=ChunkRelevanceResponse,
        )

    async def _score_faithfulness(self, query: str, chunks: list[Document], answer: str) -> FaithfulnessResponse:
        user_msg = (
            f"Query: {query}\n\n"
            f"Context:\n{_format_chunks(chunks)}\n\n"
            f"Answer: {answer}"
        )
        return await self.llm.call_text(
            system_prompt=_FAITHFULNESS_SYSTEM,
            content=user_msg,
            response_model=FaithfulnessResponse,
        )
