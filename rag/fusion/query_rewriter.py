import logging

from pydantic import BaseModel

from rag.llm.base import BaseLLMClient

logger = logging.getLogger(__name__)


class _RewriteOnly(BaseModel):
    queries: list[str]


_REWRITE_SYSTEM = """You are an expert at reformulating search queries to improve document retrieval.
Given an original query, generate alternative versions that:
- Rephrase the question from different angles
- Add more specific technical terms where relevant
- Decompose into sub-questions if the query is complex
- Vary perspective (definition, mechanism, example, comparison)

Each query must be distinct and designed to retrieve different but relevant documents."""


class QueryRewriter:
    def __init__(self, llm: BaseLLMClient):
        self.llm = llm

    async def rewrite(self, query: str, n: int = 3) -> tuple[str, list[str]]:
        """Returns (query, variants)."""
        try:
            result: _RewriteOnly = await self.llm.call_text(
                system_prompt=_REWRITE_SYSTEM,
                content=f"Original query: {query}\n\nGenerate exactly {n} alternative queries.",
                response_model=_RewriteOnly,
            )
            variants = result.queries
            if not variants:
                logger.warning("[rewriter] LLM returned 0 variants, fusion will run on original query only")
            elif len(variants) != n:
                logger.warning("[rewriter] expected %d variants, got %d", n, len(variants))
            return query, variants
        except Exception as e:
            logger.warning("[rewriter] LLM call failed (%s), skipping rewrite", e)
            return query, []
