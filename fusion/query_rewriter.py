from pydantic import BaseModel

from rag.llm.base import BaseLLMClient


class RewrittenQueries(BaseModel):
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

    async def rewrite(self, query: str, n: int = 3) -> list[str]:
        result: RewrittenQueries = await self.llm.call_text(
            system_prompt=_REWRITE_SYSTEM,
            content=f"Original query: {query}\n\nGenerate exactly {n} alternative queries.",
            response_model=RewrittenQueries,
        )
        return result.queries
