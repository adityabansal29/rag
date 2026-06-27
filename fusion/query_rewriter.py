from pydantic import BaseModel

from rag.llm.base import BaseLLMClient
from rag.conversation.history import ConversationHistory


class _RewriteOnly(BaseModel):
    queries: list[str]


class _ContextualizeAndRewrite(BaseModel):
    standalone: str     # follow-up rewritten as a self-contained question
    variants: list[str] # n alternative phrasings for retrieval


_REWRITE_SYSTEM = """You are an expert at reformulating search queries to improve document retrieval.
Given an original query, generate alternative versions that:
- Rephrase the question from different angles
- Add more specific technical terms where relevant
- Decompose into sub-questions if the query is complex
- Vary perspective (definition, mechanism, example, comparison)

Each query must be distinct and designed to retrieve different but relevant documents."""

_CONTEXTUALIZE_AND_REWRITE_SYSTEM = """You are an expert at reformulating conversational follow-up questions for document retrieval.
Given a conversation history and a follow-up question:
1. Rewrite the follow-up as a fully self-contained standalone question (resolve all pronouns and references).
2. Generate alternative phrasings of that standalone question for better retrieval coverage.

Each variant must be distinct and designed to retrieve different but relevant documents."""


class QueryRewriter:
    def __init__(self, llm: BaseLLMClient):
        self.llm = llm

    async def rewrite(
        self,
        query: str,
        n: int = 3,
        history: ConversationHistory | None = None,
    ) -> tuple[str, list[str]]:
        """Returns (standalone_query, variants). When no history, standalone == query."""
        try:
            if history is not None and not history.is_empty():
                result: _ContextualizeAndRewrite = await self.llm.call_text(
                    system_prompt=_CONTEXTUALIZE_AND_REWRITE_SYSTEM,
                    content=(
                        f"Conversation history:\n{history.format()}\n\n"
                        f"Follow-up question: {query}\n\n"
                        f"Generate the standalone question and exactly {n} variants."
                    ),
                    response_model=_ContextualizeAndRewrite,
                )
                return result.standalone, result.variants
            else:
                result: _RewriteOnly = await self.llm.call_text(
                    system_prompt=_REWRITE_SYSTEM,
                    content=f"Original query: {query}\n\nGenerate exactly {n} alternative queries.",
                    response_model=_RewriteOnly,
                )
                return query, result.queries
        except Exception as e:
            print(f"  [rewriter] warning: LLM call failed ({e}), skipping rewrite")
            return query, []
