from pydantic import BaseModel

from rag.llm.base import BaseLLMClient
from rag.conversation.history import ConversationHistory


_SYSTEM_PROMPT = """You are a conversational assistant helping reformulate follow-up questions.
Given a conversation history and a follow-up question, rewrite the question as a fully \
self-contained standalone question that can be understood and answered without any prior context.
Do not answer the question — only rewrite it.
If the follow-up question is already self-contained, return it verbatim."""


class StandaloneQuery(BaseModel):
    query: str


class QueryContextualizer:
    def __init__(self, llm: BaseLLMClient):
        self.llm = llm

    async def contextualize(self, query: str, history: ConversationHistory) -> str:
        if history.is_empty():
            return query

        try:
            result: StandaloneQuery = await self.llm.call_text(
                system_prompt=_SYSTEM_PROMPT,
                content=f"Conversation history:\n{history.format()}\n\nFollow-up question: {query}",
                response_model=StandaloneQuery,
            )
            return result.query
        except Exception as e:
            print(f"  [contextualize] warning: LLM call failed ({e}), using original query")
            return query
