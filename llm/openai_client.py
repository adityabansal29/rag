from typing import TypeVar, Type

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel

from rag.llm.base import BaseLLMClient

T = TypeVar("T", bound=BaseModel)


class OpenAILLMClient(BaseLLMClient):
    def __init__(self, model: str = "gpt-4o"):
        self.llm = ChatOpenAI(model=model, temperature=0, max_tokens=1024)

    async def call_text(
        self,
        system_prompt: str,
        content: str,
        response_model: Type[T] | None = None,
    ) -> str | T:
        messages = [SystemMessage(system_prompt), HumanMessage(content)]
        if response_model is not None:
            return await self.llm.with_structured_output(response_model).ainvoke(messages)
        try:
            result = await self.llm.ainvoke(messages)
            return result.content
        except Exception as e:
            return f"[enrichment error: {str(e)}]"

    async def call_vision(self, system_prompt: str, image_b64: str) -> str:
        if not image_b64:
            return "[no image data]"
        messages = [
            SystemMessage(system_prompt),
            HumanMessage(content=[{
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_b64}", "detail": "high"},
            }]),
        ]
        try:
            result = await self.llm.ainvoke(messages)
            return result.content
        except Exception as e:
            return f"[vision enrichment error: {str(e)}]"
