from abc import ABC, abstractmethod
from typing import TypeVar, Type

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class BaseLLMClient(ABC):
    @abstractmethod
    async def call_text(
        self,
        system_prompt: str,
        content: str,
        response_model: Type[T] | None = None,
    ) -> str | T:
        ...

    @abstractmethod
    async def call_vision(self, system_prompt: str, image_b64: str) -> str:
        ...


def build_llm_client(provider: str, model: str | None = None) -> "BaseLLMClient":
    match provider:
        case "openai":
            from rag.llm.openai_client import OpenAILLMClient
            return OpenAILLMClient(model=model or "gpt-4o")
        case "anthropic":
            from rag.llm.anthropic_client import AnthropicLLMClient
            return AnthropicLLMClient(model=model or "claude-opus-4-7")
        case "gemini":
            from rag.llm.gemini_client import GeminiLLMClient
            return GeminiLLMClient(model=model or "gemini-2.0-flash")
        case _:
            raise ValueError(f"Unknown provider: {provider!r}. Choose 'openai', 'anthropic', or 'gemini'.")
