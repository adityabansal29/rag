from abc import ABC, abstractmethod


class BaseLLMClient(ABC):
    @abstractmethod
    async def call_text(self, system_prompt: str, content: str) -> str:
        ...

    @abstractmethod
    async def call_vision(self, system_prompt: str, image_b64: str) -> str:
        ...
