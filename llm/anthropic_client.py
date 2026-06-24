from anthropic import AsyncAnthropic

from rag.llm.base import BaseLLMClient


class AnthropicLLMClient(BaseLLMClient):
    def __init__(self, model: str = "claude-opus-4-7"):
        self.client = AsyncAnthropic()
        self.model = model

    async def call_text(self, system_prompt: str, content: str) -> str:
        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=512,
                system=system_prompt,
                messages=[{"role": "user", "content": content}],
            )
            return response.content[0].text
        except Exception as e:
            return f"[enrichment error: {str(e)}]"

    async def call_vision(self, system_prompt: str, image_b64: str) -> str:
        if not image_b64:
            return "[no image data]"
        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=512,
                system=system_prompt,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image_b64,
                            },
                        }
                    ],
                }],
            )
            return response.content[0].text
        except Exception as e:
            return f"[vision enrichment error: {str(e)}]"
