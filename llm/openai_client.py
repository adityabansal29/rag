from openai import AsyncOpenAI

from rag.llm.base import BaseLLMClient


class OpenAILLMClient(BaseLLMClient):
    def __init__(self, model: str = "gpt-4o"):
        self.client = AsyncOpenAI()
        self.model = model

    async def call_text(self, system_prompt: str, content: str) -> str:
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": content},
                ],
                max_tokens=512,
                temperature=0,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            return f"[enrichment error: {str(e)}]"

    async def call_vision(self, system_prompt: str, image_b64: str) -> str:
        if not image_b64:
            return "[no image data]"
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_b64}",
                                    "detail": "high",
                                },
                            }
                        ],
                    },
                ],
                max_tokens=512,
                temperature=0,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            return f"[vision enrichment error: {str(e)}]"
