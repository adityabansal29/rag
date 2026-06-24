import base64
import os

from google import genai
from google.genai import types

from rag.llm.base import BaseLLMClient


class GeminiLLMClient(BaseLLMClient):
    def __init__(self, model: str = "gemini-2.0-flash"):
        self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model = model

    async def call_text(self, system_prompt: str, content: str) -> str:
        try:
            response = await self.client.aio.models.generate_content(
                model=self.model,
                contents=content,
                config={"system_instruction": system_prompt, "max_output_tokens": 512},
            )
            return response.text or ""
        except Exception as e:
            return f"[enrichment error: {str(e)}]"

    async def call_vision(self, system_prompt: str, image_b64: str) -> str:
        if not image_b64:
            return "[no image data]"
        try:
            response = await self.client.aio.models.generate_content(
                model=self.model,
                contents=[
                    types.Part.from_bytes(
                        data=base64.b64decode(image_b64),
                        mime_type="image/png",
                    ),
                    system_prompt,
                ],
                config={"max_output_tokens": 512},
            )
            return response.text or ""
        except Exception as e:
            return f"[vision enrichment error: {str(e)}]"
