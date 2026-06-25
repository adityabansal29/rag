import asyncio

from rag.models import Chunk, ChunkType
from rag.llm.base import BaseLLMClient
from collections import Counter


SYSTEM_PROMPTS: dict[ChunkType, str] = {
    ChunkType.IMAGE: """You are an expert at analyzing images.
Describe the image in detail. Include:
- What is shown (objects, people, charts, diagrams)
- Key information, numbers, labels visible
- Overall purpose or message of the image
Be concise but complete. 2-4 sentences.""",

    ChunkType.TABLE: """You are an expert at analyzing data tables.
Summarize the table content. Include:
- What the table is about
- Key data points, trends, or comparisons
- Column headers and what they represent
- Any notable values or outliers
Be concise but complete. 2-4 sentences.""",

    ChunkType.CODE: """You are an expert software engineer.
Explain what this code does. Include:
- Purpose and functionality
- Key operations performed
- Inputs and outputs if visible
- Language if identifiable
Be concise but complete. 2-4 sentences.""",

    ChunkType.DIAGRAM: """You are an expert at reading technical diagrams.
Describe this diagram. Include:
- Type of diagram (flowchart, UML, architecture etc)
- Main components or nodes
- Key relationships or flows shown
- Overall purpose
Be concise but complete. 2-4 sentences.""",
}

PARENT_SUMMARY_PROMPT = """You are an expert at summarizing document sections.
Given the following content from a document section, write a concise summary.
Capture the main topics, key facts, and important details.
3-5 sentences maximum."""


class LLMEnricher:
    """
    Async LLM enricher. Accepts any BaseLLMClient or a provider string.
    - Enriches non-text child chunks with retrieved_content
    - Text child chunks use raw_content as retrieved_content directly
    - Summarizes parent chunk from all children retrieved_content
    """

    def __init__(
        self,
        provider: str | BaseLLMClient = "openai",
        model: str | None = None,
        max_concurrency: int = 10,
    ):
        self.semaphore = asyncio.Semaphore(max_concurrency)

        if isinstance(provider, BaseLLMClient):
            self.llm = provider
        else:
            self.llm = self._build_client(provider, model)

    def _build_client(self, provider: str, model: str | None) -> BaseLLMClient:
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
                raise ValueError(
                    f"Unknown provider: {provider!r}. Choose 'openai', 'anthropic', or 'gemini'."
                )

    async def enrich_all(self, parent_chunks: list[Chunk]) -> None:
        all_children: list[Chunk] = []
        for parent in parent_chunks:
            all_children.extend(parent.children)

        await asyncio.gather(*[self._enrich_child(child) for child in all_children])
        await asyncio.gather(*[self._summarize_parent(parent) for parent in parent_chunks])

    async def _enrich_child(self, chunk: Chunk) -> None:
        if chunk.chunk_type == ChunkType.TEXT:
            chunk.retrieved_content = chunk.raw_content
            return

        async with self.semaphore:
            system_prompt = SYSTEM_PROMPTS.get(chunk.chunk_type, "Describe this content.")

            if chunk.chunk_type == ChunkType.IMAGE:
                chunk.retrieved_content = await self.llm.call_vision(
                    system_prompt=system_prompt,
                    image_b64=chunk.raw_content or "",
                )
            else:
                chunk.retrieved_content = await self.llm.call_text(
                    system_prompt=system_prompt,
                    content=str(chunk.raw_content or ""),
                )
            print(f"  [enriched] type={chunk.chunk_type.value}  page={chunk.metadata.get('page', '?')}  id={chunk.id}  → {chunk.retrieved_content[:50]!r}")

    async def _summarize_parent(self, parent: Chunk) -> None:
        if not parent.children:
            parent.retrieved_content = parent.raw_content
            return

        combined = "\n\n".join(
            f"[{child.chunk_type.value.upper()}]\n{child.retrieved_content or ''}"
            for child in parent.children
            if child.retrieved_content
        )

        if not combined.strip():
            parent.retrieved_content = parent.raw_content
            return

        async with self.semaphore:
            parent.retrieved_content = await self.llm.call_text(
                system_prompt=PARENT_SUMMARY_PROMPT,
                content=combined,
            )
            print(f"  [parent enriched] id={parent.id}  section={parent.metadata.get('section_title', '')[:30]!r}  → {parent.retrieved_content[:50]!r}")


def build_embedding_content(parent_chunks: list[Chunk]) -> None:
    for parent in parent_chunks:
        section_title = parent.metadata.get("section_title", "")
        for child in parent.children:
            child.embedding_content = (
                f"Section: {section_title}\n"
                f"Type: {child.chunk_type.value}\n"
                f"Content: {child.retrieved_content or ''}"
            )
