from abc import ABC, abstractmethod
from collections import Counter

from rag.models import Chunk


class BaseParser(ABC):
    """
    Both parsers return list of parent Chunk objects.
    Each parent has children populated.
    heading/section = parent, all elements under it = children.
    """

    @abstractmethod
    def parse(self, file_path: str) -> list[Chunk]:
        """
        Parse file and return list of parent chunks.
        Each parent chunk has its children already attached.
        """
        ...

    @staticmethod
    def print_chunks(chunks: list[Chunk]) -> None:
        print(f"\n  {'#':>4}  {'Chunk ID':34}  {'Parent Text':50}  {'total':>5}  types")
        print(f"  {'─'*4}  {'─'*34}  {'─'*50}  {'─'*5}  {'─'*30}")
        for i, parent in enumerate(chunks):
            counts = Counter(child.chunk_type.value for child in parent.children)
            counts_str = "  ".join(f"{k}={v}" for k, v in sorted(counts.items()))
            title = str(parent.raw_content or "")[:50].replace("\n", " ")
            print(f"  {i:>4}  {parent.id:34}  {title:50}  {len(parent.children):>5}  {counts_str}")
        print()
