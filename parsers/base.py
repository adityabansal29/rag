from abc import ABC, abstractmethod

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
