from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class ChunkType(str, Enum):
    TEXT    = "text"
    TABLE   = "table"
    IMAGE   = "image"
    CODE    = "code"
    DIAGRAM = "diagram"


@dataclass
class Chunk:
    """
    Represents both parent and child chunks.

    Parent chunk:
        - parent_id is None
        - children list is populated
        - raw_content is the heading/section title
        - retrieved_content is summarized from all children (set after enrichment)

    Child chunk:
        - parent_id points to parent chunk id
        - children list is empty
        - raw_content is text / base64 / html depending on chunk_type
        - retrieved_content is LLM enriched summary (set after enrichment)
        - embedding_content is final text sent to embedding model
    """

    id:                str
    parent_id:         Optional[str]
    chunk_type:        ChunkType
    metadata:          dict                       # source, page, section, heading_level etc
    raw_content:       Any                        # str / base64 str / html str
    retrieved_content: Optional[str]  = None      # LLM enriched content
    embedding_content: Optional[str]  = None      # final text for embedding
    children:          list[Chunk]    = field(default_factory=list)

    @staticmethod
    def make_id(source: str, page: int, index: int) -> str:
        key = f"{source}:{page}:{index}"
        return hashlib.md5(key.encode()).hexdigest()

    def is_parent(self) -> bool:
        return self.parent_id is None

    def is_text(self) -> bool:
        return self.chunk_type == ChunkType.TEXT

    def add_child(self, child: Chunk) -> None:
        self.children.append(child)

    def __repr__(self) -> str:
        role = "parent" if self.is_parent() else "child"
        content_preview = str(self.raw_content or "")[:80].replace("\n", " ")
        lines = [
            f"Chunk({role})",
            f"  id            : {self.id}",
            f"  type          : {self.chunk_type.value}",
            f"  page          : {self.metadata.get('page', '?')}",
            f"  source        : {self.metadata.get('source', '?')}",
            f"  raw_content   : {content_preview!r}",
        ]
        if self.retrieved_content:
            lines.append(f"  retrieved     : {self.retrieved_content[:80]!r}")
        if self.embedding_content:
            lines.append(f"  embedding     : {self.embedding_content[:80]!r}")
        if self.is_parent():
            lines.append(f"  children      : {len(self.children)}")
        return "\n".join(lines)
