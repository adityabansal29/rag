import base64
from pathlib import Path

from unstructured.partition.auto import partition
from unstructured.documents.elements import (
    Title, Text, Header
)

from rag.models import Chunk, ChunkType
from rag.parsers.base import BaseParser


class UnstructuredParser(BaseParser):
    """
    Parses PDF/docx using unstructured library.
    Converts elements into parent->children Chunk hierarchy.
    heading/Title element = parent, everything under it = children.
    """

    def __init__(self, strategy: str = "hi_res"):
        self.strategy = strategy

    def parse(self, file_path: str) -> list[Chunk]:
        source = Path(file_path).name

        # partition file into unstructured elements
        elements = partition(
            filename=file_path, # Path to your PDF file
            strategy=self.strategy, # Use the processing method of extraction
            extract_image_block_to_payload=True, # Store images as base64 data you can actually use
            infer_table_structure=True, # Keep tables as structured HTML, not jumbled text
            extract_image_block_types=["Image"], # Grab images found in the PDF
        )

        parent_chunks: list[Chunk] = []
        current_parent: Chunk | None = None
        child_index = 0

        for elem_index, elem in enumerate(elements):
            page = elem.metadata.page_number or 0

            # heading/title = new parent chunk
            if isinstance(elem, (Title, Header)):
                current_parent = Chunk(
                    id=Chunk.make_id(source, page, elem_index),
                    parent_id=None,
                    chunk_type=ChunkType.TEXT,
                    metadata={
                        "source":        source,
                        "file_path":     file_path,
                        "page":          page,
                        "heading_level": self._heading_level(elem),
                        "section_title": elem.text,
                    },
                    raw_content=elem.text,
                )
                parent_chunks.append(current_parent)
                child_index = 0
                continue

            # no heading seen yet — create default parent
            if current_parent is None:
                current_parent = Chunk(
                    id=Chunk.make_id(source, 0, -1),
                    parent_id=None,
                    chunk_type=ChunkType.TEXT,
                    metadata={
                        "source":        source,
                        "file_path":     file_path,
                        "page":          0,
                        "heading_level": 0,
                        "section_title": "document_start",
                    },
                    raw_content="document_start",
                )
                parent_chunks.append(current_parent)

            # convert element to child chunk
            child = self._to_child_chunk(
                elem=elem,
                source=source,
                page=page,
                parent_id=current_parent.id,
                index=child_index,
            )

            if child:
                current_parent.add_child(child)
                child_index += 1

        return parent_chunks

    def _to_child_chunk(
        self,
        elem,
        source: str,
        page: int,
        parent_id: str,
        index: int,
    ) -> Chunk | None:

        chunk_id = Chunk.make_id(source, page, index)
        base_metadata = {
            "source":    source,
            "file_path": elem.metadata.filename or source,
            "page":      page,
        }

        element_type = type(elem).__name__

        if element_type == 'Table':
            return Chunk(
                id=chunk_id,
                parent_id=parent_id,
                chunk_type=ChunkType.TABLE,
                metadata=base_metadata,
                raw_content=elem.metadata.text_as_html or elem.text,
            )

        elif element_type == 'Image':
            return Chunk(
                id=chunk_id,
                parent_id=parent_id,
                chunk_type=ChunkType.IMAGE,
                metadata=base_metadata,
                raw_content=elem.metadata.image_base64 or "",
            )

        elif element_type == 'CodeSnippet':
            return Chunk(
                id=chunk_id,
                parent_id=parent_id,
                chunk_type=ChunkType.CODE,
                metadata={
                    **base_metadata,
                    "language": getattr(elem.metadata, "language", ""),
                },
                raw_content=elem.text,
            )

        elif isinstance(elem, Text):
            return Chunk(
                id=chunk_id,
                parent_id=parent_id,
                chunk_type=ChunkType.TEXT,
                metadata=base_metadata,
                raw_content=elem.text,
            )

        return None

    def _heading_level(self, elem) -> int:
        category = getattr(elem, "category", "")
        if "H1" in category or isinstance(elem, Title):
            return 1
        if "H2" in category:
            return 2
        if "H3" in category:
            return 3
        return 1

    def _encode_image(self, image_path: str) -> str:
        if not image_path:
            return ""
        try:
            with open(image_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception:
            return ""
