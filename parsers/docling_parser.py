import base64
from pathlib import Path

from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import DoclingDocument
from docling.datamodel.base_models import DocItemLabel

from rag.models import Chunk, ChunkType
from rag.parsers.base import BaseParser


class DoclingParser(BaseParser):
    """
    Parses PDF/docx using Docling library.
    Converts docling document model into parent->children Chunk hierarchy.
    heading element = parent, all elements under that heading = children.
    """

    def __init__(self):
        self.converter = DocumentConverter()

    def parse(self, file_path: str) -> list[Chunk]:
        source = Path(file_path).name

        # convert document via docling
        result = self.converter.convert(file_path)
        doc: DoclingDocument = result.document

        parent_chunks: list[Chunk] = []
        current_parent: Chunk | None = None
        child_index = 0
        elem_index = 0

        for item, level in doc.iterate_items():
            label = getattr(item, "label", None)
            page = self._get_page(item)

            # heading = new parent
            if label in (
                DocItemLabel.SECTION_HEADER,
                DocItemLabel.TITLE,
                DocItemLabel.PAGE_HEADER,
            ):
                heading_text = item.text if hasattr(item, "text") else str(item)
                current_parent = Chunk(
                    id=Chunk.make_id(source, page, elem_index),
                    parent_id=None,
                    chunk_type=ChunkType.TEXT,
                    metadata={
                        "source":        source,
                        "file_path":     file_path,
                        "page":          page,
                        "heading_level": self._heading_level(label, level),
                        "section_title": heading_text,
                    },
                    raw_content=heading_text,
                )
                parent_chunks.append(current_parent)
                child_index = 0
                elem_index += 1
                continue

            # no heading seen yet
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

            # convert to child chunk
            child = self._to_child_chunk(
                item=item,
                label=label,
                source=source,
                page=page,
                parent_id=current_parent.id,
                index=child_index,
            )

            if child:
                current_parent.add_child(child)
                child_index += 1

            elem_index += 1

        return parent_chunks

    def _to_child_chunk(
        self,
        item,
        label,
        source: str,
        page: int,
        parent_id: str,
        index: int,
    ) -> Chunk | None:

        chunk_id = Chunk.make_id(source, page, index)
        base_metadata = {
            "source":    source,
            "page":      page,
        }

        # text elements
        if label in (
            DocItemLabel.TEXT,
            DocItemLabel.PARAGRAPH,
            DocItemLabel.LIST_ITEM,
            DocItemLabel.CAPTION,
            DocItemLabel.FOOTNOTE,
        ):
            text = item.text if hasattr(item, "text") else str(item)
            return Chunk(
                id=chunk_id,
                parent_id=parent_id,
                chunk_type=ChunkType.TEXT,
                metadata=base_metadata,
                raw_content=text,
            )

        # table
        if label == DocItemLabel.TABLE:
            html = item.export_to_html() if hasattr(item, "export_to_html") else ""
            markdown = item.export_to_markdown() if hasattr(item, "export_to_markdown") else ""
            return Chunk(
                id=chunk_id,
                parent_id=parent_id,
                chunk_type=ChunkType.TABLE,
                metadata={
                    **base_metadata,
                    "html":     html,
                    "markdown": markdown,
                },
                raw_content=html or markdown,
            )

        # picture / image
        if label == DocItemLabel.PICTURE:
            b64 = self._extract_image_b64(item)
            return Chunk(
                id=chunk_id,
                parent_id=parent_id,
                chunk_type=ChunkType.IMAGE,
                metadata=base_metadata,
                raw_content=b64,
            )

        # code
        if label == DocItemLabel.CODE:
            text = item.text if hasattr(item, "text") else str(item)
            return Chunk(
                id=chunk_id,
                parent_id=parent_id,
                chunk_type=ChunkType.CODE,
                metadata={
                    **base_metadata,
                    "language": getattr(item, "language", ""),
                },
                raw_content=text,
            )

        return None

    def _get_page(self, item) -> int:
        try:
            return item.prov[0].page_no if item.prov else 0
        except Exception:
            return 0

    def _heading_level(self, label, level) -> int:
        if label == DocItemLabel.TITLE:
            return 1
        # docling level is 1-based
        return level if isinstance(level, int) else 1

    def _extract_image_b64(self, item) -> str:
        try:
            image = item.get_image()
            if image:
                import io
                buf = io.BytesIO()
                image.save(buf, format="PNG")
                return base64.b64encode(buf.getvalue()).decode("utf-8")
        except Exception:
            pass
        return ""
