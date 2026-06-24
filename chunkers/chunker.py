import hashlib
from dataclasses import replace

import tiktoken

from rag.models import Chunk, ChunkType


# token count range for text child chunks
MIN_TOKENS = 100
MAX_TOKENS = 512


def _count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    enc = tiktoken.get_encoding(encoding_name)
    return len(enc.encode(text))


def _make_merged_id(parent_id: str, indices: list[int]) -> str:
    key = f"{parent_id}:merged:{'-'.join(str(i) for i in indices)}"
    return hashlib.md5(key.encode()).hexdigest()


def merge_split_text_chunks(parent: Chunk) -> Chunk:
    """
    For a given parent chunk, process its text children:
      - merge consecutive text chunks that are below MIN_TOKENS
      - split text chunks that exceed MAX_TOKENS

    Non-text children (image, table, code) are left untouched and
    re-inserted in their original relative position.

    Returns a new parent Chunk with updated children list.
    """
    processed: list[Chunk] = []
    text_buffer: list[Chunk] = []
    buffer_indices: list[int] = []

    def flush_buffer():
        """Merge buffered text chunks into one or more final chunks."""
        if not text_buffer:
            return

        combined_text = " ".join(c.raw_content for c in text_buffer)
        combined_tokens = _count_tokens(combined_text)

        if combined_tokens <= MAX_TOKENS:
            # entire buffer fits in one chunk
            merged = Chunk(
                id=_make_merged_id(parent.id, buffer_indices),
                parent_id=parent.id,
                chunk_type=ChunkType.TEXT,
                metadata={**text_buffer[0].metadata, "merged_count": len(text_buffer)},
                raw_content=combined_text,
            )
            processed.append(merged)
        else:
            # split combined text into MAX_TOKENS sized pieces
            enc = tiktoken.get_encoding("cl100k_base")
            tokens = enc.encode(combined_text)
            overlap = 50  # token overlap between split pieces

            start = 0
            split_index = 0
            while start < len(tokens):
                end = min(start + MAX_TOKENS, len(tokens))
                piece_tokens = tokens[start:end]
                piece_text = enc.decode(piece_tokens)

                split_id = hashlib.md5(
                    f"{parent.id}:split:{buffer_indices[0]}:{split_index}".encode()
                ).hexdigest()

                processed.append(Chunk(
                    id=split_id,
                    parent_id=parent.id,
                    chunk_type=ChunkType.TEXT,
                    metadata={
                        **text_buffer[0].metadata,
                        "split_index":  split_index,
                        "split_from":   buffer_indices,
                    },
                    raw_content=piece_text,
                ))
                start += MAX_TOKENS - overlap
                split_index += 1

        text_buffer.clear()
        buffer_indices.clear()

    for i, child in enumerate(parent.children):
        if child.chunk_type == ChunkType.TEXT:
            token_count = _count_tokens(child.raw_content or "")

            if not text_buffer:
                # start new buffer
                text_buffer.append(child)
                buffer_indices.append(i)
            else:
                buffer_tokens = _count_tokens(
                    " ".join(c.raw_content for c in text_buffer)
                )
                if buffer_tokens + token_count <= MAX_TOKENS:
                    # fits — add to buffer
                    text_buffer.append(child)
                    buffer_indices.append(i)
                else:
                    # flush current buffer, start new one
                    flush_buffer()
                    text_buffer.append(child)
                    buffer_indices.append(i)
        else:
            # non-text — flush pending text buffer first, then add as-is
            flush_buffer()
            processed.append(child)

    # flush any remaining text buffer
    flush_buffer()

    # return parent with updated children
    new_parent = Chunk(
        id=parent.id,
        parent_id=parent.parent_id,
        chunk_type=parent.chunk_type,
        metadata=parent.metadata,
        raw_content=parent.raw_content,
        retrieved_content=parent.retrieved_content,
        embedding_content=parent.embedding_content,
        children=processed,
    )
    return new_parent


def process_all_parents(parent_chunks: list[Chunk]) -> list[Chunk]:
    """Apply merge/split to all parent chunks."""
    return [merge_split_text_chunks(p) for p in parent_chunks]
