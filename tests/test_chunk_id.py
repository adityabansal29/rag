"""Tests for Task 7: chunk ID must be content-addressed (no silent overwrite on re-parse)."""
from rag.models import Chunk


def test_same_content_same_id():
    """Idempotent upsert: re-indexing the same content produces the same ID."""
    id1 = Chunk.make_id("doc.pdf", page=1, index=0, content="hello world")
    id2 = Chunk.make_id("doc.pdf", page=1, index=0, content="hello world")
    assert id1 == id2


def test_different_content_different_id():
    """Different content at the same (source, page, index) position → different ID."""
    id_a = Chunk.make_id("doc.pdf", page=1, index=0, content="chunk A content")
    id_b = Chunk.make_id("doc.pdf", page=1, index=0, content="different content entirely")
    assert id_a != id_b, \
        "make_id must incorporate content so re-parsed chunks with changed content get a new ID"


def test_make_id_signature_accepts_content():
    """make_id must accept a content parameter."""
    import inspect
    sig = inspect.signature(Chunk.make_id)
    assert "content" in sig.parameters, \
        "Chunk.make_id must have a 'content' parameter for content-addressing"
