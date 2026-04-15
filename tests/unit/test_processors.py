"""Tests for engram.processors — TiktokenProcessor + base types."""

from __future__ import annotations

from engram.processors.base import ChunkCandidate, ChunkerKind, Modality
from engram.processors.tiktoken_processor import TiktokenProcessor

_SAMPLE_TEXT = "\n".join(f"Line {i}: " + "word " * 20 for i in range(30))


# ---------------------------------------------------------------------------
# ChunkCandidate dataclass
# ---------------------------------------------------------------------------


def test_chunk_candidate_defaults() -> None:
    c = ChunkCandidate(content="hello", chunk_index=0)
    assert c.modality == Modality.TEXT
    assert c.chunker == ChunkerKind.TIKTOKEN_FALLBACK
    assert c.chunker_version is None
    assert c.media_ref is None
    assert c.media_metadata == {}


def test_chunk_candidate_fields_stored() -> None:
    c = ChunkCandidate(
        content="some text",
        chunk_index=3,
        modality=Modality.IMAGE,
        chunker=ChunkerKind.DOCLING_HYBRID,
        chunker_version="1.2.3",
        media_ref="objects/img.png",
        media_metadata={"width": 800},
    )
    assert c.content == "some text"
    assert c.chunk_index == 3
    assert c.modality == Modality.IMAGE
    assert c.chunker == ChunkerKind.DOCLING_HYBRID
    assert c.chunker_version == "1.2.3"
    assert c.media_ref == "objects/img.png"
    assert c.media_metadata == {"width": 800}


# ---------------------------------------------------------------------------
# TiktokenProcessor
# ---------------------------------------------------------------------------


def test_tiktoken_processor_returns_candidates() -> None:
    proc = TiktokenProcessor()
    results = proc.process(_SAMPLE_TEXT)
    assert isinstance(results, list)
    assert all(isinstance(c, ChunkCandidate) for c in results)


def test_tiktoken_processor_nonempty_content() -> None:
    proc = TiktokenProcessor()
    for candidate in proc.process(_SAMPLE_TEXT):
        assert candidate.content.strip()


def test_tiktoken_processor_chunker_is_fallback() -> None:
    """Every candidate must be labelled tiktoken-fallback — no ambiguity."""
    proc = TiktokenProcessor()
    for candidate in proc.process(_SAMPLE_TEXT):
        assert candidate.chunker == ChunkerKind.TIKTOKEN_FALLBACK


def test_tiktoken_processor_modality_is_text() -> None:
    proc = TiktokenProcessor()
    for candidate in proc.process(_SAMPLE_TEXT):
        assert candidate.modality == Modality.TEXT


def test_tiktoken_processor_chunk_index_sequential() -> None:
    proc = TiktokenProcessor()
    candidates = proc.process(_SAMPLE_TEXT)
    for expected_i, c in enumerate(candidates):
        assert c.chunk_index == expected_i


def test_tiktoken_processor_empty_input_returns_empty() -> None:
    proc = TiktokenProcessor()
    assert proc.process("") == []
    assert proc.process("   ") == []


def test_tiktoken_processor_version_set() -> None:
    proc = TiktokenProcessor()
    candidates = proc.process("some text to chunk")
    assert all(c.chunker_version == TiktokenProcessor.CHUNKER_VERSION for c in candidates)
