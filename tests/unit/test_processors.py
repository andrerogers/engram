"""Tests for engram.processors — TiktokenProcessor, DoclingTextProcessor,
DoclingFileProcessor, and the get_text_processor / get_file_processor registry."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from engram.clients.docling import DoclingUnavailable
from engram.processors.base import ChunkCandidate, ChunkerKind, Modality
from engram.processors.docling_file import DoclingFileProcessor
from engram.processors.docling_text import DoclingTextProcessor
from engram.processors.tiktoken_processor import TiktokenProcessor

_SAMPLE_TEXT = "\n".join(f"Line {i}: " + "word " * 20 for i in range(30))

_DOCLING_CHUNKS = [
    {"text": "First chunk from Docling.", "meta": {}},
    {"text": "Second chunk from Docling.", "meta": {}},
]


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


async def test_tiktoken_processor_returns_candidates() -> None:
    proc = TiktokenProcessor()
    results = await proc.process(_SAMPLE_TEXT)
    assert isinstance(results, list)
    assert all(isinstance(c, ChunkCandidate) for c in results)


async def test_tiktoken_processor_nonempty_content() -> None:
    proc = TiktokenProcessor()
    for candidate in await proc.process(_SAMPLE_TEXT):
        assert candidate.content.strip()


async def test_tiktoken_processor_chunker_is_fallback() -> None:
    """Every candidate must be labelled tiktoken-fallback — no ambiguity."""
    proc = TiktokenProcessor()
    for candidate in await proc.process(_SAMPLE_TEXT):
        assert candidate.chunker == ChunkerKind.TIKTOKEN_FALLBACK


async def test_tiktoken_processor_modality_is_text() -> None:
    proc = TiktokenProcessor()
    for candidate in await proc.process(_SAMPLE_TEXT):
        assert candidate.modality == Modality.TEXT


async def test_tiktoken_processor_chunk_index_sequential() -> None:
    proc = TiktokenProcessor()
    candidates = await proc.process(_SAMPLE_TEXT)
    for expected_i, c in enumerate(candidates):
        assert c.chunk_index == expected_i


async def test_tiktoken_processor_empty_input_returns_empty() -> None:
    proc = TiktokenProcessor()
    assert await proc.process("") == []
    assert await proc.process("   ") == []


async def test_tiktoken_processor_version_set() -> None:
    proc = TiktokenProcessor()
    candidates = await proc.process("some text to chunk")
    assert all(c.chunker_version == TiktokenProcessor.CHUNKER_VERSION for c in candidates)


# ---------------------------------------------------------------------------
# DoclingTextProcessor — happy path
# ---------------------------------------------------------------------------


def _mock_docling_client(chunks: list[dict] | None = None) -> MagicMock:
    client = MagicMock()
    client.chunk_text_hybrid = AsyncMock(return_value=chunks or _DOCLING_CHUNKS)
    return client


async def test_docling_text_processor_returns_candidates() -> None:
    proc = DoclingTextProcessor(_mock_docling_client())
    results = await proc.process("some text")
    assert isinstance(results, list)
    assert len(results) == 2
    assert all(isinstance(c, ChunkCandidate) for c in results)


async def test_docling_text_processor_chunker_is_docling_hybrid() -> None:
    proc = DoclingTextProcessor(_mock_docling_client())
    for candidate in await proc.process("some text"):
        assert candidate.chunker == ChunkerKind.DOCLING_HYBRID


async def test_docling_text_processor_modality_is_text() -> None:
    proc = DoclingTextProcessor(_mock_docling_client())
    for candidate in await proc.process("some text"):
        assert candidate.modality == Modality.TEXT


async def test_docling_text_processor_content_matches_chunks() -> None:
    proc = DoclingTextProcessor(_mock_docling_client())
    candidates = await proc.process("some text")
    assert candidates[0].content == "First chunk from Docling."
    assert candidates[1].content == "Second chunk from Docling."


async def test_docling_text_processor_chunk_index_sequential() -> None:
    proc = DoclingTextProcessor(_mock_docling_client())
    candidates = await proc.process("some text")
    for expected_i, c in enumerate(candidates):
        assert c.chunk_index == expected_i


async def test_docling_text_processor_skips_empty_chunks() -> None:
    chunks = [{"text": "good chunk"}, {"text": "   "}, {"text": ""}]
    proc = DoclingTextProcessor(_mock_docling_client(chunks))
    candidates = await proc.process("text")
    assert len(candidates) == 1
    assert candidates[0].content == "good chunk"


# ---------------------------------------------------------------------------
# DoclingTextProcessor — fallback on DoclingUnavailable
# ---------------------------------------------------------------------------


async def test_docling_text_processor_falls_back_on_unavailable() -> None:
    """DoclingUnavailable must trigger silent tiktoken fallback."""
    client = MagicMock()
    client.chunk_text_hybrid = AsyncMock(side_effect=DoclingUnavailable("off"))
    proc = DoclingTextProcessor(client)
    results = await proc.process(_SAMPLE_TEXT)
    # Tiktoken fallback produces ChunkCandidates labelled tiktoken-fallback
    assert len(results) > 0
    assert all(c.chunker == ChunkerKind.TIKTOKEN_FALLBACK for c in results)


async def test_docling_text_processor_fallback_empty_input() -> None:
    client = MagicMock()
    client.chunk_text_hybrid = AsyncMock(side_effect=DoclingUnavailable("off"))
    proc = DoclingTextProcessor(client)
    assert await proc.process("") == []


# ---------------------------------------------------------------------------
# DoclingFileProcessor — happy path
# ---------------------------------------------------------------------------


def _mock_file_client(chunks: list[dict] | None = None) -> MagicMock:
    client = MagicMock()
    client.chunk_hybrid_file = AsyncMock(return_value=chunks or _DOCLING_CHUNKS)
    return client


async def test_docling_file_processor_returns_candidates() -> None:
    proc = DoclingFileProcessor(_mock_file_client())
    results = await proc.process(b"fake pdf bytes", "doc.pdf")
    assert isinstance(results, list)
    assert len(results) == 2
    assert all(isinstance(c, ChunkCandidate) for c in results)


async def test_docling_file_processor_chunker_is_docling_hybrid() -> None:
    proc = DoclingFileProcessor(_mock_file_client())
    for candidate in await proc.process(b"bytes", "doc.pdf"):
        assert candidate.chunker == ChunkerKind.DOCLING_HYBRID


async def test_docling_file_processor_content_matches_chunks() -> None:
    proc = DoclingFileProcessor(_mock_file_client())
    candidates = await proc.process(b"bytes", "doc.pdf")
    assert candidates[0].content == "First chunk from Docling."
    assert candidates[1].content == "Second chunk from Docling."


async def test_docling_file_processor_skips_empty_chunks() -> None:
    chunks = [{"text": "real content"}, {"text": ""}, {"text": "  "}]
    proc = DoclingFileProcessor(_mock_file_client(chunks))
    candidates = await proc.process(b"bytes", "doc.pdf")
    assert len(candidates) == 1
    assert candidates[0].content == "real content"


async def test_docling_file_processor_propagates_unavailable() -> None:
    """DoclingUnavailable must propagate — no silent fallback for binary files."""
    client = MagicMock()
    client.chunk_hybrid_file = AsyncMock(side_effect=DoclingUnavailable("off"))
    proc = DoclingFileProcessor(client)
    with pytest.raises(DoclingUnavailable):
        await proc.process(b"bytes", "doc.pdf")


# ---------------------------------------------------------------------------
# Processor registry
# ---------------------------------------------------------------------------


async def test_get_text_processor_returns_docling_by_default() -> None:
    from engram.processors import get_text_processor

    client = MagicMock()
    with patch("engram.config.DEFAULT_TEXT_CHUNKER", "docling-hybrid"):
        proc = get_text_processor(client)
    assert isinstance(proc, DoclingTextProcessor)


async def test_get_text_processor_returns_tiktoken_when_configured() -> None:
    from engram.processors import get_text_processor

    client = MagicMock()
    with patch("engram.config.DEFAULT_TEXT_CHUNKER", "tiktoken"):
        proc = get_text_processor(client)
    assert isinstance(proc, TiktokenProcessor)


async def test_get_file_processor_returns_docling_file_processor() -> None:
    from engram.processors import get_file_processor

    client = MagicMock()
    proc = get_file_processor(client)
    assert isinstance(proc, DoclingFileProcessor)
