"""Processor base types — modality enum, chunker kind enum, ChunkCandidate dataclass.

All processors produce a list[ChunkCandidate] from a text or file input.
The store layer accepts ChunkCandidate directly so chunker metadata is
persisted alongside the vector.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol


class Modality(StrEnum):
    """Content modality recorded on each chunk row."""

    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


class ChunkerKind(StrEnum):
    """Identifies which chunker produced a chunk — observable in telemetry."""

    TIKTOKEN_FALLBACK = "tiktoken-fallback"
    DOCLING_HYBRID = "docling-hybrid"
    DOCLING_HIERARCHICAL = "docling-hierarchical"
    DOCLING_MARKDOWN = "docling-markdown"


@dataclass
class ChunkCandidate:
    """A chunk ready to embed and store.

    Fields:
        content:          The text content of the chunk.
        chunk_index:      Position within the source document (0-based).
        modality:         Content modality (default: text).
        chunker:          Which chunker produced this chunk.
        chunker_version:  Optional version string for the chunker.
        media_ref:        Optional reference to a media object in object storage.
        media_metadata:   Optional freeform metadata dict (e.g. image dimensions).
    """

    content: str
    chunk_index: int
    modality: Modality = Modality.TEXT
    chunker: ChunkerKind = ChunkerKind.TIKTOKEN_FALLBACK
    chunker_version: str | None = None
    media_ref: str | None = None
    media_metadata: dict[str, object] = field(default_factory=dict)


class TextProcessor(Protocol):
    """Protocol for text-input processors."""

    def process(self, text: str) -> list[ChunkCandidate]:
        """Chunk *text* into candidates."""
        ...


class FileProcessor(Protocol):
    """Protocol for binary-file processors (PDF, DOCX, etc.)."""

    async def process(self, data: bytes, filename: str) -> list[ChunkCandidate]:
        """Parse and chunk *data* from *filename*."""
        ...
