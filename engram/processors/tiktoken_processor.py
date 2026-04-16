"""TiktokenProcessor — wraps the existing chunk_text() chunker.

Used as the tiktoken-fallback when Docling is unreachable.
The ``chunker`` field on every produced ChunkCandidate is explicitly
set to ``ChunkerKind.TIKTOKEN_FALLBACK`` so fallback paths are
observable in telemetry and test assertions.
"""

from __future__ import annotations

from engram.chunker import chunk_text
from engram.processors.base import ChunkCandidate, ChunkerKind, Modality


class TiktokenProcessor:
    """Async text processor backed by tiktoken fixed-size chunking."""

    CHUNKER_VERSION = "tiktoken-cl100k-512"

    async def process(self, text: str) -> list[ChunkCandidate]:
        """Split *text* into ChunkCandidates using tiktoken chunking."""
        raw_chunks = chunk_text(text)
        return [
            ChunkCandidate(
                content=chunk,
                chunk_index=i,
                modality=Modality.TEXT,
                chunker=ChunkerKind.TIKTOKEN_FALLBACK,
                chunker_version=self.CHUNKER_VERSION,
            )
            for i, chunk in enumerate(raw_chunks)
        ]
