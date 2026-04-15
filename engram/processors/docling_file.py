"""DoclingFileProcessor stub — real implementation wired in PR E8.

Placeholder so E3 can reference the class in the processor registry
without requiring Docling to be running.
"""

from __future__ import annotations

from engram.processors.base import ChunkCandidate


class DoclingFileProcessor:
    """Docling-backed file processor (stub — implemented in E8)."""

    async def process(self, data: bytes, filename: str) -> list[ChunkCandidate]:
        raise NotImplementedError("DoclingFileProcessor not yet implemented")
