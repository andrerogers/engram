"""DoclingTextProcessor stub — real implementation wired in PR E8.

Placeholder so E3 can reference the class in the processor registry
without requiring Docling to be running. Raises NotImplementedError
until E8 replaces this with the real implementation.
"""

from __future__ import annotations

from engram.processors.base import ChunkCandidate


class DoclingTextProcessor:
    """Docling-backed text processor (stub — implemented in E8)."""

    def process(self, text: str) -> list[ChunkCandidate]:
        raise NotImplementedError(
            "DoclingTextProcessor not yet implemented — use TiktokenProcessor"
        )
