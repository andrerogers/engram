"""DoclingTextProcessor — hybrid-chunking of raw text via Docling.

Falls back to TiktokenProcessor when Docling is disabled or unreachable
(``DoclingUnavailable``), so the /index route degrades gracefully without
any changes to the caller.
"""

from __future__ import annotations

import logging

from engram.clients.docling import DoclingClient, DoclingUnavailable
from engram.processors.base import ChunkCandidate, ChunkerKind, Modality

log = logging.getLogger(__name__)


class DoclingTextProcessor:
    """Async text processor that uses Docling hybrid chunking.

    Constructed with a shared ``DoclingClient``.  On ``DoclingUnavailable``
    the processor transparently falls back to tiktoken so callers do not
    need conditional logic.
    """

    def __init__(self, client: DoclingClient) -> None:
        self._client = client

    async def process(self, text: str) -> list[ChunkCandidate]:
        """Chunk *text* via Docling hybrid chunking.

        Falls back to ``TiktokenProcessor`` when Docling is unavailable.
        """
        try:
            raw_chunks = await self._client.chunk_text_hybrid(text=text, filename="input.md")
        except DoclingUnavailable:
            log.warning("Docling unavailable — falling back to tiktoken for text chunk")
            from engram.processors.tiktoken_processor import TiktokenProcessor

            return await TiktokenProcessor().process(text)

        return [
            ChunkCandidate(
                content=chunk["text"],
                chunk_index=i,
                modality=Modality.TEXT,
                chunker=ChunkerKind.DOCLING_HYBRID,
            )
            for i, chunk in enumerate(raw_chunks)
            if str(chunk.get("text", "")).strip()
        ]
