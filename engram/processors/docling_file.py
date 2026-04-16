"""DoclingFileProcessor — hybrid-chunking of binary files via Docling.

Binary files (PDF, DOCX, etc.) require Docling for text extraction.
Unlike the text processor, there is no tiktoken fallback — ``DoclingUnavailable``
propagates to the caller (the E9 job runner marks the job as failed).
"""

from __future__ import annotations

from engram.clients.docling import DoclingClient
from engram.processors.base import ChunkCandidate, ChunkerKind, Modality


class DoclingFileProcessor:
    """Async file processor that uses Docling hybrid chunking.

    Constructed with a shared ``DoclingClient``.  ``DoclingUnavailable``
    propagates to callers — binary files cannot be chunked without Docling.
    """

    def __init__(self, client: DoclingClient) -> None:
        self._client = client

    async def process(self, data: bytes, filename: str) -> list[ChunkCandidate]:
        """Chunk *data* from *filename* via Docling hybrid chunking.

        Raises:
            DoclingUnavailable: when Docling is disabled or unreachable.
            DoclingTaskFailed:  when Docling fails to process the file.
        """
        raw_chunks = await self._client.chunk_hybrid_file(file_bytes=data, filename=filename)
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
