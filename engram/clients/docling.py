"""Docling in-process — PDF/DOCX conversion and hybrid chunking, no sidecar.

Docling is an **optional extra** (`uv sync --extra docling`): it pulls torch and transformers,
several GB, which has no place in a CI install. When the package is absent, or
``DOCLING_ENABLED=false``, every call raises ``DoclingUnavailable``; text chunking falls back to
tiktoken and binary ingest fails with a message naming the extra.

Conversion and chunking are CPU-bound and synchronous, so each call runs in a worker thread. The
converter and chunker are built once at startup — the first conversion otherwise pays the model
load.
"""

from __future__ import annotations

import asyncio
import logging
from io import BytesIO
from typing import Any

from engram.config import DOCLING_ENABLED

log = logging.getLogger(__name__)


class DoclingUnavailable(RuntimeError):
    """Docling is disabled, not installed, or has not been started."""


class DoclingFailed(RuntimeError):
    """Docling could not convert or chunk the input."""


class DoclingEngine:
    """In-process Docling: convert bytes to a document, chunk it, export Markdown."""

    def __init__(self, enabled: bool | None = None) -> None:
        self._enabled = DOCLING_ENABLED if enabled is None else enabled
        self._converter: Any = None
        self._chunker: Any = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _build(self) -> tuple[Any, Any]:
        # docling.chunking re-exports this without __all__, so import from its defining module.
        from docling.document_converter import DocumentConverter
        from docling_core.transforms.chunker.hybrid_chunker import HybridChunker

        return DocumentConverter(), HybridChunker()

    async def startup(self) -> None:
        """Import Docling and build the converter and chunker. Idempotent.

        A missing package disables the engine rather than failing startup: text ingest still
        works through the tiktoken fallback.
        """
        if not self._enabled or self._converter is not None:
            return
        try:
            self._converter, self._chunker = await asyncio.to_thread(self._build)
        except ImportError:
            self._enabled = False
            log.warning(
                "docling is not installed — text falls back to tiktoken chunking and binary "
                "ingest will fail. Install with: uv sync --extra docling"
            )
            return
        log.info("engram: docling ready (in-process)")

    async def shutdown(self) -> None:
        """Release the converter and chunker."""
        self._converter = None
        self._chunker = None

    async def health(self) -> bool:
        """True when Docling is enabled and loaded."""
        return self._enabled and self._converter is not None

    def _require(self) -> None:
        if not self._enabled:
            raise DoclingUnavailable("Docling is disabled or not installed (uv sync --extra docling)")
        if self._converter is None:
            raise DoclingUnavailable("Docling has not been started")

    def _convert(self, data: bytes, filename: str) -> Any:
        from docling_core.types.io import DocumentStream

        try:
            result = self._converter.convert(DocumentStream(name=filename, stream=BytesIO(data)))
        except Exception as exc:  # docling raises a family of conversion errors
            raise DoclingFailed(f"Docling could not convert {filename!r}: {exc}") from exc
        return result.document

    def _chunk(self, document: Any) -> list[dict[str, Any]]:
        try:
            chunks = list(self._chunker.chunk(dl_doc=document))
        except Exception as exc:
            raise DoclingFailed(f"Docling could not chunk the document: {exc}") from exc
        return [{"text": self._chunker.contextualize(chunk=c) or c.text} for c in chunks]

    async def chunk_hybrid_file(self, file_bytes: bytes, filename: str) -> list[dict[str, Any]]:
        """Convert *file_bytes* and return hybrid chunks as ``[{"text": ...}]``."""
        self._require()

        def _run() -> list[dict[str, Any]]:
            return self._chunk(self._convert(file_bytes, filename))

        return await asyncio.to_thread(_run)

    async def chunk_text_hybrid(
        self, text: str, filename: str = "input.md"
    ) -> list[dict[str, Any]]:
        """Chunk Markdown/plain text through the same pipeline."""
        self._require()

        def _run() -> list[dict[str, Any]]:
            return self._chunk(self._convert(text.encode("utf-8"), filename))

        return await asyncio.to_thread(_run)

    async def convert_file_to_markdown(self, file_bytes: bytes, filename: str) -> str:
        """Convert *file_bytes* to Markdown."""
        self._require()

        def _run() -> str:
            return str(self._convert(file_bytes, filename).export_to_markdown())

        return await asyncio.to_thread(_run)
