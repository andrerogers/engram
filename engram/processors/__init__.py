"""Processor registry — returns the configured text/file processor.

``get_text_processor`` respects the ``DEFAULT_TEXT_CHUNKER`` env var:
- ``"docling-hybrid"`` (default) → ``DoclingTextProcessor`` with tiktoken fallback
- ``"tiktoken"`` → ``TiktokenProcessor`` directly (Docling never called)

``get_file_processor`` always returns ``DoclingFileProcessor``; there is
no tiktoken fallback for binary files.
"""

from __future__ import annotations

from engram.processors.base import FileProcessor, TextProcessor


def get_text_processor(client: object) -> TextProcessor:
    """Return the configured text processor for *client*."""
    from engram.config import DEFAULT_TEXT_CHUNKER
    from engram.processors.docling_text import DoclingTextProcessor
    from engram.processors.tiktoken_processor import TiktokenProcessor

    if DEFAULT_TEXT_CHUNKER == "tiktoken":
        return TiktokenProcessor()
    return DoclingTextProcessor(client)  # type: ignore[arg-type]


def get_file_processor(client: object) -> FileProcessor:
    """Return the configured file processor for *client*."""
    from engram.processors.docling_file import DoclingFileProcessor

    return DoclingFileProcessor(client)  # type: ignore[arg-type]
