"""Text chunker — fixed-size with overlap, respecting line boundaries.

Uses tiktoken for token counting (cl100k_base, same tokenizer as
text-embedding-3-small).
"""

from __future__ import annotations

import tiktoken

from engram.config import CHUNK_OVERLAP_TOKENS, CHUNK_SIZE_TOKENS

_enc: tiktoken.Encoding | None = None


def _get_encoder() -> tiktoken.Encoding:
    global _enc  # noqa: PLW0603
    if _enc is None:
        _enc = tiktoken.get_encoding("cl100k_base")
    return _enc


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE_TOKENS,
    overlap: int = CHUNK_OVERLAP_TOKENS,
) -> list[str]:
    """Split text into chunks of approximately `chunk_size` tokens.

    Splits at newline boundaries near the target size.  Overlap is applied
    by rewinding `overlap` tokens into the previous chunk's content.
    """
    if not text.strip():
        return []

    enc = _get_encoder()
    lines = text.split("\n")
    chunks: list[str] = []
    current_lines: list[str] = []
    current_tokens = 0

    for line in lines:
        line_tokens = len(enc.encode(line))

        # If a single line exceeds chunk_size, force-split it
        if line_tokens > chunk_size and not current_lines:
            tokens = enc.encode(line)
            for j in range(0, len(tokens), chunk_size):
                chunk_tokens = tokens[j : j + chunk_size]
                chunks.append(enc.decode(chunk_tokens))
            continue

        if current_tokens + line_tokens > chunk_size and current_lines:
            chunks.append("\n".join(current_lines))
            # Overlap: keep last lines that fit within overlap tokens
            overlap_lines: list[str] = []
            overlap_count = 0
            for prev_line in reversed(current_lines):
                lt = len(enc.encode(prev_line))
                if overlap_count + lt > overlap:
                    break
                overlap_lines.insert(0, prev_line)
                overlap_count += lt
            current_lines = overlap_lines
            current_tokens = overlap_count

        current_lines.append(line)
        current_tokens += line_tokens

    if current_lines:
        chunks.append("\n".join(current_lines))

    return chunks
