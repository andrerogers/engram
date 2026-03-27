"""Tests for the text chunker."""

from __future__ import annotations

from engram.chunker import chunk_text


def test_empty_text() -> None:
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_short_text() -> None:
    text = "Hello world\nThis is a test"
    chunks = chunk_text(text, chunk_size=1000)
    assert len(chunks) == 1
    assert "Hello world" in chunks[0]


def test_splits_at_line_boundaries() -> None:
    # Create text with many lines that exceeds chunk_size
    lines = [f"Line {i}: some content here to make each line a bit longer" for i in range(100)]
    text = "\n".join(lines)
    chunks = chunk_text(text, chunk_size=50, overlap=10)
    assert len(chunks) > 1
    # Each chunk should contain complete lines (no mid-line splits for normal lines)
    for chunk in chunks:
        # Lines should not be cut in the middle
        for line in chunk.split("\n"):
            assert line.startswith("Line ") or line == ""


def test_overlap_preserves_context() -> None:
    lines = [f"Line {i}" for i in range(20)]
    text = "\n".join(lines)
    chunks = chunk_text(text, chunk_size=10, overlap=3)
    if len(chunks) >= 2:
        # The last line(s) of chunk N should appear at the start of chunk N+1
        last_lines_of_first = chunks[0].split("\n")[-2:]
        first_lines_of_second = chunks[1].split("\n")[:2]
        # At least one overlapping line
        assert any(line in first_lines_of_second for line in last_lines_of_first)
