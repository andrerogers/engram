"""E7 — Verification gate: DoclingClient against real Docling-serve.

Run with:
    docker compose -f compose.test.yml up -d docling
    uv run pytest -m integration tests/integration/test_docling_real.py -v

All verified endpoint paths, field names, and response shapes are recorded in
VERIFICATION.md at the engram repo root. Update VERIFICATION.md if any test
fails due to an API shape mismatch — the fix goes in DoclingClient, the record
goes in VERIFICATION.md.

DO NOT skip this gate. Downstream code (E8 processors, E9 job runner) is built
on top of a verified DoclingClient, not an assumed one.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from engram.clients.docling import DoclingClient, DoclingTaskFailed

pytestmark = pytest.mark.integration

_FIXTURES = Path(__file__).parent / "fixtures"
_DOCLING_URL = "http://localhost:15001"


@pytest.fixture
async def docling() -> DoclingClient:  # type: ignore[misc]
    """Real DoclingClient pointed at compose.test.yml Docling on port 15001."""
    client = DoclingClient(
        base_url=_DOCLING_URL,
        enabled=True,
        timeout=120.0,
        poll_interval=2.0,
        max_wait=300.0,
    )
    await client.startup()
    yield client  # type: ignore[misc]
    await client.shutdown()


@pytest.mark.integration
class TestDoclingClientReal:
    async def test_health(self, docling: DoclingClient) -> None:
        """Docling /health must respond 200."""
        assert await docling.health() is True

    async def test_convert_pdf_to_markdown(self, docling: DoclingClient) -> None:
        """PDF → Markdown conversion must return non-empty text containing fixture content."""
        pdf_bytes = (_FIXTURES / "sample.pdf").read_bytes()
        markdown = await docling.convert_file_to_markdown(
            file_bytes=pdf_bytes,
            filename="sample.pdf",
        )
        assert isinstance(markdown, str)
        assert len(markdown) > 0
        assert "Engram test document" in markdown

    async def test_chunk_hybrid_pdf(self, docling: DoclingClient) -> None:
        """Hybrid chunking of a PDF must return a list of dicts each with a 'text' field."""
        pdf_bytes = (_FIXTURES / "sample.pdf").read_bytes()
        chunks = await docling.chunk_hybrid_file(
            file_bytes=pdf_bytes,
            filename="sample.pdf",
        )
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        for chunk in chunks:
            assert "text" in chunk, f"Chunk missing 'text' field: {chunk}"
            assert isinstance(chunk["text"], str)
            assert chunk["text"].strip()

    async def test_chunk_hybrid_markdown_text(self, docling: DoclingClient) -> None:
        """Hybrid chunking of a Markdown text input must preserve section content."""
        md_text = (_FIXTURES / "sample.md").read_text()
        chunks = await docling.chunk_text_hybrid(text=md_text, filename="sample.md")
        assert len(chunks) > 0
        combined = " ".join(c["text"] for c in chunks)
        assert "Section One" in combined
        assert "Section Two" in combined

    async def test_hierarchical_chunking(self, docling: DoclingClient) -> None:
        """Hierarchical chunking must return at least one chunk."""
        pdf_bytes = (_FIXTURES / "sample.pdf").read_bytes()
        chunks = await docling.chunk_hierarchical_file(
            file_bytes=pdf_bytes,
            filename="sample.pdf",
        )
        assert len(chunks) > 0

    async def test_corrupted_file_raises_task_failed(self, docling: DoclingClient) -> None:
        """Submitting garbage bytes as a PDF must raise DoclingTaskFailed."""
        with pytest.raises(DoclingTaskFailed):
            await docling.convert_file_to_markdown(
                file_bytes=b"this is not a real pdf file",
                filename="bogus.pdf",
            )

    async def test_clear_results_does_not_raise(self, docling: DoclingClient) -> None:
        """clear_results() must complete without raising (best-effort maintenance)."""
        await docling.clear_results(older_than_seconds=0)
