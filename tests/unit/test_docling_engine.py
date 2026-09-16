"""DoclingEngine — availability, fallbacks, and (behind the `docling` marker) real conversion."""

from __future__ import annotations

import builtins
from collections.abc import Iterator
from pathlib import Path

import pytest

from engram.clients.docling import DoclingEngine, DoclingFailed, DoclingUnavailable
from engram.processors.base import ChunkerKind
from engram.processors.docling_file import DoclingFileProcessor
from engram.processors.docling_text import DoclingTextProcessor

_FIXTURE_PDF = Path(__file__).parents[1] / "integration" / "fixtures" / "sample.pdf"


@pytest.fixture
def hide_docling(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Make `import docling` fail, as it does without the optional extra."""
    real_import = builtins.__import__

    def _fake(name: str, *args: object, **kwargs: object) -> object:
        if name.startswith("docling"):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", _fake)
    yield


# ── availability ──────────────────────────────────────────────────────────


async def test_disabled_engine_raises_unavailable() -> None:
    engine = DoclingEngine(enabled=False)
    await engine.startup()
    assert engine.enabled is False
    assert await engine.health() is False
    with pytest.raises(DoclingUnavailable):
        await engine.chunk_text_hybrid("hello")


async def test_missing_package_disables_instead_of_failing_startup(hide_docling: None) -> None:
    engine = DoclingEngine(enabled=True)
    await engine.startup()  # must not raise
    assert engine.enabled is False
    with pytest.raises(DoclingUnavailable):
        await engine.chunk_hybrid_file(b"%PDF-1.4", "x.pdf")


async def test_calls_before_startup_are_unavailable() -> None:
    engine = DoclingEngine(enabled=True)
    with pytest.raises(DoclingUnavailable):
        await engine.chunk_text_hybrid("hello")


async def test_shutdown_makes_the_engine_unhealthy_again() -> None:
    engine = DoclingEngine(enabled=True)
    engine._converter, engine._chunker, engine._stream = object(), object(), object()
    assert await engine.health() is True
    await engine.shutdown()
    assert await engine.health() is False


async def test_conversion_errors_surface_as_docling_failed(hide_docling: None) -> None:
    """A failed conversion raises DoclingFailed — and does so without importing Docling,
    which is what proves the engine touches the package only at startup."""

    class _Boom:
        def convert(self, *_: object, **__: object) -> object:
            raise RuntimeError("bad pdf")

    class _Stream:
        def __init__(self, **_: object) -> None: ...

    engine = DoclingEngine(enabled=True)
    engine._converter, engine._chunker, engine._stream = _Boom(), object(), _Stream
    with pytest.raises(DoclingFailed, match="could not convert"):
        await engine.chunk_hybrid_file(b"not a pdf", "x.pdf")


# ── processor fallbacks ───────────────────────────────────────────────────


async def test_text_processor_falls_back_to_tiktoken_without_docling() -> None:
    engine = DoclingEngine(enabled=False)
    await engine.startup()
    chunks = await DoclingTextProcessor(engine).process("some text to chunk")
    assert chunks and chunks[0].chunker == ChunkerKind.TIKTOKEN_FALLBACK


async def test_file_processor_has_no_fallback_without_docling() -> None:
    engine = DoclingEngine(enabled=False)
    await engine.startup()
    with pytest.raises(DoclingUnavailable):
        await DoclingFileProcessor(engine).process(b"%PDF-1.4", "sample.pdf")


# ── real Docling (opt-in: `uv sync --extra docling && pytest -m docling`) ──


@pytest.mark.docling
async def test_real_docling_chunks_markdown() -> None:
    engine = DoclingEngine(enabled=True)
    await engine.startup()
    assert await engine.health() is True, "install the extra: uv sync --extra docling"

    chunks = await engine.chunk_text_hybrid(
        "# Title\n\nFirst paragraph.\n\n## Second\n\nMore text."
    )
    assert chunks and all(c["text"].strip() for c in chunks)
    await engine.shutdown()


@pytest.mark.docling
async def test_real_docling_converts_a_pdf() -> None:
    engine = DoclingEngine(enabled=True)
    await engine.startup()

    data = _FIXTURE_PDF.read_bytes()
    markdown = await engine.convert_file_to_markdown(data, "sample.pdf")
    chunks = await engine.chunk_hybrid_file(data, "sample.pdf")

    assert markdown.strip()
    assert chunks and all(c["text"].strip() for c in chunks)
    await engine.shutdown()
