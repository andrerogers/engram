# Engram — Docling Verification Record

**Status:** verified 2026-09-15 against `docling 2.127.0` / `docling-core 2.96.0`, running
**in-process** (no docling-serve sidecar). Previous record covered docling-serve 1.16.1 over HTTP;
that client and its polling are gone.

This file records what `engram/clients/docling.py` is written against, verified by running it —
not read from documentation. Update it whenever Docling is upgraded and behaviour changes, or an
assumption below is found to be wrong. The fix goes in `DoclingEngine`; the record goes here.

---

## How to run the gate

```bash
uv sync --extra docling                       # ~1.6 GB (CPU wheels)
uv run pytest -m docling -v                   # engine + end-to-end pipeline
```

`-m docling` is deselected by default (`addopts` in `pyproject.toml`), so neither CI nor a normal
`uv run task test` pays for it.

## Verified API surface

| Call | Verified behaviour |
|---|---|
| `from docling.document_converter import DocumentConverter` | Constructed once at startup; the first conversion otherwise pays model load. |
| `from docling.chunking import HybridChunker` | Default constructor needs no tokenizer argument. |
| `from docling_core.types.io import DocumentStream` | `DocumentStream(name=<filename>, stream=BytesIO(data))` is how bytes are converted without touching disk. The **name matters**: the extension selects the backend. |
| `converter.convert(source)` | Returns a result whose `.document` is the `DoclingDocument`. Raises on unsupported or corrupt input — wrapped as `DoclingFailed`. |
| `chunker.chunk(dl_doc=document)` | Returns an iterator; the keyword is `dl_doc`, not a positional document. |
| `chunker.contextualize(chunk=c)` | Returns the chunk text enriched with its heading context. Used in preference to `c.text`, falling back to it when empty. |
| `document.export_to_markdown()` | Markdown for the whole document. |

## Behaviour worth knowing

- **Synchronous and CPU-bound.** Every call runs in a worker thread (`asyncio.to_thread`);
  nothing in the event loop blocks on conversion.
- **First run downloads OCR models** (RapidOCR, ~21 MB) into the virtualenv's `site-packages`,
  from an external host. A first conversion on a machine with no network will fail; subsequent
  ones are local. Conversions in the gate take ~8 s for the sample PDF after warm-up.
- **CPU wheels are pinned deliberately.** `pyproject.toml` lists `torch` and `torchvision`
  explicitly in the `docling` extra and points them at PyTorch's CPU index. Without that, uv
  resolves CUDA builds (~5 GB, plus triton and nvidia-* packages). uv's `tool.uv.sources` only
  binds a project's **own** dependencies, so listing them transitively through `docling` is not
  enough — that mismatch produced `RuntimeError: operator torchvision::nms does not exist`,
  because torch came from the CPU index and torchvision from PyPI.
- **Absence is a supported state.** Without the extra, `DoclingEngine.startup()` logs a warning
  and disables itself: text chunking falls back to tiktoken, binary ingest raises
  `DoclingUnavailable` naming the install command. Covered by tests that fake the missing import.
