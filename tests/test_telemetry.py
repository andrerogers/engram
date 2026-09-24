"""Engram's spans and metrics: a slow recall names its slow part, and the calls happen.

Before these, Engram had ``instrument_fastapi`` and nothing more, so a recall was one HTTP
duration with the embedding call, the vector search and the BM25 search indistinguishable
inside it. ``llm.token_usage`` in Optics sat defined and never called for months, and no test
could see it, so these tests assert each record call is *made*, from the real code path: the app,
a real SQLite store, and ``embed()`` itself, with only OpenRouter replaced at the HTTP transport.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

import engram.app as app_module
from engram import embeddings
from engram.app import app
from engram.clients.storage.memory import InMemoryObjectStore
from engram.jobs.ingest import run_ingest_job
from engram.processors.base import ChunkCandidate
from engram.store import Store

_METRICS = "optics._metrics.EngramMetrics"


@pytest.fixture()
def spans() -> Iterator[InMemorySpanExporter]:
    # app.py ran setup_optics() at import, so there is an SDK provider to join.
    provider = trace.get_tracer_provider()
    assert isinstance(provider, TracerProvider)
    exporter = InMemorySpanExporter()
    processor = SimpleSpanProcessor(exporter)
    provider.add_span_processor(processor)
    yield exporter
    processor.shutdown()


def _openrouter(status: int = 200) -> httpx.AsyncClient:
    """OpenRouter's embeddings endpoint, answering every text with the same unit vector."""

    def handle(request: httpx.Request) -> httpx.Response:
        if status != 200:
            return httpx.Response(status, json={"error": "no"})
        texts = json.loads(request.content)["input"]
        vector = [1.0] + [0.0] * 1535
        return httpx.Response(
            200, json={"data": [{"index": i, "embedding": vector} for i in range(len(texts))]}
        )

    return httpx.AsyncClient(transport=httpx.MockTransport(handle))


@pytest.fixture()
def engram(tmp_path: Path) -> Iterator[TestClient]:
    """The app over a real store. Not entered as a context manager: startup loads Docling."""
    store = Store(tmp_path / "engram.db")
    asyncio.run(store.init_db())
    with (
        patch.object(app_module, "_store", store),
        patch.object(embeddings, "OPENROUTER_API_KEY", "test-key"),
        patch.object(embeddings, "_get_client", return_value=_openrouter()),
    ):
        yield TestClient(app)


def _children_of(parent: ReadableSpan, finished: list[ReadableSpan]) -> set[str]:
    return {
        s.name
        for s in finished
        if s.parent is not None and s.parent.span_id == parent.context.span_id
    }


def _recall_span(finished: list[ReadableSpan], kind: str) -> ReadableSpan:
    return next(
        s
        for s in finished
        if s.name == "engram.recall" and (s.attributes or {})["engram.recall.kind"] == kind
    )


def test_a_fact_recall_names_its_embedding_and_both_halves_of_the_search(
    engram: TestClient, spans: InMemorySpanExporter
) -> None:
    engram.post("/facts", json={"workspace_id": "w", "content": "deploys go out on fridays"})
    spans.clear()

    r = engram.get("/facts/recall", params={"workspace_id": "w", "q": "when do deploys go out"})

    assert r.status_code == 200 and r.json()
    finished = list(spans.get_finished_spans())
    recall = _recall_span(finished, "facts")
    assert _children_of(recall, finished) == {
        "engram.embed",
        "engram.search.vector",
        "engram.search.bm25",
    }


def test_a_chunk_retrieval_names_its_parts_and_indexing_counts_the_chunks(
    engram: TestClient, spans: InMemorySpanExporter
) -> None:
    with patch(f"{_METRICS}.record_indexed") as indexed:
        index = engram.post(
            "/index",
            json={"documents": [{"path": "a.md", "content": "route_web_socket opens a socket"}]},
        ).json()
    indexed.assert_called_once_with(chunks=index["chunk_count"])
    spans.clear()

    r = engram.get(
        "/retrieve", params={"q": "route_web_socket", "collection_id": index["collection_id"]}
    )

    assert r.status_code == 200 and r.json()["results"]
    finished = list(spans.get_finished_spans())
    assert _children_of(_recall_span(finished, "chunks"), finished) == {
        "engram.embed",
        "engram.search.vector",
        "engram.search.bm25",
    }


def test_every_recall_records_its_latency_and_its_embedding(engram: TestClient) -> None:
    with (
        patch(f"{_METRICS}.record_recall") as recall,
        patch(f"{_METRICS}.record_embedding") as embedding,
    ):
        engram.get("/facts/recall", params={"workspace_id": "w", "q": "anything"})
        engram.get("/signals/recall", params={"workspace_id": "w", "q": "anything"})

    assert [c.kwargs["kind"] for c in recall.call_args_list] == ["facts", "signals"]
    assert all(c.kwargs["duration_s"] > 0 for c in recall.call_args_list)
    assert [c.kwargs["outcome"] for c in embedding.call_args_list] == ["ok", "ok"]


def test_a_failed_embedding_is_counted_and_fails_the_recall_span(
    engram: TestClient, spans: InMemorySpanExporter
) -> None:
    with (
        patch.object(embeddings, "_get_client", return_value=_openrouter(status=400)),
        patch(f"{_METRICS}.record_embedding") as embedding,
    ):
        r = engram.get("/facts/recall", params={"workspace_id": "w", "q": "anything"})

    assert r.status_code == 503
    embedding.assert_called_once()
    assert embedding.call_args.kwargs["outcome"] == "error"
    finished = list(spans.get_finished_spans())
    assert _recall_span(finished, "facts").status.status_code is StatusCode.ERROR
    assert next(s for s in finished if s.name == "engram.embed").status.status_code is (
        StatusCode.ERROR
    )


def test_a_broken_metric_never_breaks_a_recall(engram: TestClient) -> None:
    with patch(f"{_METRICS}.record_recall", side_effect=RuntimeError("otel down")):
        r = engram.get("/facts/recall", params={"workspace_id": "w", "q": "anything"})
    assert r.status_code == 200


class _Chunks:
    """A file processor that yields one chunk — the processor is not what is measured here."""

    async def process(self, data: bytes, filename: str) -> list[ChunkCandidate]:
        return [ChunkCandidate(content=data.decode(), chunk_index=0)]


async def _ingest(tmp_path: Path, *, stored: bool) -> MagicMock:
    store = Store(tmp_path / "jobs.db")
    await store.init_db()
    objects = InMemoryObjectStore()
    if stored:
        await objects.put("k", b"a document", "text/plain")
    collection = await store.get_or_create_collection("w", "c")
    job = await store.create_ingest_job(collection, filename="a.txt", object_key="k")
    with (
        patch.object(embeddings, "OPENROUTER_API_KEY", "test-key"),
        patch.object(embeddings, "_get_client", return_value=_openrouter()),
        patch(f"{_METRICS}.record_ingest") as ingest,
    ):
        await run_ingest_job(job, store, objects, _Chunks())  # type: ignore[arg-type]
    await store.close()
    return ingest


async def test_an_ingest_job_records_its_outcome(tmp_path: Path) -> None:
    completed: Any = await _ingest(tmp_path / "ok", stored=True)
    completed.assert_called_once_with(outcome="completed")

    # The object is missing, so the job fails — a number now, not only a log line.
    failed: Any = await _ingest(tmp_path / "missing", stored=False)
    failed.assert_called_once_with(outcome="failed")
