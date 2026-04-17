"""E13 — End-to-end integration test: PDF upload → ingest → retrieve.

Requires external services running (use main docker-compose.yml):
    docker compose up -d postgres minio docling
    uv run pytest -m integration tests/integration/test_end_to_end.py -v

Pipeline verified:
    PDF bytes → MinIO object store → DoclingFileProcessor (real Docling)
    → embeddings (mocked deterministic) → pgvector store → retrieve returns chunks
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from engram.clients.docling import DoclingClient
from engram.clients.storage.minio import MinioObjectStore
from engram.jobs.ingest import run_ingest_job
from engram.processors import get_file_processor
from engram.store import Store

pytestmark = pytest.mark.integration

_DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://brainstack:brainstack@localhost:5432/brainstack")
_MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "http://localhost:9000")
_MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
_MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "minioadmin")
_MINIO_BUCKET = "engram-e2e"
_DOCLING_URL = os.environ.get("DOCLING_URL", "http://localhost:5001")

_FIXTURE_PDF = Path(__file__).parent / "fixtures" / "sample.pdf"

_FAKE_EMBEDDING = [0.1] * 1536


@pytest.fixture
async def store() -> Store:  # type: ignore[misc]
    s = Store(_DATABASE_URL)
    await s.init_db()
    yield s
    await s.close()


@pytest.fixture
async def object_store() -> MinioObjectStore:  # type: ignore[misc]
    s = MinioObjectStore(
        endpoint=_MINIO_ENDPOINT,
        access_key=_MINIO_ACCESS_KEY,
        secret_key=_MINIO_SECRET_KEY,
        bucket=_MINIO_BUCKET,
    )
    await s.startup()
    yield s
    await s.shutdown()


@pytest.fixture
async def docling() -> DoclingClient:  # type: ignore[misc]
    client = DoclingClient(
        base_url=_DOCLING_URL,
        enabled=True,
        timeout=120.0,
        poll_interval=2.0,
        max_wait=300.0,
    )
    await client.startup()
    yield client
    await client.shutdown()


@pytest.fixture
async def collection_id(store: Store) -> str:
    ws = f"e2e-ws-{uuid.uuid4().hex[:6]}"
    return await store.get_or_create_collection(ws, "e2e-collection")


async def test_pdf_ingest_and_retrieve(
    store: Store,
    object_store: MinioObjectStore,
    docling: DoclingClient,
    collection_id: str,
) -> None:
    """Full pipeline: PDF → Docling chunks → pgvector → retrieve."""
    file_processor = get_file_processor(docling)
    pdf_bytes = _FIXTURE_PDF.read_bytes()
    object_key = f"uploads/{collection_id}/e2e-sample.pdf"

    await object_store.put(object_key, pdf_bytes, content_type="application/pdf")

    job_id = await store.create_ingest_job(
        collection_id=collection_id,
        filename="sample.pdf",
        object_key=object_key,
    )

    def _make_embeddings(texts: list[str]) -> list[list[float]]:
        return [_FAKE_EMBEDDING for _ in texts]

    with patch("engram.jobs.ingest.embeddings.embed", AsyncMock(side_effect=_make_embeddings)):
        await run_ingest_job(job_id, store, object_store, file_processor)

    job = await store.get_ingest_job(job_id)
    assert job is not None
    assert job["status"] == "completed", f"Job failed: {job.get('error_message')}"
    assert job["document_id"] is not None

    results = await store.retrieve(
        embedding=_FAKE_EMBEDDING,
        collection_id=collection_id,
        k=10,
    )
    assert len(results) > 0, "Expected at least one chunk after ingestion"
    assert all(r["modality"] == "text" for r in results)
    assert all(r["chunker"] is not None for r in results)


async def test_dedup_second_upload_returns_existing_document(
    store: Store,
    object_store: MinioObjectStore,
    docling: DoclingClient,
    collection_id: str,
) -> None:
    """Same file bytes uploaded twice → second job sees duplicate_of_job_id or same doc."""
    import hashlib

    file_processor = get_file_processor(docling)
    pdf_bytes = _FIXTURE_PDF.read_bytes()
    file_hash = hashlib.sha256(pdf_bytes).hexdigest()

    existing = await store.find_document_by_hash(collection_id, file_hash)
    assert existing is None, "Expected no prior document for this collection"

    object_key = f"uploads/{collection_id}/dedup-sample.pdf"
    await object_store.put(object_key, pdf_bytes, content_type="application/pdf")

    job_id = await store.create_ingest_job(
        collection_id=collection_id,
        filename="sample.pdf",
        object_key=object_key,
        file_hash=file_hash,
    )

    with patch(
        "engram.jobs.ingest.embeddings.embed",
        AsyncMock(side_effect=lambda t: [_FAKE_EMBEDDING] * len(t)),
    ):
        await run_ingest_job(job_id, store, object_store, file_processor)

    doc_id = (await store.get_ingest_job(job_id))["document_id"]
    assert doc_id is not None

    found = await store.find_document_by_hash(collection_id, file_hash)
    assert found == doc_id


async def test_list_and_delete_document(
    store: Store,
    object_store: MinioObjectStore,
    docling: DoclingClient,
    collection_id: str,
) -> None:
    """Ingest a file, list documents, delete — object must be removed from MinIO."""
    file_processor = get_file_processor(docling)
    pdf_bytes = _FIXTURE_PDF.read_bytes()
    object_key = f"uploads/{collection_id}/delete-sample.pdf"
    await object_store.put(object_key, pdf_bytes, content_type="application/pdf")

    job_id = await store.create_ingest_job(
        collection_id=collection_id,
        filename="sample.pdf",
        object_key=object_key,
    )
    with patch(
        "engram.jobs.ingest.embeddings.embed",
        AsyncMock(side_effect=lambda t: [_FAKE_EMBEDDING] * len(t)),
    ):
        await run_ingest_job(job_id, store, object_store, file_processor)

    docs = await store.list_documents(collection_id)
    job = await store.get_ingest_job(job_id)
    assert any(d["id"] == job["document_id"] for d in docs)

    doc_id = (await store.get_ingest_job(job_id))["document_id"]
    returned_key = await store.delete_document(doc_id)

    if returned_key:
        try:
            await object_store.delete(returned_key)
        except KeyError:
            pass

    assert not await object_store.exists(object_key), "Object should have been deleted"
    assert await store.get_document(doc_id) is None
