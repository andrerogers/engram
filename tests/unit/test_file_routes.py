"""Tests for file ingestion routes — POST /index/file, GET /index/file/status/{job_id},
GET /documents/{id}/original, GET /documents/_object/{key:path}."""

from __future__ import annotations

import hashlib
import io
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from engram.app import app

client = TestClient(app)

# Patches
_STORE = "engram.app._store"
_OBJECT_STORE = "engram.app._object_store"
_RUNNER = "engram.app._runner"
_FILE_PROCESSOR = "engram.app._file_processor"

_FAKE_PDF = b"%PDF-1.4 fake content"
_FILE_HASH = hashlib.sha256(_FAKE_PDF).hexdigest()
_COLLECTION_ID = "coll-1"
_JOB_ID = "job-abc"
_DOC_ID = "doc-xyz"
_OBJECT_KEY = f"uploads/{_COLLECTION_ID}/{_FILE_HASH[:12]}-sample.pdf"


def _mock_store(existing_doc: str | None = None) -> AsyncMock:
    store = AsyncMock()
    store.get_or_create_collection = AsyncMock(return_value=_COLLECTION_ID)
    store.find_document_by_hash = AsyncMock(return_value=existing_doc)
    store.create_ingest_job = AsyncMock(return_value=_JOB_ID)
    store.get_ingest_job = AsyncMock(
        return_value={
            "id": _JOB_ID,
            "collection_id": _COLLECTION_ID,
            "status": "pending",
            "filename": "sample.pdf",
            "object_key": _OBJECT_KEY,
            "document_id": None,
            "error_message": None,
            "last_heartbeat": None,
            "created_at": "2026-04-16T12:00:00+00:00",
            "updated_at": "2026-04-16T12:00:00+00:00",
        }
    )
    store.get_document = AsyncMock(
        return_value={
            "id": _DOC_ID,
            "collection_id": _COLLECTION_ID,
            "path": "sample.pdf",
            "metadata": {},
            "object_key": _OBJECT_KEY,
            "created_at": "2026-04-16T12:00:00+00:00",
        }
    )
    return store


def _mock_object_store(data: bytes = _FAKE_PDF) -> AsyncMock:
    obj = AsyncMock()
    obj.put = AsyncMock()
    obj.get = AsyncMock(return_value=data)
    obj.presigned_url = AsyncMock(return_value=f"/documents/_object/{_OBJECT_KEY}")
    return obj


def _mock_runner() -> MagicMock:
    runner = MagicMock()
    runner.schedule = MagicMock()
    return runner


# ---------------------------------------------------------------------------
# POST /index/file — happy path
# ---------------------------------------------------------------------------


def test_index_file_returns_202() -> None:
    store = _mock_store()
    obj = _mock_object_store()
    runner = _mock_runner()

    with patch(_STORE, store), patch(_OBJECT_STORE, obj), patch(_RUNNER, runner):
        r = client.post(
            "/index/file",
            files={"file": ("sample.pdf", io.BytesIO(_FAKE_PDF), "application/pdf")},
            params={"collection_id": _COLLECTION_ID},
        )

    assert r.status_code == 202
    body = r.json()
    assert body["status"] == "accepted"
    assert body["job_id"] == _JOB_ID


def test_index_file_uploads_to_object_store() -> None:
    store = _mock_store()
    obj = _mock_object_store()
    runner = _mock_runner()

    with patch(_STORE, store), patch(_OBJECT_STORE, obj), patch(_RUNNER, runner):
        client.post(
            "/index/file",
            files={"file": ("sample.pdf", io.BytesIO(_FAKE_PDF), "application/pdf")},
            params={"collection_id": _COLLECTION_ID},
        )

    obj.put.assert_awaited_once()
    call_args = obj.put.call_args
    assert call_args.args[1] == _FAKE_PDF


def test_index_file_schedules_runner_job() -> None:
    store = _mock_store()
    obj = _mock_object_store()
    runner = _mock_runner()

    with patch(_STORE, store), patch(_OBJECT_STORE, obj), patch(_RUNNER, runner):
        client.post(
            "/index/file",
            files={"file": ("sample.pdf", io.BytesIO(_FAKE_PDF), "application/pdf")},
            params={"collection_id": _COLLECTION_ID},
        )

    runner.schedule.assert_called_once()
    assert runner.schedule.call_args.args[0] == _JOB_ID


def test_index_file_creates_collection_when_not_given() -> None:
    store = _mock_store()
    obj = _mock_object_store()
    runner = _mock_runner()

    with patch(_STORE, store), patch(_OBJECT_STORE, obj), patch(_RUNNER, runner):
        client.post(
            "/index/file",
            files={"file": ("f.pdf", io.BytesIO(_FAKE_PDF), "application/pdf")},
            params={"workspace_id": "ws-1", "collection_name": "docs"},
        )

    store.get_or_create_collection.assert_awaited_once_with("ws-1", "docs")


# ---------------------------------------------------------------------------
# POST /index/file — deduplication
# ---------------------------------------------------------------------------


def test_index_file_duplicate_returns_200() -> None:
    store = _mock_store(existing_doc=_DOC_ID)
    obj = _mock_object_store()
    runner = _mock_runner()

    with patch(_STORE, store), patch(_OBJECT_STORE, obj), patch(_RUNNER, runner):
        r = client.post(
            "/index/file",
            files={"file": ("sample.pdf", io.BytesIO(_FAKE_PDF), "application/pdf")},
            params={"collection_id": _COLLECTION_ID},
        )

    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "duplicate"
    assert body["document_id"] == _DOC_ID


def test_index_file_duplicate_does_not_upload() -> None:
    store = _mock_store(existing_doc=_DOC_ID)
    obj = _mock_object_store()
    runner = _mock_runner()

    with patch(_STORE, store), patch(_OBJECT_STORE, obj), patch(_RUNNER, runner):
        client.post(
            "/index/file",
            files={"file": ("sample.pdf", io.BytesIO(_FAKE_PDF), "application/pdf")},
            params={"collection_id": _COLLECTION_ID},
        )

    obj.put.assert_not_awaited()
    runner.schedule.assert_not_called()


# ---------------------------------------------------------------------------
# POST /index/file — oversize rejection
# ---------------------------------------------------------------------------


def test_index_file_oversize_returns_413() -> None:
    store = _mock_store()
    obj = _mock_object_store()
    runner = _mock_runner()

    big_file = b"x" * (51 * 1024 * 1024)  # 51 MB

    with (
        patch(_STORE, store),
        patch(_OBJECT_STORE, obj),
        patch(_RUNNER, runner),
        patch("engram.app.MAX_FILE_SIZE_MB", 50),
    ):
        r = client.post(
            "/index/file",
            files={"file": ("big.bin", io.BytesIO(big_file), "application/octet-stream")},
            params={"collection_id": _COLLECTION_ID},
        )

    assert r.status_code == 413


# ---------------------------------------------------------------------------
# GET /index/file/status/{job_id}
# ---------------------------------------------------------------------------


def test_ingest_job_status_returns_job() -> None:
    store = _mock_store()

    with patch(_STORE, store):
        r = client.get(f"/index/file/status/{_JOB_ID}")

    assert r.status_code == 200
    body = r.json()
    assert body["id"] == _JOB_ID
    assert body["status"] == "pending"
    assert body["filename"] == "sample.pdf"


def test_ingest_job_status_404_for_missing_job() -> None:
    store = _mock_store()
    store.get_ingest_job = AsyncMock(return_value=None)

    with patch(_STORE, store):
        r = client.get("/index/file/status/nonexistent")

    assert r.status_code == 404


# ---------------------------------------------------------------------------
# GET /documents/{id}/original
# ---------------------------------------------------------------------------


def test_document_original_redirects() -> None:
    store = _mock_store()
    obj = _mock_object_store()

    with patch(_STORE, store), patch(_OBJECT_STORE, obj):
        r = client.get(f"/documents/{_DOC_ID}/original", follow_redirects=False)

    assert r.status_code == 302
    assert _OBJECT_KEY in r.headers["location"]


def test_document_original_404_for_missing_document() -> None:
    store = _mock_store()
    store.get_document = AsyncMock(return_value=None)
    obj = _mock_object_store()

    with patch(_STORE, store), patch(_OBJECT_STORE, obj):
        r = client.get("/documents/nonexistent/original")

    assert r.status_code == 404


def test_document_original_404_when_no_object_key() -> None:
    store = _mock_store()
    store.get_document = AsyncMock(
        return_value={
            "id": _DOC_ID,
            "collection_id": _COLLECTION_ID,
            "path": "inline.txt",
            "metadata": {},
            "object_key": None,
            "created_at": "2026-04-16T12:00:00+00:00",
        }
    )
    obj = _mock_object_store()

    with patch(_STORE, store), patch(_OBJECT_STORE, obj):
        r = client.get(f"/documents/{_DOC_ID}/original")

    assert r.status_code == 404


# ---------------------------------------------------------------------------
# GET /documents/_object/{key:path}  (InMemory read-through)
# ---------------------------------------------------------------------------


def test_object_passthrough_returns_bytes() -> None:
    obj = _mock_object_store()

    with patch(_OBJECT_STORE, obj), patch("engram.app.MINIO_ENABLED", False):
        r = client.get(f"/documents/_object/{_OBJECT_KEY}")

    assert r.status_code == 200
    assert r.content == _FAKE_PDF


def test_object_passthrough_404_for_missing_key() -> None:
    obj = _mock_object_store()
    obj.get = AsyncMock(side_effect=KeyError("not found"))

    with patch(_OBJECT_STORE, obj), patch("engram.app.MINIO_ENABLED", False):
        r = client.get("/documents/_object/uploads/missing/key.pdf")

    assert r.status_code == 404


def test_object_passthrough_disabled_with_minio() -> None:
    obj = _mock_object_store()

    with patch(_OBJECT_STORE, obj), patch("engram.app.MINIO_ENABLED", True):
        r = client.get(f"/documents/_object/{_OBJECT_KEY}")

    assert r.status_code == 404
