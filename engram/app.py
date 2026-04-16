# ruff: noqa: E402
from optics import instrument_fastapi, setup_optics

setup_optics("engram", service_version="0.0.1")

import hashlib
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, UploadFile
from fastapi.responses import RedirectResponse

log = logging.getLogger(__name__)
from fastapi.middleware.cors import CORSMiddleware

from engram import embeddings
from engram.clients.docling import DoclingClient
from engram.clients.storage import InMemoryObjectStore, ObjectStore
from engram.config import (
    DATABASE_URL,
    MAX_CONCURRENT_INGEST_JOBS,
    MAX_FILE_SIZE_MB,
    MINIO_ACCESS_KEY,
    MINIO_BUCKET,
    MINIO_ENABLED,
    MINIO_ENDPOINT,
    MINIO_SECRET_KEY,
)
from engram.jobs.runner import Runner
from engram.models import (
    CollectionOut,
    DocumentIn,
    DocumentOut,
    FileIngestResponse,
    IndexRequest,
    IndexResponse,
    IngestJobOut,
    RetrieveResponse,
    RetrieveResult,
)
from engram.processors import get_file_processor, get_text_processor
from engram.store import Store


def _make_object_store() -> ObjectStore:
    if MINIO_ENABLED:
        from engram.clients.storage.minio import MinioObjectStore

        return MinioObjectStore(
            endpoint=MINIO_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            bucket=MINIO_BUCKET,
        )
    return InMemoryObjectStore()


_docling: DoclingClient = DoclingClient()
_processor = get_text_processor(_docling)
_file_processor = get_file_processor(_docling)
_runner: Runner = Runner(MAX_CONCURRENT_INGEST_JOBS, _docling)
_object_store: ObjectStore = _make_object_store()

_store: Store | None = None


def _get_store() -> Store:
    if _store is None:
        raise RuntimeError("Store not initialised")
    return _store


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _store  # noqa: PLW0603
    await _docling.startup()
    await _object_store.startup()
    if DATABASE_URL:
        _store = Store(DATABASE_URL)
        await _store.init_db()
        await _runner.startup(_store)
        log.info("engram started — store initialised")
    else:
        log.warning("engram started — no DATABASE_URL, store unavailable")
    yield
    await _runner.shutdown_wait()
    if _store is not None:
        await _store.close()
        log.info("engram shutting down — store closed")
    await _object_store.shutdown()
    await _docling.shutdown()


app = FastAPI(title="Engram", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174"],
    allow_methods=["*"],
    allow_headers=["*"],
)

instrument_fastapi(app)


# ── Health ────────────────────────────────────────────────────────────────


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "engram"}


# ── Indexing ──────────────────────────────────────────────────────────────


async def _index_documents(
    store: Store,
    collection_id: str,
    documents: list[DocumentIn],
) -> tuple[int, int]:
    """Chunk, embed, and store documents. Returns (doc_count, chunk_count)."""
    total_chunks = 0
    for doc in documents:
        candidates = await _processor.process(doc.content)
        if not candidates:
            continue
        vecs = await embeddings.embed([c.content for c in candidates])
        _, n = await store.index_document(
            collection_id=collection_id,
            path=doc.path,
            metadata=doc.metadata,
            candidates=candidates,
            embeddings=vecs,
        )
        total_chunks += n
    return len(documents), total_chunks


@app.post("/index", response_model=IndexResponse)
async def index_documents(req: IndexRequest) -> IndexResponse:
    store = _get_store()
    workspace_id = req.workspace_id or "default"
    collection_name = req.collection_name or "default"

    if req.collection_id:
        cid = req.collection_id
    else:
        cid = await store.get_or_create_collection(workspace_id, collection_name)

    doc_count, chunk_count = await _index_documents(store, cid, req.documents)
    log.info("indexed collection=%s docs=%d chunks=%d", cid, doc_count, chunk_count)
    return IndexResponse(
        indexed_count=doc_count,
        collection_id=cid,
        chunk_count=chunk_count,
    )


# ── Retrieval ─────────────────────────────────────────────────────────────


@app.get("/retrieve", response_model=RetrieveResponse)
async def retrieve(
    q: str = Query(..., description="Search query"),
    collection_id: str = Query(..., description="Collection to search"),
    k: int = Query(default=5, ge=1, le=100),
    modalities: list[str] | None = Query(
        default=None, description="Modality filter (default: text)"
    ),
) -> RetrieveResponse:
    store = _get_store()
    vecs = await embeddings.embed([q])
    results = await store.retrieve(
        embedding=vecs[0],
        collection_id=collection_id,
        k=k,
        modalities=modalities,
    )
    return RetrieveResponse(results=[RetrieveResult(**r) for r in results])


# ── Collections ───────────────────────────────────────────────────────────


@app.get("/collections", response_model=list[CollectionOut])
async def list_collections(
    workspace_id: str | None = Query(default=None),
) -> list[CollectionOut]:
    store = _get_store()
    rows = await store.list_collections(workspace_id)
    return [CollectionOut(**r) for r in rows]


@app.delete("/collections/{collection_id}", status_code=204)
async def delete_collection(collection_id: str) -> None:
    store = _get_store()
    deleted = await store.delete_collection(collection_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Collection not found")


# ── File ingestion ────────────────────────────────────────────────────────


def _object_key(collection_id: str, file_hash: str, filename: str) -> str:
    return f"uploads/{collection_id}/{file_hash[:12]}-{filename}"


@app.post("/index/file")
async def index_file(
    file: UploadFile,
    collection_id: str | None = Query(default=None),
    workspace_id: str | None = Query(default=None),
    collection_name: str | None = Query(default=None),
) -> FileIngestResponse:
    """Upload a binary file for async ingestion.

    Returns 202 (accepted) with job_id on new files.
    Returns 200 (duplicate) with document_id when the same bytes already exist
    in the collection.
    """
    store = _get_store()

    data = await file.read()
    max_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds MAX_FILE_SIZE_MB ({MAX_FILE_SIZE_MB} MB)",
        )

    file_hash = hashlib.sha256(data).hexdigest()
    filename = file.filename or "upload"
    content_type = file.content_type or "application/octet-stream"

    ws_id = workspace_id or "default"
    col_name = collection_name or "default"
    cid = collection_id or await store.get_or_create_collection(ws_id, col_name)

    existing_doc_id = await store.find_document_by_hash(cid, file_hash)
    if existing_doc_id:
        log.info("index_file: duplicate detected for collection=%s hash=%s", cid, file_hash[:12])
        return FileIngestResponse(status="duplicate", document_id=existing_doc_id)

    key = _object_key(cid, file_hash, filename)
    await _object_store.put(key, data, content_type=content_type)

    job_id = await store.create_ingest_job(
        collection_id=cid,
        filename=filename,
        object_key=key,
    )
    _runner.schedule(job_id, store, _object_store, _file_processor)

    log.info("index_file: accepted job=%s collection=%s file=%s", job_id, cid, filename)
    from fastapi.responses import JSONResponse

    return JSONResponse(
        status_code=202,
        content=FileIngestResponse(status="accepted", job_id=job_id).model_dump(),
    )


@app.get("/index/file/status/{job_id}", response_model=IngestJobOut)
async def ingest_job_status(job_id: str) -> IngestJobOut:
    """Return the current status of an ingest job."""
    store = _get_store()
    job = await store.get_ingest_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return IngestJobOut(**job)


# ── Document management ───────────────────────────────────────────────────


@app.get("/documents", response_model=list[DocumentOut])
async def list_documents(
    collection_id: str = Query(..., description="Collection to list documents from"),
) -> list[DocumentOut]:
    store = _get_store()
    rows = await store.list_documents(collection_id)
    return [DocumentOut(**r) for r in rows]


@app.delete("/documents/{document_id}", status_code=204)
async def delete_document(document_id: str) -> None:
    """Delete a document, its chunks (via FK cascade), and original object if present."""
    store = _get_store()
    doc = await store.get_document(document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")
    await store.delete_document(document_id)
    object_key = doc.get("object_key")
    if object_key:
        try:
            await _object_store.delete(object_key)
        except KeyError:
            log.warning("delete_document: object already missing key=%s", object_key)


# ── Document access ───────────────────────────────────────────────────────


@app.get("/documents/{document_id}/original")
async def document_original(document_id: str) -> RedirectResponse:
    """Redirect to a presigned URL for the original file."""
    store = _get_store()
    doc = await store.get_document(document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")
    object_key: str | None = doc.get("object_key")
    if not object_key:
        raise HTTPException(status_code=404, detail="No original file for this document")
    url = await _object_store.presigned_url(object_key)
    return RedirectResponse(url=url, status_code=302)


@app.get("/documents/_object/{key:path}")
async def document_object_passthrough(key: str) -> bytes:
    """Read-through route for InMemory object store (config-gated — not for MinIO).

    Returns raw bytes with status 200.  Disabled when MINIO_ENABLED=True —
    use the presigned URL from GET /documents/{id}/original instead.
    """
    if MINIO_ENABLED:
        raise HTTPException(
            status_code=404,
            detail="Read-through not available with MinIO — use /documents/{id}/original",
        )
    try:
        data = await _object_store.get(key)
    except KeyError:
        raise HTTPException(status_code=404, detail="Object not found") from None
    from fastapi.responses import Response

    return Response(content=data, media_type="application/octet-stream")
