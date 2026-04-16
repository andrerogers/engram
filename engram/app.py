# ruff: noqa: E402
from optics import instrument_fastapi, setup_optics

setup_optics("engram", service_version="0.0.1")

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query

log = logging.getLogger(__name__)
from fastapi.middleware.cors import CORSMiddleware

from engram import embeddings
from engram.clients.docling import DoclingClient
from engram.config import DATABASE_URL, MAX_CONCURRENT_INGEST_JOBS
from engram.jobs.runner import Runner
from engram.models import (
    CollectionOut,
    DocumentIn,
    IndexRequest,
    IndexResponse,
    RetrieveResponse,
    RetrieveResult,
)
from engram.processors import get_text_processor
from engram.store import Store

_docling: DoclingClient = DoclingClient()
_processor = get_text_processor(_docling)
_runner: Runner = Runner(MAX_CONCURRENT_INGEST_JOBS, _docling)

_store: Store | None = None


def _get_store() -> Store:
    if _store is None:
        raise RuntimeError("Store not initialised")
    return _store


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _store  # noqa: PLW0603
    await _docling.startup()
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
) -> RetrieveResponse:
    store = _get_store()
    vecs = await embeddings.embed([q])
    results = await store.retrieve(
        embedding=vecs[0],
        collection_id=collection_id,
        k=k,
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
