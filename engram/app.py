# ruff: noqa: E402
from optics import instrument_fastapi, setup_optics

setup_optics("engram", service_version="0.0.1")

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from engram import embeddings
from engram.chunker import chunk_text
from engram.config import DATABASE_URL
from engram.models import (
    CollectionOut,
    DocumentIn,
    IndexRequest,
    IndexResponse,
    RetrieveResponse,
    RetrieveResult,
)
from engram.store import Store

_store: Store | None = None


def _get_store() -> Store:
    if _store is None:
        raise RuntimeError("Store not initialised")
    return _store


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _store  # noqa: PLW0603
    if DATABASE_URL:
        _store = Store(DATABASE_URL)
        await _store.init_db()
    yield


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
        chunks = chunk_text(doc.content)
        if not chunks:
            continue
        vecs = await embeddings.embed(chunks)
        _, n = await store.index_document(
            collection_id=collection_id,
            path=doc.path,
            metadata=doc.metadata,
            chunks=chunks,
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
