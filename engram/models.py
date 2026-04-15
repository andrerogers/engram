"""Pydantic request/response schemas."""

from __future__ import annotations

from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------


class DocumentIn(BaseModel):
    path: str | None = None
    content: str
    metadata: dict[str, str] | None = None


class IndexRequest(BaseModel):
    collection_id: str | None = None
    workspace_id: str | None = None
    collection_name: str | None = None
    documents: list[DocumentIn]


class IndexResponse(BaseModel):
    indexed_count: int
    collection_id: str
    chunk_count: int


class IndexWorkspaceRequest(BaseModel):
    workspace_id: str
    workspace_path: str
    collection_name: str | None = None
    file_globs: list[str] | None = None


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


class RetrieveResult(BaseModel):
    chunk_id: str
    document_path: str | None
    content: str
    score: float
    modality: str = "text"
    chunker: str = "tiktoken-fallback"


class RetrieveResponse(BaseModel):
    results: list[RetrieveResult]


# ---------------------------------------------------------------------------
# Collections
# ---------------------------------------------------------------------------


class CollectionOut(BaseModel):
    id: str
    workspace_id: str
    name: str
    created_at: str
