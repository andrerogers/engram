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
# File ingestion
# ---------------------------------------------------------------------------


class FileIngestResponse(BaseModel):
    """Response for POST /index/file.

    status="accepted"  → job_id is set, HTTP 202.
    status="duplicate" → document_id is set (existing doc), HTTP 200.
    """

    status: str
    job_id: str | None = None
    document_id: str | None = None


class IngestJobOut(BaseModel):
    id: str
    collection_id: str
    status: str
    filename: str | None = None
    object_key: str | None = None
    document_id: str | None = None
    error_message: str | None = None
    last_heartbeat: str | None = None
    created_at: str
    updated_at: str


# ---------------------------------------------------------------------------
# Collections
# ---------------------------------------------------------------------------


class CollectionOut(BaseModel):
    id: str
    workspace_id: str
    name: str
    created_at: str


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------


class DocumentOut(BaseModel):
    id: str
    collection_id: str
    path: str | None = None
    metadata: dict[str, str] | None = None
    created_at: str
