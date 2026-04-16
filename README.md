# Engram

**Engram** is the RAG (Retrieval-Augmented Generation) service of [brainstack](https://github.com/andrerogers/brainstack). It handles async document ingestion via Docling, MinIO object storage, token-aware chunking, OpenRouter embeddings, and pgvector similarity search — giving Hive access to a searchable, multimodal-ready knowledge base.

---

## Role in the pipeline

```
Hive (orchestration core)
  │  HTTP  POST /index             ← index text documents
  │  HTTP  POST /index/file        ← async file ingestion (E10+)
  │  HTTP  GET  /retrieve          ← fetch relevant chunks for a chat request
  ▼
Engram (FastAPI :8613)
  │  Docling-serve sidecar (:5001) — PDF/DOCX → Markdown + chunking
  │  MinIO (:9000)                 — raw file storage (Postgres holds zero bytes)
  ▼
PostgreSQL + pgvector (engram.* schema)
  │  OpenRouter embeddings
  ▼
openai/text-embedding-3-small (1536 dims)
```

Hive calls Engram directly over HTTP — Cortex does not proxy these calls.

---

## Routes

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness check |
| `POST` | `/index` | Chunk, embed, and index text document(s) into a collection |
| `GET` | `/retrieve` | Semantic search — `?q=...&collection_id=...&k=5` |
| `GET` | `/collections` | List collections (filter: `?workspace_id=...`) |
| `DELETE` | `/collections/{id}` | Delete collection and all its chunks |

> File ingestion routes (`POST /index/file`, `/documents/*`) land in E10–E11.

---

## Directory layout

```
engram/
├── engram/
│   ├── app.py                      FastAPI app + lifespan (DB init, DoclingClient startup/shutdown)
│   ├── store.py                    Async pgvector store (psycopg3 pool)
│   ├── models.py                   Pydantic request/response schemas
│   ├── embeddings.py               OpenRouter batch embedding client (retry + backoff)
│   ├── chunker.py                  Token-aware text chunking (tiktoken)
│   ├── config.py                   All settings from env vars
│   ├── clients/
│   │   ├── docling.py              DoclingClient — async Docling-serve HTTP client
│   │   └── storage/
│   │       ├── base.py             ObjectStore ABC (put/get/exists/delete/presigned_url)
│   │       ├── memory.py           InMemoryObjectStore (dev/test; sentinel presigned URLs)
│   │       └── minio.py            MinioObjectStore (aioboto3, S3-compatible)
│   └── processors/
│       ├── base.py                 Modality + ChunkerKind enums, ChunkCandidate dataclass, Protocols
│       ├── tiktoken_processor.py   TiktokenProcessor (sync; current default; Docling fallback)
│       ├── docling_text.py         DoclingTextProcessor stub (E8)
│       └── docling_file.py         DoclingFileProcessor stub (E8)
├── migrations/
│   ├── 0001.create_collections.sql
│   ├── 0002.create_documents_and_chunks.sql
│   ├── 0003.add_indexes.sql
│   ├── 0004.add_modality_and_chunker.sql    modality/chunker fields + partial HNSW WHERE modality='text'
│   ├── 0005.create_ingest_jobs.sql          async job table with last_heartbeat
│   └── 0006.add_document_object_storage.sql  object_key/source_mime/file_size/file_hash + SHA-256 dedup index
├── tests/
│   ├── unit/                       Hermetic tests, no network (53 passing, <1s)
│   │   ├── test_storage_contract.py    ObjectStoreContract — 11 behavioral tests
│   │   ├── test_storage_memory.py      InMemory passes contract
│   │   ├── test_processors.py          TiktokenProcessor
│   │   ├── test_ingest_jobs.py         Job CRUD + orphan recovery
│   │   └── test_docling_client.py      DoclingClient (httpx-mocked, 17 tests)
│   ├── integration/                Real-service tests (opt-in: -m integration)
│   │   ├── fixtures/sample.{pdf,md}    Test fixtures
│   │   ├── test_storage_minio.py       MinIO passes ObjectStoreContract
│   │   └── test_docling_real.py        E7 gate — 7 tests against real Docling-serve
│   └── test_routes.py              Route tests (mocked store + embeddings)
├── compose.test.yml                MinIO (:19000) + Docling (:15001) for integration tests
├── VERIFICATION.md                 Verified Docling-serve API shape (fill after E7 gate)
├── pyproject.toml
└── .env.example
```

---

## Setup

**Prerequisites:** Python 3.13, [uv](https://docs.astral.sh/uv/), PostgreSQL with pgvector (`pgvector/pgvector:pg16` — requires pgvector ≥ 0.7.0 for partial HNSW index)

```bash
cd engram
cp .env.example .env
# Set DATABASE_URL, OPENROUTER_API_KEY
uv sync
uv run task dev      # uvicorn on :8613 with --reload
```

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | — | PostgreSQL connection string (required) |
| `ENGRAM_PORT` | `8613` | HTTP port |
| `OPENROUTER_API_KEY` | — | Required for `/index` and `/retrieve` |
| `DOCLING_URL` | `http://localhost:5001` | Docling-serve base URL |
| `DOCLING_ENABLED` | `true` | Set `false` → tiktoken fallback |
| `DOCLING_TIMEOUT` | `120.0` | Per-request HTTP timeout (s) |
| `DOCLING_POLL_INTERVAL` | `2.0` | Poll sleep interval (s) |
| `DOCLING_MAX_WAIT` | `600.0` | Max wait per Docling task (s) |
| `MINIO_ENDPOINT` | `http://localhost:9000` | MinIO endpoint |
| `MINIO_ACCESS_KEY` | `minioadmin` | MinIO access key |
| `MINIO_SECRET_KEY` | `minioadmin` | MinIO secret key |
| `MINIO_BUCKET` | `engram` | MinIO bucket name |
| `MINIO_ENABLED` | `false` | `true` → MinIO; `false` → InMemory |
| `MAX_CONCURRENT_INGEST_JOBS` | `4` | Worker concurrency cap (E9) |
| `MAX_FILE_SIZE_MB` | `50` | Max upload size for `/index/file` (E10) |
| `INGEST_HEARTBEAT_STALE_SECONDS` | `60` | Heartbeat age before orphan recovery |
| `INGEST_JOB_RETENTION_DAYS` | `7` | Retention for completed/failed jobs |
| `DEFAULT_TEXT_CHUNKER` | `docling-hybrid` | Active chunker for `/index` (E8) |
| `CHUNK_SIZE_TOKENS` | `512` | Tiktoken chunk size |
| `CHUNK_OVERLAP_TOKENS` | `64` | Tiktoken overlap |
| `EMBEDDING_BATCH_SIZE` | `100` | Max texts per embedding API call |

---

## Database schema

All tables in the `engram` PostgreSQL schema. Migrations via yoyo-migrations (isolated `_engram_yoyo_*` tables).

| Table | Key columns |
|-------|-------------|
| `engram.collections` | `id`, `workspace_id`, `name`, `created_at` |
| `engram.documents` | `id`, `collection_id`, `path`, `metadata`, `object_key`, `file_hash`, `file_size`, `source_mime` |
| `engram.chunks` | `id`, `document_id`, `content`, `embedding vector(1536)`, `modality`, `chunker`, `chunker_version`, `media_ref` |
| `engram.ingest_jobs` | `id`, `collection_id`, `document_id`, `status`, `filename`, `object_key`, `last_heartbeat`, `error_message` |

**SHA-256 dedup:** partial unique index on `(collection_id, file_hash) WHERE file_hash IS NOT NULL`. Same file submitted to the same collection is a no-op.

**Partial HNSW index:** `WHERE modality = 'text'` — only text chunks are ANN-searched in Phase 1. Image/audio/video embeddings (Phase 2+) don't inflate search cost.

---

## Chunking and processors

All processors output `list[ChunkCandidate]`. Each candidate carries `content`, `chunk_index`, `modality`, `chunker`, `chunker_version`, and optional `media_ref`/`media_metadata`.

| Processor | Kind | Status |
|-----------|------|--------|
| `TiktokenProcessor` | `tiktoken-fallback` | Active — current `/index` default |
| `DoclingTextProcessor` | `docling-hybrid` | Stub — implemented in E8 |
| `DoclingFileProcessor` | `docling-hybrid` | Stub — implemented in E8 |

`InMemoryObjectStore` and `MinioObjectStore` share `ObjectStoreContract` (11 behavioral tests) — drift between backends is structurally impossible.

---

## Ingest job lifecycle (E9+)

```
POST /index/file → create job (pending) → schedule worker
worker: upload to MinIO → Docling parse → embed → store → mark completed
         ↕ bump last_heartbeat every 10s
startup: recover_orphan_jobs() re-queues stale processing jobs (DB now(), not app clock)
```

---

## Development

```bash
uv run task dev       # uvicorn --reload on :8613
uv run task test      # pytest -v (integration tests excluded by default)
uv run task check     # ruff check + format --check + pytest

# Integration tests (requires compose.test.yml)
docker compose -f compose.test.yml up -d
uv run pytest -m integration -v

# E7 verification gate (must pass before E8)
docker compose -f compose.test.yml up -d docling
uv run pytest -m integration tests/integration/test_docling_real.py -v
```

---

## Tech stack

| Layer | Library |
|-------|---------|
| Web framework | FastAPI + Uvicorn |
| Validation | Pydantic v2 |
| Database | psycopg v3 async + psycopg-pool |
| Migrations | yoyo-migrations |
| Vector search | pgvector ≥ 0.7.0 (partial HNSW, cosine) |
| Tokenizer | tiktoken `cl100k_base` |
| Embeddings | OpenRouter `openai/text-embedding-3-small` |
| Object storage | aioboto3 / MinIO (S3-compatible) |
| Document parsing | Docling-serve (async task API) |
| HTTP client | httpx (async) |
| Observability | brainstack-optics (OTel) |

---

MIT © 2025 [Andre Rogers](https://github.com/andrerogers)
