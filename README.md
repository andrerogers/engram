# Engram

**Engram** is the RAG (Retrieval-Augmented Generation) service of [brainstack](https://github.com/andrerogers/brainstack). It handles async document ingestion via Docling (in-process), local file object storage, token-aware chunking, OpenRouter embeddings, and sqlite-vec similarity search — giving Hive access to a searchable, multimodal-ready knowledge base.

---

## Role in the pipeline

```
Hive (orchestration core)
  │  HTTP  POST /index             ← index text documents
  │  HTTP  POST /index/file        ← async file ingestion (E10+)
  │  HTTP  GET  /retrieve          ← fetch relevant chunks for a chat request
  ▼
Engram (FastAPI :8613)
  │  Docling (in-process, optional extra) — PDF/DOCX → Markdown + chunking
  │  ~/.brainstack/objects/               — raw file storage (engram.db holds zero bytes)
  ▼
SQLite engram.db + sqlite-vec (vec0 cosine indexes)
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
│   ├── store.py                    SQLite store (engram.db) — rows, jobs, facts, signals
│   ├── vector_store.py             VectorStore protocol + SqliteVecStore (the only sqlite-vec code)
│   ├── models.py                   Pydantic request/response schemas
│   ├── embeddings.py               OpenRouter batch embedding client (retry + backoff)
│   ├── chunker.py                  Token-aware text chunking (tiktoken)
│   ├── config.py                   All settings from env vars
│   ├── clients/
│   │   ├── docling.py              DoclingEngine — in-process convert + hybrid chunk
│   │   └── storage/
│   │       ├── base.py             ObjectStore ABC (put/get/exists/delete/presigned_url)
│   │       ├── memory.py           InMemoryObjectStore (tests; sentinel presigned URLs)
│   │       └── local.py            LocalFileObjectStore — atomic writes, key validation
│   └── processors/
│       ├── base.py                 Modality + ChunkerKind enums, ChunkCandidate dataclass, Protocols
│       ├── tiktoken_processor.py   TiktokenProcessor (sync; current default; Docling fallback)
│       ├── docling_text.py         DoclingTextProcessor stub (E8)
│       └── docling_file.py         DoclingFileProcessor stub (E8)
├── tests/
│   ├── unit/                       Hermetic tests, no network (53 passing, <1s)
│   │   ├── test_storage_contract.py    ObjectStoreContract — 11 behavioral tests
│   │   ├── test_storage_memory.py      InMemory passes contract
│   │   ├── test_processors.py          TiktokenProcessor
│   │   ├── test_ingest_jobs.py         Job CRUD + orphan recovery
│   │   ├── test_storage_local.py       LocalFile passes contract + traversal/atomicity
│   │   └── test_docling_engine.py      DoclingEngine availability + fallbacks (+ `docling` mark)
│   ├── integration/                Opt-in suites
│   │   ├── fixtures/sample.{pdf,md}    Test fixtures
│   │   └── test_end_to_end.py          PDF → objects → Docling → sqlite-vec (`-m docling`)
│   └── test_routes.py              Route tests (mocked store + embeddings)
├── VERIFICATION.md                 Verified Docling API shape (run `-m docling` to re-verify)
├── pyproject.toml
└── .env.example
```

---

## Setup

**Prerequisites:** Python 3.13, [uv](https://docs.astral.sh/uv/), nothing else — the database is a local SQLite file

```bash
cd engram
cp .env.example .env
# Set OPENROUTER_API_KEY (engram.db is created at ~/.brainstack/engram.db on first start)
uv sync
uv run task dev      # uvicorn on :8613 with --reload
```

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `ENGRAM_DB_PATH` | `~/.brainstack/engram.db` | SQLite database file |
| `ENGRAM_PORT` | `8613` | HTTP port |
| `OPENROUTER_API_KEY` | — | Required for `/index` and `/retrieve` |
| `BRAINSTACK_HOME` | `~/.brainstack` | Base for the database and object store |
| `ENGRAM_OBJECT_DIR` | `<BRAINSTACK_HOME>/objects` | Raw file storage |
| `DOCLING_ENABLED` | `true` | Set `false` → tiktoken fallback, no Docling |
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

All tables live in `engram.db`. Migrations are ordered SQL in `engram/store.py` (`_migrations`), applied with `PRAGMA user_version`.

| Table | Key columns |
|-------|-------------|
| `collections` | `id`, `workspace_id`, `name`, `created_at` |
| `documents` | `id`, `collection_id`, `path`, `metadata`, `object_key`, `file_hash`, `file_size`, `source_mime` |
| `chunks` | `id`, `document_id`, `content`, `modality`, `chunker`, `chunker_version`, `media_ref` |
| `ingest_jobs` | `id`, `collection_id`, `document_id`, `status`, `filename`, `object_key`, `last_heartbeat`, `error_message` |

**SHA-256 dedup:** partial unique index on `(collection_id, file_hash) WHERE file_hash IS NOT NULL`. Same file submitted to the same collection is a no-op.

| `facts` / `signals` | distilled facts and outcome signals, scoped by `workspace_id` |
| `chunk_vectors` / `fact_vectors` / `signal_vectors` | vec0 virtual tables: `FLOAT[1536] distance_metric=cosine`, partitioned by collection or workspace |

**Vectors:** every vector search goes through `VectorStore` (`engram/vector_store.py`). Search is exact k-nearest-neighbour — sqlite-vec has no approximate (HNSW) index — filtered by partition key and metadata (`modality`, `signal_type`). Retrieval defaults to `modality = 'text'`. Triggers delete a row's vector when the row is deleted, including by `ON DELETE CASCADE`.

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
worker: write to the object store → Docling parse → embed → store → mark completed
         ↕ bump last_heartbeat every 10s
startup: recover_orphan_jobs() re-queues stale processing jobs
```

---

## Development

```bash
uv run task dev       # uvicorn --reload on :8613
uv run task test      # pytest -v (integration tests excluded by default)
uv run task check     # ruff check + format --check + pytest

# Real Docling (optional extra, ~1.6 GB of CPU wheels — never installed in CI)
uv sync --extra docling
uv run pytest -m docling -v
```

---

## Tech stack

| Layer | Library |
|-------|---------|
| Web framework | FastAPI + Uvicorn |
| Validation | Pydantic v2 |
| Database | SQLite (stdlib `sqlite3`, WAL) |
| Migrations | ordered SQL + `PRAGMA user_version` |
| Vector search | sqlite-vec vec0 (exact KNN, cosine) |
| Tokenizer | tiktoken `cl100k_base` |
| Embeddings | OpenRouter `openai/text-embedding-3-small` |
| Object storage | local filesystem (`LocalFileObjectStore`, atomic writes) |
| Document parsing | Docling, in-process (optional `docling` extra) |
| HTTP client | httpx (async) |
| Observability | brainstack-optics (OTel) |

---

MIT © 2025 [Andre Rogers](https://github.com/andrerogers)
