# Engram

**Engram** is the RAG (Retrieval-Augmented Generation) service of [brainstack](https://github.com/andrerogers/brainstack). It handles document indexing with token-aware chunking, OpenRouter embeddings, and pgvector similarity search — giving Hive access to a searchable knowledge base during chat.

---

## Role in the pipeline

```
Hive (orchestration core)
  │  HTTP  GET /retrieve?query=…   ← fetch relevant chunks for a chat request
  │  HTTP  POST /index             ← index a document (separate from chat)
  ▼
Engram (FastAPI) — chunking · embedding · retrieval
  │
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
| `POST` | `/index` | Chunk, embed, and index a document into a collection |
| `GET` | `/retrieve` | Semantic search — embed query, return top-k chunks |
| `GET` | `/collections` | List all collections |
| `DELETE` | `/collections/{id}` | Delete a collection and all its chunks |

---

## Directory layout

```
engram/
├── engram/
│   ├── app.py          FastAPI app, lifespan (pool open/close), route registration
│   ├── store.py        Async Postgres store — AsyncConnectionPool (psycopg-pool)
│   ├── models.py       Pydantic schemas for all routes
│   ├── embeddings.py   OpenRouter batch embedding client (up to 100 chunks per call)
│   ├── chunker.py      Token-aware splitter — tiktoken cl100k_base, 512 tok/64 overlap
│   └── config.py       Env-based config (DATABASE_URL, ENGRAM_PORT, OPENROUTER_API_KEY)
├── migrations/
│   ├── 0001.create_schema.sql          engram schema + documents + chunks tables
│   ├── 0002.add_chunk_embedding.sql    embedding vector(1536) on chunks
│   ├── 0003.add_collections.sql        collections table
│   └── 0004.add_chunks_hnsw_index.sql  HNSW index on chunks.embedding
├── tests/
│   └── test_engram_api.py   12 tests — health, index, retrieve, collections
├── pyproject.toml
└── .env.example
```

---

## Setup

**Prerequisites:** Python 3.13, [uv](https://docs.astral.sh/uv/), PostgreSQL with pgvector (`pgvector/pgvector:pg16`)

```bash
cd engram
cp .env.example .env
# Edit .env — set DATABASE_URL and OPENROUTER_API_KEY
uv sync
uv run task dev      # starts uvicorn on port 8613 with --reload
```

---

## Configuration (`.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | — | PostgreSQL connection string (required) |
| `ENGRAM_PORT` | `8613` | Port to bind |
| `OPENROUTER_API_KEY` | — | API key for OpenRouter embeddings (required for `/index` and `/retrieve`) |

---

## Schema

All tables live in the `engram` schema in the shared PostgreSQL database.

| Table | Key columns |
|-------|-------------|
| `engram.collections` | `id UUID`, `name`, `created_at` |
| `engram.documents` | `id UUID`, `collection_id`, `title`, `source`, `created_at` |
| `engram.chunks` | `id`, `document_id`, `content`, `embedding vector(1536)`, `token_count`, `chunk_index` |

HNSW index on `engram.chunks.embedding` for fast cosine similarity search.

---

## Chunking

Documents are split with `chunker.py` before indexing:

| Parameter | Value |
|-----------|-------|
| Tokenizer | tiktoken `cl100k_base` |
| Chunk size | 512 tokens |
| Overlap | 64 tokens |
| Boundary | Line-aware (won't split mid-line unless a single line exceeds the limit) |

---

## Embeddings

Chunks are batch-embedded via OpenRouter `openai/text-embedding-3-small` (1536 dims, up to 100 chunks per API call). Retrieval embeds the query with the same model, then runs cosine similarity (`<=>`) against the HNSW index.

---

## Connection pooling

Engram uses `psycopg-pool AsyncConnectionPool(min_size=2, max_size=10)` opened lazily on first request and closed in the lifespan teardown. All queries go through `_run(fn)`:

```python
async with pool.connection() as conn:
    return await fn(conn)
```

---

## Development

```bash
uv run task dev       # uvicorn --reload on port 8613
uv run task test      # pytest -v
uv run task lint      # ruff check .
uv run task typecheck # mypy engram/
uv run task fmt       # ruff format .
uv run task check     # lint + typecheck + test
```

---

## Tech stack

| Layer | Library |
|-------|---------|
| Web framework | FastAPI + Uvicorn |
| Validation | Pydantic v2 |
| Database | psycopg v3 async + psycopg-pool |
| Migrations | yoyo-migrations (isolated `_engram_yoyo_*` tables) |
| Vector search | pgvector (1536-dim HNSW cosine) |
| Tokenizer | tiktoken `cl100k_base` |
| Embeddings | OpenRouter `openai/text-embedding-3-small` |
| Observability | brainstack-optics (OTel) |

---

**Project devlog:** [andrerogers/vault — brainstack.md](https://github.com/andrerogers/vault/blob/master/projects/brainstack/brainstack.md)

---

MIT © 2025 [Andre Rogers](https://github.com/andrerogers)
