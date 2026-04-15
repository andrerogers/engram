"""Configuration from environment variables."""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

DATABASE_URL: str = os.environ.get("DATABASE_URL", "")
ENGRAM_PORT: int = int(os.environ.get("ENGRAM_PORT", "8613"))
OPENROUTER_API_KEY: str = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_EMBEDDINGS_URL: str = "https://openrouter.ai/api/v1/embeddings"
EMBEDDING_MODEL: str = "openai/text-embedding-3-small"
EMBEDDING_DIMENSIONS: int = 1536

# Chunking defaults
CHUNK_SIZE_TOKENS: int = 512
CHUNK_OVERLAP_TOKENS: int = 64
EMBEDDING_BATCH_SIZE: int = 100

# Docling sidecar
DOCLING_URL: str = os.environ.get("DOCLING_URL", "http://localhost:5001")
DOCLING_ENABLED: bool = os.environ.get("DOCLING_ENABLED", "true").lower() == "true"
DOCLING_TIMEOUT: float = float(os.environ.get("DOCLING_TIMEOUT", "120.0"))
DOCLING_POLL_INTERVAL: float = float(os.environ.get("DOCLING_POLL_INTERVAL", "2.0"))
DOCLING_MAX_WAIT: float = float(os.environ.get("DOCLING_MAX_WAIT", "600.0"))

# MinIO object storage
MINIO_ENDPOINT: str = os.environ.get("MINIO_ENDPOINT", "http://localhost:9000")
MINIO_ACCESS_KEY: str = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY: str = os.environ.get("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET: str = os.environ.get("MINIO_BUCKET", "engram")
MINIO_ENABLED: bool = os.environ.get("MINIO_ENABLED", "false").lower() == "true"

# Ingest job settings
MAX_CONCURRENT_INGEST_JOBS: int = int(os.environ.get("MAX_CONCURRENT_INGEST_JOBS", "4"))
MAX_FILE_SIZE_MB: int = int(os.environ.get("MAX_FILE_SIZE_MB", "50"))
INGEST_HEARTBEAT_STALE_SECONDS: int = int(os.environ.get("INGEST_HEARTBEAT_STALE_SECONDS", "60"))
INGEST_JOB_RETENTION_DAYS: int = int(os.environ.get("INGEST_JOB_RETENTION_DAYS", "7"))

# Chunker registry
DEFAULT_TEXT_CHUNKER: str = os.environ.get("DEFAULT_TEXT_CHUNKER", "docling-hybrid")
