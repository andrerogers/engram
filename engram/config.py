"""Configuration from environment variables."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BRAINSTACK_HOME: Path = Path(
    os.environ.get("BRAINSTACK_HOME") or Path.home() / ".brainstack"
).expanduser()
ENGRAM_DB_PATH: Path = Path(os.environ.get("ENGRAM_DB_PATH") or BRAINSTACK_HOME / "engram.db")
ENGRAM_OBJECT_DIR: Path = Path(
    os.environ.get("ENGRAM_OBJECT_DIR") or BRAINSTACK_HOME / "objects"
).expanduser()
ENGRAM_PORT: int = int(os.environ.get("ENGRAM_PORT", "8613"))
OPENROUTER_API_KEY: str = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_EMBEDDINGS_URL: str = "https://openrouter.ai/api/v1/embeddings"
EMBEDDING_MODEL: str = "openai/text-embedding-3-small"
EMBEDDING_DIMENSIONS: int = 1536

# Chunking defaults
CHUNK_SIZE_TOKENS: int = 512
CHUNK_OVERLAP_TOKENS: int = 64
EMBEDDING_BATCH_SIZE: int = 100

# Docling runs in-process (optional extra); false forces the tiktoken fallback.
DOCLING_ENABLED: bool = os.environ.get("DOCLING_ENABLED", "true").lower() == "true"

# Ingest job settings
MAX_CONCURRENT_INGEST_JOBS: int = int(os.environ.get("MAX_CONCURRENT_INGEST_JOBS", "4"))
MAX_FILE_SIZE_MB: int = int(os.environ.get("MAX_FILE_SIZE_MB", "50"))
INGEST_HEARTBEAT_STALE_SECONDS: int = int(os.environ.get("INGEST_HEARTBEAT_STALE_SECONDS", "60"))
INGEST_JOB_RETENTION_DAYS: int = int(os.environ.get("INGEST_JOB_RETENTION_DAYS", "7"))

# Chunker registry
DEFAULT_TEXT_CHUNKER: str = os.environ.get("DEFAULT_TEXT_CHUNKER", "docling-hybrid")
