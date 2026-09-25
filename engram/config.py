"""Configuration from environment variables."""

from __future__ import annotations

import json
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


def stored_openrouter_key() -> str:
    """The OpenRouter key the user set in Settings, or "" — read on every call, not at import.

    Hive writes ``<home>/credentials.json`` (``hive/hive/credentials.py``); a key there wins over
    OPENROUTER_API_KEY. Read per call so a key added in the panel works without a restart.
    """
    try:
        data = json.loads((BRAINSTACK_HOME / "credentials.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    entry = data.get("openrouter") if isinstance(data, dict) else None
    key = entry.get("api_key") if isinstance(entry, dict) else None
    return key.strip() if isinstance(key, str) else ""


OPENROUTER_EMBEDDINGS_URL: str = "https://openrouter.ai/api/v1/embeddings"
EMBEDDING_MODEL: str = "openai/text-embedding-3-small"
EMBEDDING_DIMENSIONS: int = 1536
# "openrouter" calls the model above; "substitute" derives a vector from the text's own words and
# calls nothing — what the e2e suite runs on (engram/embeddings.py).
EMBEDDING_PROVIDER: str = os.environ.get("EMBEDDING_PROVIDER", "openrouter").lower()

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
