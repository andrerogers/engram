"""Embedding client — calls OpenRouter for text-embedding-3-small, or a substitute that calls nothing.

Retries each batch on 429 / 5xx with exponential backoff: 0.5s → 1.0s → 2.0s (3 attempts).

``EMBEDDING_PROVIDER=substitute`` is what the e2e suite runs on, beside Hive's substitute chat
model. Until it existed, the suite was said to spend nothing while every fact, index and recall
in it was a real OpenRouter call that needed the network and a funded key.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import re

import httpx

from engram import telemetry
from engram.config import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    EMBEDDING_PROVIDER,
    OPENROUTER_API_KEY,
    OPENROUTER_EMBEDDINGS_URL,
)

PROVIDERS = ("openrouter", "substitute")

log = logging.getLogger(__name__)

_client: httpx.AsyncClient | None = None
_RETRY_STATUSES = {429, 500, 502, 503, 504}
_RETRY_DELAYS = [0.5, 1.0, 2.0]


def _get_client() -> httpx.AsyncClient:
    global _client  # noqa: PLW0603
    if _client is None:
        _client = httpx.AsyncClient(timeout=60.0)
    return _client


async def _post_with_retry(client: httpx.AsyncClient, url: str, **kwargs: object) -> httpx.Response:
    """POST with retry on transient HTTP errors (429 / 5xx)."""
    resp: httpx.Response | None = None
    for attempt, delay in enumerate(_RETRY_DELAYS):
        resp = await client.post(url, **kwargs)  # type: ignore[arg-type]
        if resp.status_code not in _RETRY_STATUSES:
            return resp
        if attempt < len(_RETRY_DELAYS) - 1:
            log.warning(
                "embed: HTTP %s on attempt %d/%d — retrying in %.1fs",
                resp.status_code,
                attempt + 1,
                len(_RETRY_DELAYS),
                delay,
            )
            await asyncio.sleep(delay)
    assert resp is not None
    return resp


async def embed(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts via OpenRouter (OpenAI-compatible endpoint).

    Automatically splits into sub-batches of EMBEDDING_BATCH_SIZE.
    Retries each batch up to 3 times on 429 / 5xx before raising.
    Returns a list of float vectors, one per input text.
    """
    if EMBEDDING_PROVIDER not in PROVIDERS:
        raise RuntimeError(
            f"EMBEDDING_PROVIDER={EMBEDDING_PROVIDER!r} is not one of {', '.join(PROVIDERS)}"
        )
    with telemetry.embedding(len(texts), EMBEDDING_PROVIDER):
        if EMBEDDING_PROVIDER == "substitute":
            return [substitute_vector(t) for t in texts]
        return await _embed(texts)


def substitute_vector(text: str) -> list[float]:
    """A deterministic unit vector built from the text's words — the hashing trick.

    Each lowercased word adds ±1 to one of the dimensions, chosen by its hash, so two texts that
    share words point the same way and cosine similarity measures their overlap. That keeps
    vector search ranking something meaningful rather than noise, which a random or constant
    vector would not. It knows no synonyms: "deploy" and "release" are unrelated to it.
    """
    vector = [0.0] * EMBEDDING_DIMENSIONS
    for word in re.findall(r"\w+", text.lower()):
        digest = hashlib.blake2b(word.encode(), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "big") % EMBEDDING_DIMENSIONS
        vector[index] += 1.0 if digest[4] & 1 else -1.0
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0.0:
        # No words at all. A zero vector has no direction, and cosine against it is undefined.
        vector[0] = 1.0
        return vector
    return [x / norm for x in vector]


async def _embed(texts: list[str]) -> list[list[float]]:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set — cannot generate embeddings")

    client = _get_client()
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[i : i + EMBEDDING_BATCH_SIZE]
        resp = await _post_with_retry(
            client,
            OPENROUTER_EMBEDDINGS_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": EMBEDDING_MODEL,
                "input": batch,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        sorted_items = sorted(data["data"], key=lambda x: x["index"])
        all_embeddings.extend(item["embedding"] for item in sorted_items)

    return all_embeddings
