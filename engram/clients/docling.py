"""DoclingClient — async HTTP client for the Docling-serve sidecar.

Docling-serve exposes an async task API:
  POST /v1/convert/file/async            → {"task_id": "..."}
  POST /v1/chunk/hybrid/file/async       → {"task_id": "..."}
  POST /v1/chunk/hierarchical/file/async → {"task_id": "..."}
  GET  /v1/status/poll/{task_id}?wait=30 → {"task_status": "success|failure|pending|started", ...}
  GET  /v1/result/{task_id}              → full result payload
  GET  /v1/clear/results?older_then=N    → (maintenance; response ignored)

All verified endpoint paths, field names, and response shapes are documented
in VERIFICATION.md at the repo root (created during PR E7).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from engram.config import (
    DOCLING_ENABLED,
    DOCLING_MAX_WAIT,
    DOCLING_POLL_INTERVAL,
    DOCLING_TIMEOUT,
    DOCLING_URL,
)

log = logging.getLogger(__name__)


class DoclingUnavailable(RuntimeError):
    """Raised when the Docling client is disabled or has not been started."""


class DoclingTaskFailed(RuntimeError):
    """Raised when a Docling task reports failure or exceeds max_wait."""


class DoclingClient:
    """Async HTTP client for the Docling-serve sidecar."""

    def __init__(
        self,
        base_url: str | None = None,
        timeout: float | None = None,
        poll_interval: float | None = None,
        max_wait: float | None = None,
        enabled: bool | None = None,
    ) -> None:
        self.base_url = (base_url or DOCLING_URL).rstrip("/")
        self.timeout = timeout if timeout is not None else DOCLING_TIMEOUT
        self.poll_interval = poll_interval if poll_interval is not None else DOCLING_POLL_INTERVAL
        self.max_wait = max_wait if max_wait is not None else DOCLING_MAX_WAIT
        self._enabled = (enabled if enabled is not None else DOCLING_ENABLED) and bool(self.base_url)
        self._client: httpx.AsyncClient | None = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    # ── Lifecycle ─────────────────────────────────────────────────────────

    async def startup(self) -> None:
        """Open the underlying httpx session."""
        if self._enabled:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
            )
            log.info("DoclingClient started — base_url=%s", self.base_url)

    async def shutdown(self) -> None:
        """Close the underlying httpx session."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
            log.info("DoclingClient stopped")

    # ── Health ────────────────────────────────────────────────────────────

    async def health(self) -> bool:
        """Return True if Docling responds to /health successfully."""
        if not self._enabled or self._client is None:
            return False
        try:
            r = await self._client.get("/health", timeout=2.0)
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    # ── Public API ────────────────────────────────────────────────────────

    async def chunk_hybrid_file(self, file_bytes: bytes, filename: str) -> list[dict[str, Any]]:
        """Chunk a binary file using the Docling hybrid chunker.

        Returns a list of chunk dicts, each with at minimum a "text" key.
        """
        task_id = await self._submit_chunk_hybrid_file(file_bytes, filename)
        result = await self._await_task(task_id)
        return list(result.get("chunks") or [])

    async def chunk_hierarchical_file(
        self, file_bytes: bytes, filename: str
    ) -> list[dict[str, Any]]:
        """Chunk a binary file using the Docling hierarchical chunker."""
        task_id = await self._submit_chunk_hierarchical_file(file_bytes, filename)
        result = await self._await_task(task_id)
        return list(result.get("chunks") or [])

    async def chunk_text_hybrid(
        self, text: str, filename: str = "text.md"
    ) -> list[dict[str, Any]]:
        """Chunk a plain-text string via the hybrid chunker (submitted as UTF-8 bytes)."""
        return await self.chunk_hybrid_file(
            file_bytes=text.encode("utf-8"),
            filename=filename,
        )

    async def convert_file_to_markdown(self, file_bytes: bytes, filename: str) -> str:
        """Convert a binary file to Markdown via Docling.

        Returns the extracted Markdown string.
        """
        task_id = await self._submit_convert_file(file_bytes, filename)
        result = await self._await_task(task_id)
        document = result.get("document") or {}
        return str(document.get("md_content") or "")

    async def clear_results(self, older_than_seconds: int = 3600) -> None:
        """Ask Docling to delete cached task results older than *older_than_seconds*.

        Best-effort maintenance call — response body is ignored.
        """
        self._require()
        assert self._client is not None
        try:
            await self._client.get(
                "/v1/clear/results",
                params={"older_then": older_than_seconds},
            )
        except httpx.HTTPError as exc:
            log.warning("DoclingClient.clear_results failed (non-fatal): %s", exc)

    # ── Internal: submission ──────────────────────────────────────────────

    async def _submit_convert_file(self, file_bytes: bytes, filename: str) -> str:
        self._require()
        assert self._client is not None
        r = await self._client.post(
            "/v1/convert/file/async",
            files={"files": (filename, file_bytes)},
            data={"to": "markdown"},
        )
        r.raise_for_status()
        return str(r.json()["task_id"])

    async def _submit_chunk_hybrid_file(self, file_bytes: bytes, filename: str) -> str:
        self._require()
        assert self._client is not None
        r = await self._client.post(
            "/v1/chunk/hybrid/file/async",
            files={"files": (filename, file_bytes)},
            data={"convert_to": "markdown", "chunking_chunker": "hybrid"},
        )
        r.raise_for_status()
        return str(r.json()["task_id"])

    async def _submit_chunk_hierarchical_file(self, file_bytes: bytes, filename: str) -> str:
        self._require()
        assert self._client is not None
        r = await self._client.post(
            "/v1/chunk/hierarchical/file/async",
            files={"files": (filename, file_bytes)},
            data={"convert_to": "markdown", "chunking_chunker": "hierarchical"},
        )
        r.raise_for_status()
        return str(r.json()["task_id"])

    # ── Internal: polling ─────────────────────────────────────────────────

    async def _await_task(self, task_id: str) -> dict[str, Any]:
        """Poll /v1/status/poll/{task_id} until success or failure.

        Raises:
            DoclingTaskFailed: On task failure or max_wait exceeded.
        """
        self._require()
        assert self._client is not None
        loop = asyncio.get_event_loop()
        deadline = loop.time() + self.max_wait

        while True:
            if loop.time() > deadline:
                raise DoclingTaskFailed(
                    f"Task {task_id!r} exceeded max_wait ({self.max_wait}s)"
                )

            status_resp = await self._client.get(
                f"/v1/status/poll/{task_id}",
                params={"wait": 30},
                timeout=35.0,
            )
            status_resp.raise_for_status()
            status = status_resp.json()

            task_status = status.get("task_status")
            if task_status == "success":
                result_resp = await self._client.get(f"/v1/result/{task_id}")
                result_resp.raise_for_status()
                return dict(result_resp.json())
            if task_status == "failure":
                raise DoclingTaskFailed(
                    f"Task {task_id!r}: {status.get('error_message', 'unknown error')}"
                )

            await asyncio.sleep(self.poll_interval)

    def _require(self) -> None:
        if not self._enabled or self._client is None:
            raise DoclingUnavailable(
                "Docling integration is disabled or client not started. "
                "Set DOCLING_ENABLED=true and call startup() first."
            )
