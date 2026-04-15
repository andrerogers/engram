"""Unit tests for DoclingClient — all HTTP calls mocked via httpx.

Tests verify:
- Client lifecycle (startup / shutdown)
- Disabled client raises DoclingUnavailable
- Each submission endpoint called with correct path + multipart fields
- _await_task polls until success / raises on failure / raises on timeout
- Public API methods return the correct parsed values
- clear_results is best-effort (errors do not propagate)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from engram.clients.docling import DoclingClient, DoclingTaskFailed, DoclingUnavailable

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TASK_ID = "task-abc-123"
_BASE = "http://localhost:15001"


def _make_client(enabled: bool = True) -> DoclingClient:
    return DoclingClient(
        base_url=_BASE,
        timeout=10.0,
        poll_interval=0.01,
        max_wait=5.0,
        enabled=enabled,
    )


def _mock_response(status: int = 200, json: object = None) -> MagicMock:
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status
    resp.json.return_value = json or {}
    resp.raise_for_status = MagicMock()
    if status >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            f"HTTP {status}", request=MagicMock(), response=resp
        )
    return resp


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


async def test_startup_creates_client() -> None:
    client = _make_client()
    assert client._client is None
    await client.startup()
    assert client._client is not None
    await client.shutdown()


async def test_shutdown_clears_client() -> None:
    client = _make_client()
    await client.startup()
    await client.shutdown()
    assert client._client is None


async def test_shutdown_idempotent() -> None:
    client = _make_client()
    await client.shutdown()  # without startup — must not raise
    await client.shutdown()


async def test_disabled_client_startup_noop() -> None:
    client = _make_client(enabled=False)
    await client.startup()
    assert client._client is None


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


async def test_health_true_on_200() -> None:
    client = _make_client()
    await client.startup()
    client._client.get = AsyncMock(return_value=_mock_response(200))  # type: ignore[union-attr]
    assert await client.health() is True
    await client.shutdown()


async def test_health_false_on_http_error() -> None:
    client = _make_client()
    await client.startup()
    client._client.get = AsyncMock(side_effect=httpx.ConnectError("refused"))  # type: ignore[union-attr]
    assert await client.health() is False
    await client.shutdown()


async def test_health_false_when_disabled() -> None:
    client = _make_client(enabled=False)
    assert await client.health() is False


# ---------------------------------------------------------------------------
# _require raises when disabled / not started
# ---------------------------------------------------------------------------


async def test_require_raises_when_disabled() -> None:
    client = _make_client(enabled=False)
    with pytest.raises(DoclingUnavailable):
        client._require()


async def test_require_raises_when_not_started() -> None:
    client = _make_client()
    with pytest.raises(DoclingUnavailable):
        client._require()


# ---------------------------------------------------------------------------
# convert_file_to_markdown
# ---------------------------------------------------------------------------


async def test_convert_calls_correct_endpoint() -> None:
    client = _make_client()
    await client.startup()
    submit_resp = _mock_response(200, {"task_id": _TASK_ID})
    status_resp = _mock_response(200, {"task_status": "success"})
    result_resp = _mock_response(200, {"document": {"md_content": "# Hello\nEngram test document"}})

    client._client.post = AsyncMock(return_value=submit_resp)  # type: ignore[union-attr]
    client._client.get = AsyncMock(side_effect=[status_resp, result_resp])  # type: ignore[union-attr]

    md = await client.convert_file_to_markdown(b"%PDF", "test.pdf")

    assert "Engram test document" in md
    post_call = client._client.post.call_args
    assert post_call.args[0] == "/v1/convert/file/async"
    await client.shutdown()


# ---------------------------------------------------------------------------
# chunk_hybrid_file
# ---------------------------------------------------------------------------


async def test_chunk_hybrid_returns_chunk_list() -> None:
    client = _make_client()
    await client.startup()
    submit_resp = _mock_response(200, {"task_id": _TASK_ID})
    status_resp = _mock_response(200, {"task_status": "success"})
    result_resp = _mock_response(200, {"chunks": [{"text": "chunk one"}, {"text": "chunk two"}]})

    client._client.post = AsyncMock(return_value=submit_resp)  # type: ignore[union-attr]
    client._client.get = AsyncMock(side_effect=[status_resp, result_resp])  # type: ignore[union-attr]

    chunks = await client.chunk_hybrid_file(b"data", "doc.pdf")
    assert len(chunks) == 2
    assert chunks[0]["text"] == "chunk one"

    post_call = client._client.post.call_args
    assert "hybrid" in post_call.args[0]
    await client.shutdown()


# ---------------------------------------------------------------------------
# chunk_hierarchical_file
# ---------------------------------------------------------------------------


async def test_chunk_hierarchical_calls_correct_endpoint() -> None:
    client = _make_client()
    await client.startup()
    submit_resp = _mock_response(200, {"task_id": _TASK_ID})
    status_resp = _mock_response(200, {"task_status": "success"})
    result_resp = _mock_response(200, {"chunks": [{"text": "h-chunk"}]})

    client._client.post = AsyncMock(return_value=submit_resp)  # type: ignore[union-attr]
    client._client.get = AsyncMock(side_effect=[status_resp, result_resp])  # type: ignore[union-attr]

    chunks = await client.chunk_hierarchical_file(b"data", "doc.pdf")
    assert len(chunks) == 1

    post_call = client._client.post.call_args
    assert "hierarchical" in post_call.args[0]
    await client.shutdown()


# ---------------------------------------------------------------------------
# chunk_text_hybrid
# ---------------------------------------------------------------------------


async def test_chunk_text_hybrid_sends_utf8() -> None:
    client = _make_client()
    await client.startup()
    submit_resp = _mock_response(200, {"task_id": _TASK_ID})
    status_resp = _mock_response(200, {"task_status": "success"})
    result_resp = _mock_response(200, {"chunks": [{"text": "text chunk"}]})

    client._client.post = AsyncMock(return_value=submit_resp)  # type: ignore[union-attr]
    client._client.get = AsyncMock(side_effect=[status_resp, result_resp])  # type: ignore[union-attr]

    chunks = await client.chunk_text_hybrid("Hello world")
    assert chunks[0]["text"] == "text chunk"

    # Verify bytes were sent
    files = client._client.post.call_args.kwargs["files"]
    assert files["files"][1] == b"Hello world"
    await client.shutdown()


# ---------------------------------------------------------------------------
# _await_task — failure + timeout
# ---------------------------------------------------------------------------


async def test_await_task_raises_on_failure_status() -> None:
    client = _make_client()
    await client.startup()
    submit_resp = _mock_response(200, {"task_id": _TASK_ID})
    failure_resp = _mock_response(200, {"task_status": "failure", "error_message": "bad PDF"})

    client._client.post = AsyncMock(return_value=submit_resp)  # type: ignore[union-attr]
    client._client.get = AsyncMock(return_value=failure_resp)  # type: ignore[union-attr]

    with pytest.raises(DoclingTaskFailed, match="bad PDF"):
        await client.chunk_hybrid_file(b"data", "bad.pdf")
    await client.shutdown()


async def test_await_task_raises_on_timeout() -> None:
    client = DoclingClient(
        base_url=_BASE,
        timeout=10.0,
        poll_interval=0.01,
        max_wait=0.02,  # tiny window — will time out after one poll
        enabled=True,
    )
    await client.startup()
    submit_resp = _mock_response(200, {"task_id": _TASK_ID})
    # Always return "pending" so deadline is exceeded
    pending_resp = _mock_response(200, {"task_status": "pending"})

    client._client.post = AsyncMock(return_value=submit_resp)  # type: ignore[union-attr]
    client._client.get = AsyncMock(return_value=pending_resp)  # type: ignore[union-attr]

    with pytest.raises(DoclingTaskFailed, match="max_wait"):
        await client.chunk_hybrid_file(b"data", "doc.pdf")
    await client.shutdown()


# ---------------------------------------------------------------------------
# clear_results — best-effort (errors swallowed)
# ---------------------------------------------------------------------------


async def test_clear_results_swallows_http_error() -> None:
    client = _make_client()
    await client.startup()
    client._client.get = AsyncMock(side_effect=httpx.ConnectError("refused"))  # type: ignore[union-attr]
    await client.clear_results()  # must not raise
    await client.shutdown()


async def test_clear_results_calls_correct_endpoint() -> None:
    client = _make_client()
    await client.startup()
    client._client.get = AsyncMock(return_value=_mock_response(200, {}))  # type: ignore[union-attr]
    await client.clear_results(older_than_seconds=300)

    call = client._client.get.call_args
    assert call.args[0] == "/v1/clear/results"
    assert call.kwargs["params"]["older_then"] == 300
    await client.shutdown()
