"""DoclingClient stub — real implementation wired in PR E6.

Provides the interface that processors and the job runner will use.
All methods raise NotImplementedError until E6 fills them in.
"""

from __future__ import annotations


class DoclingClient:
    """HTTP client for Docling-serve (stub — implemented in E6)."""

    def __init__(self, base_url: str = "http://localhost:5001") -> None:
        self.base_url = base_url

    async def submit(self, data: bytes, filename: str, mode: str = "hybrid") -> str:
        """Submit a document for processing. Returns a task_id.

        Args:
            data:     Raw file bytes.
            filename: Original filename (used by Docling to infer format).
            mode:     Chunking mode: "hybrid", "hierarchical", or "markdown".

        Returns:
            task_id string to poll with _await_task().
        """
        raise NotImplementedError("DoclingClient not yet implemented — PR E6")

    async def _await_task(self, task_id: str, poll_interval: float = 1.0) -> dict[str, object]:
        """Poll until task completes. Returns the result dict.

        Raises:
            RuntimeError: If the task fails.
        """
        raise NotImplementedError("DoclingClient not yet implemented — PR E6")

    async def clear_results(self) -> None:
        """Delete completed task results from Docling's result cache."""
        raise NotImplementedError("DoclingClient not yet implemented — PR E6")
