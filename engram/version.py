"""Engram's own version, derived rather than typed.

Engram carried three different hardcoded versions in one file: `setup_optics` said `0.0.1`, the
FastAPI app said `0.2.0`, and `pyproject.toml` said `0.4.0`. All three were written by hand at
different times and nothing compared them, so telemetry, the OpenAPI document and the package
disagreed about which Engram was running.

Installed metadata first, the project file second — Engram runs from source in development.
"""

from __future__ import annotations

import tomllib
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

_PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


@lru_cache(maxsize=1)
def service_version() -> str:
    """The running Engram's version. ``"unknown"`` only if neither source can be read."""
    try:
        return version("engram")
    except PackageNotFoundError:
        pass
    try:
        with _PYPROJECT.open("rb") as fh:
            return str(tomllib.load(fh)["project"]["version"])
    except (OSError, KeyError, tomllib.TOMLDecodeError):
        return "unknown"
