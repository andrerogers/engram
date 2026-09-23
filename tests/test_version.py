"""Engram's version is derived, not typed.

This file existed because three hardcoded versions in `app.py` disagreed: `setup_optics` said
0.0.1, the FastAPI app said 0.2.0, and `pyproject.toml` said 0.4.0. Nothing compared them, so
telemetry, the OpenAPI document and the package each claimed a different Engram was running.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from engram.app import app
from engram.version import service_version


def test_the_version_is_the_one_in_the_project_file():
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    with pyproject.open("rb") as fh:
        declared = tomllib.load(fh)["project"]["version"]
    assert service_version() == declared


def test_the_openapi_document_reports_the_same_version():
    """The handbook's API reference renders this string, so a literal here would put a wrong
    version on a published page."""
    assert app.openapi()["info"]["version"] == service_version()


def test_it_never_reports_unknown_in_a_working_checkout():
    assert service_version() != "unknown"
