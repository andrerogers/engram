"""The suite must never write into the developer's real BrainStack home.

Isolating ENGRAM_DB_PATH alone was not enough: the object store defaults to
<BRAINSTACK_HOME>/objects, so a test that stored an object wrote it into the real ~/.brainstack.
"""

from __future__ import annotations

import os
from pathlib import Path

from engram.config import BRAINSTACK_HOME, ENGRAM_DB_PATH, ENGRAM_OBJECT_DIR


def _real_home() -> Path:
    return Path.home() / ".brainstack"


def test_brainstack_home_is_redirected() -> None:
    assert os.environ.get("BRAINSTACK_HOME"), "conftest must set BRAINSTACK_HOME before imports"
    assert not BRAINSTACK_HOME.is_relative_to(_real_home())


def test_database_is_redirected() -> None:
    assert not ENGRAM_DB_PATH.is_relative_to(_real_home())


def test_object_store_stays_out_of_the_real_home() -> None:
    assert not ENGRAM_OBJECT_DIR.is_relative_to(_real_home())
