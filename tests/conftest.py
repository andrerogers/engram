"""Suite-wide setup."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

# Before any engram import: no test may touch the developer's real ~/.brainstack — not the
# database, and not the object store, which defaults to <BRAINSTACK_HOME>/objects.
_tmp = Path(tempfile.mkdtemp(prefix="engram-tests-"))
os.environ["BRAINSTACK_HOME"] = str(_tmp / "home")
os.environ["ENGRAM_DB_PATH"] = str(_tmp / "engram.db")
