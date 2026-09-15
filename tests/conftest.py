"""Suite-wide setup."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

# Before any engram import: no test may open the developer's real ~/.brainstack/engram.db.
os.environ["ENGRAM_DB_PATH"] = str(Path(tempfile.mkdtemp(prefix="engram-tests-")) / "engram.db")
