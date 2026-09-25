"""Suite-wide setup."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

# Before any engram import: no test may touch the developer's real ~/.brainstack — not the
# database, and not the object store, which defaults to <BRAINSTACK_HOME>/objects.
_tmp = Path(tempfile.mkdtemp(prefix="engram-tests-"))
os.environ["BRAINSTACK_HOME"] = str(_tmp / "home")

# No test's telemetry may reach a running collector. Unit tests exported to localhost:4317 under
# the service's own name, so with the stack up they wrote series identical to the live service's
# into Prometheus — and the e2e recall-latency check read the collision as a counter that never
# moved (2026-09-24, confirmed by running Engram's tests alone against an idle stack).
os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "http://127.0.0.1:9"
os.environ["ENGRAM_DB_PATH"] = str(_tmp / "engram.db")
