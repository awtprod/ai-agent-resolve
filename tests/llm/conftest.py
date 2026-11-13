"""The gateway tests reuse the loopback API stub of ``tests/analyze`` (one stub, one key-leak
check), so that folder goes on ``sys.path`` as it does when pytest collects it."""

from __future__ import annotations

import sys
from pathlib import Path

_ANALYZE = str(Path(__file__).resolve().parent.parent / "analyze")
if _ANALYZE not in sys.path:
    sys.path.insert(0, _ANALYZE)
