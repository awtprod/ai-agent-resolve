"""A Windows node has no [hub] extra: aar.node remote modules import without it (S15a item 5).

Runs a fresh interpreter with starlette, uvicorn, mcp and fcntl blocked at import time.
"""

from __future__ import annotations

import subprocess
import sys


def test_remote_modules_import_without_hub_extra_or_fcntl() -> None:
    code = (
        "import builtins\n"
        "_real = builtins.__import__\n"
        "def guard(name, *a, **k):\n"
        "    if name.split('.')[0] in ('starlette', 'uvicorn', 'mcp', 'fcntl'):\n"
        "        raise ImportError('blocked: ' + name)\n"
        "    return _real(name, *a, **k)\n"
        "builtins.__import__ = guard\n"
        "import aar.node.remote\n"
        "import aar.node.hublink\n"
        "import aar.node.feed\n"
        "assert aar.node.remote.RemoteHubLink and aar.node.remote.RemoteFeedTransport\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout
