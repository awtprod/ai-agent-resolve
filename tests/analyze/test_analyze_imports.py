"""``import aar``, ``aar.cli`` and ``aar.analyze`` work with numpy, faster-whisper and OpenCV
absent, and a command that needs one fails with a typed ``missing-extra`` error naming the
extra."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROBE = r"""
import io, json, sys
from contextlib import redirect_stdout

sys.modules["numpy"] = None  # import numpy now raises ImportError
sys.modules["faster_whisper"] = None
sys.modules["cv2"] = None

import aar, aar.cli, aar.analyze, aar.analyze.audio, aar.analyze.cli, aar.analyze.sync
import aar.analyze.whisper, aar.analyze.faces, aar.analyze.common, aar.qa.cli
from aar.cli import main

loaded = sorted(m for m, mod in sys.modules.items() if mod is not None
                and m.split(".")[0] in ("numpy", "faster_whisper", "ctranslate2", "cv2"))
results = {"loaded": loaded}
media = sys.argv[1]
for name, argv in {
    "sync": ["analyze", "sync", "--ref", media, "--other", media, "--fps", "25"],
    "transcript": ["analyze", "transcript", media, "--out", media + ".json"],
    "faces": ["analyze", "faces", media, "--out", media + ".f.json", "--model", media],
}.items():
    out = io.StringIO()
    with redirect_stdout(out):
        code = main(argv)
    results[name] = {"code": code, "body": json.loads(out.getvalue())}
print(json.dumps(results))
"""


def test_imports_and_typed_errors_without_numpy_or_faster_whisper(tmp_path: Path) -> None:
    media = tmp_path / "x.wav"
    media.write_bytes(b"")
    done = subprocess.run(
        [sys.executable, "-c", PROBE, str(media)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    results = json.loads(done.stdout)
    assert results["loaded"] == []
    for name, extra in (("sync", "analyze"), ("transcript", "whisper"), ("faces", "faces")):
        code, body = results[name]["code"], results[name]["body"]
        assert code == 2, name
        assert body["ok"] is False
        assert body["error"]["type"] == "missing-extra"
        assert body["error"]["hint"] == f"pip install 'ai-agent-resolve[{extra}]'"
