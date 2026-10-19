"""S1h4 item 3: pin the surviving cli_io/plan mutants (N25, N27, N30).

Each asserts the current behaviour through the CLI; the validator's mutant changes it, so the
test fails on the mutant and passes here. Mutant ids match aar-evidence/s1h3/probes/mut.py.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

CLI = "import sys; from aar.cli import main; sys.exit(main(sys.argv[1:]))"
FIX = Path(__file__).resolve().parent / "fixtures"
BROKEN = FIX / "short_broken.plan.json"


def _run(argv: list[str], *, stdout, stderr, env_extra: dict[str, str] | None = None):  # type: ignore[no-untyped-def]
    return subprocess.run(
        [sys.executable, "-c", CLI, *argv],
        stdout=stdout,
        stderr=stderr,
        env={**os.environ, **(env_extra or {})},
        check=False,
    )


@pytest.mark.skipif(not Path("/dev/full").exists(), reason="needs /dev/full (Linux)")
def test_n25_missing_plan_with_full_stderr_exits_2_not_120(tmp_path: Path) -> None:
    with open("/dev/full", "wb") as full:
        proc = _run(
            ["plan", "lint", str(tmp_path / "nope.json")], stdout=subprocess.DEVNULL, stderr=full
        )
    assert proc.returncode == 2  # the input error; the stderr failure must not turn it into 120


@pytest.mark.skipif(not Path("/dev/full").exists(), reason="needs /dev/full (Linux)")
def test_n30_lint_to_full_stdout_exits_74(tmp_path: Path) -> None:
    with open("/dev/full", "wb") as full:
        proc = _run(["plan", "lint", str(BROKEN)], stdout=full, stderr=subprocess.PIPE)
    assert proc.returncode == 74  # the output error overrides the lint's own exit code (1)
    assert b"Traceback" not in proc.stderr


def test_n27_non_ascii_id_under_ascii_strict_exits_1(tmp_path: Path) -> None:
    # A plan that lints to error findings, with a non-ASCII source id that appears in the
    # findings. Under PYTHONIOENCODING=ascii:strict the id must be escaped (real exit 1), not
    # turned into an encode failure (74).
    plan = json.loads(BROKEN.read_text(encoding="utf-8"))
    bad_id = "cám-a"  # "cám-a"
    for src in plan["sources"]:
        if src["id"] == "cam-a":
            src["id"] = bad_id
    for seg in plan["segments"]:
        if seg["source"] == "cam-a":
            seg["source"] = bad_id
    p = tmp_path / "nonascii.plan.json"
    p.write_text(json.dumps(plan), encoding="utf-8")
    proc = _run(
        ["plan", "lint", str(p)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env_extra={"PYTHONIOENCODING": "ascii:strict"},
    )
    assert proc.returncode == 1, (proc.returncode, proc.stderr[:200])
    assert b"\\xe1" in proc.stdout or b"\\u00e1" in proc.stdout  # escaped, not raw
