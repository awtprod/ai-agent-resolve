"""S1h4 item 4: the argparse paths and ``aar hub`` give the same exit codes as ``aar drp``.

A help or version text goes to stdout, so a full (``>/dev/full``) or closed (``>&-``) stdout
gives 74 (with the same one-line reason on stderr as every other command) and a reader that went
away gives 141. A usage error keeps its exit code 2 in every state of both streams: stdout
closed, full or gone (the same decision as a drp input error with a closed stdout, pinned here),
and stderr closed or full as well, including both at once. A failing stderr never changes an
exit code. Nothing may print a traceback or end in the interpreter's own exit 120. Before S1h4
the help paths ended in 120 (full stdout, broken pipe) or a silent 0 (closed stdout); at 410a6b3
a usage error with stderr closed and stdout full or gone gave 74 or 141, because argparse then
prints the usage text to stdout.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import IO, Any

import pytest

from aar.hub.events import EventLog

CLI = "import sys; from aar.cli import main; sys.exit(main(sys.argv[1:]))"
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "drp" / "empty-project.drp"

pytestmark = pytest.mark.skipif(not Path("/dev/full").exists(), reason="needs /dev/full (Linux)")

# stdout state -> exit code for a command whose output is meant for stdout
HELP_EXIT = {"ok": 0, "full": 74, "closed": 74, "pipe-closed": 141}
HELP_ARGV = [
    pytest.param([], id="no-args"),
    pytest.param(["--help"], id="--help"),
    pytest.param(["--version"], id="--version"),
    pytest.param(["drp", "--help"], id="drp--help"),
    pytest.param(["plan", "--help"], id="plan--help"),
    pytest.param(["node", "--help"], id="node--help"),
    pytest.param(["hub", "--help"], id="hub--help"),
]
USAGE_ARGV = [
    pytest.param(["plan"], id="plan-no-subcommand"),
    pytest.param(["hub"], id="hub-no-subcommand"),
    pytest.param(["--nope"], id="unknown-flag"),
]


def run(argv: list[str], stdout: str = "ok", stderr: str = "capture") -> tuple[int, bytes]:
    """Run ``aar argv`` with stdout in one state and stderr in another; return (code, stderr)."""
    opened: list[IO[bytes]] = []
    kw: dict[str, Any] = {}
    if stdout == "full":
        opened.append(open("/dev/full", "wb"))  # noqa: SIM115 -- closed below
        kw["stdout"] = opened[-1]
    elif stdout == "pipe-closed":
        kw["stdout"] = subprocess.PIPE
    else:  # "ok" or "closed"
        kw["stdout"] = subprocess.DEVNULL
    if stderr == "full":
        opened.append(open("/dev/full", "wb"))  # noqa: SIM115
        kw["stderr"] = opened[-1]
    elif stderr == "capture":
        kw["stderr"] = subprocess.PIPE
    else:  # "closed"
        kw["stderr"] = subprocess.DEVNULL

    def close_fds() -> None:
        if stdout == "closed":
            os.close(1)
        if stderr == "closed":
            os.close(2)

    env = {k: v for k, v in os.environ.items() if k != "AAR_LIVE"}
    proc = subprocess.Popen(
        [sys.executable, "-c", CLI, *argv],
        preexec_fn=close_fds if "closed" in (stdout, stderr) else None,
        env=env,
        **kw,
    )
    try:
        if stdout == "pipe-closed":
            assert proc.stdout is not None
            proc.stdout.close()  # the reader goes away before the child writes
        _, err = proc.communicate(timeout=60)
    finally:
        for f in opened:
            f.close()
    return proc.returncode, err or b""


def assert_clean(code: int, err: bytes) -> None:
    assert code != 120, "the interpreter's own exit-120 flush failure"
    assert b"Traceback" not in err and b"Exception ignored" not in err, err[:300]


@pytest.mark.parametrize("stdout", HELP_EXIT)
@pytest.mark.parametrize("argv", HELP_ARGV)
def test_help_and_version_report_the_state_of_stdout(argv: list[str], stdout: str) -> None:
    code, err = run(argv, stdout=stdout)
    assert code == HELP_EXIT[stdout], (code, err[:300])
    assert_clean(code, err)


@pytest.mark.parametrize("stdout", HELP_EXIT)
@pytest.mark.parametrize("argv", USAGE_ARGV)
def test_usage_errors_keep_exit_2_whatever_stdout_is(argv: list[str], stdout: str) -> None:
    code, err = run(argv, stdout=stdout)
    assert code == 2, (code, err[:300])
    assert_clean(code, err)


@pytest.mark.parametrize("stdout", HELP_EXIT)
@pytest.mark.parametrize("stderr", ["full", "closed"])
@pytest.mark.parametrize("argv", USAGE_ARGV)
def test_usage_errors_keep_exit_2_with_a_failing_stderr(
    argv: list[str], stderr: str, stdout: str
) -> None:
    # Fails at 410a6b3 for stderr=closed with stdout full (74) or pipe-closed (141): argparse
    # prints the usage text to stdout when sys.stderr is None, and that flush failed the run.
    code, err = run(argv, stdout=stdout, stderr=stderr)
    assert code == 2, (code, err[:300])
    assert_clean(code, err)


REASON = {"full": "cannot write output: [Errno 28] No space left on device"}


@pytest.mark.parametrize("argv", HELP_ARGV)
def test_help_and_version_to_a_full_stdout_say_why_on_stderr(argv: list[str]) -> None:
    # Fails at 410a6b3: exit 74 with nothing on stderr (L2). The line is the one every other
    # command prints, with the program name first.
    code, err = run(argv, stdout="full")
    assert code == 74
    assert err.decode() == f"aar: {REASON['full']}\n"


@pytest.mark.parametrize("argv", HELP_ARGV)
def test_help_and_version_to_a_closed_stdout_say_so_on_stderr(argv: list[str]) -> None:
    code, err = run(argv, stdout="closed")
    assert code == 74
    assert err.decode().endswith("aar: stdout is closed; nothing written\n"), err[-200:]


@pytest.mark.parametrize("argv", HELP_ARGV)
def test_help_and_version_to_a_gone_reader_are_quiet(argv: list[str]) -> None:
    code, err = run(argv, stdout="pipe-closed")
    assert (code, err) == (141, b"")


def test_the_full_stdout_line_matches_the_one_other_commands_print() -> None:
    code, err = run(["drp", "inspect", str(FIXTURE)], stdout="full")
    assert code == 74
    assert err.decode().strip().endswith(REASON["full"]), err


@pytest.mark.parametrize("stderr", ["full", "closed"])
def test_help_with_a_failing_stderr_still_exits_0(stderr: str) -> None:
    code, err = run(["--help"], stderr=stderr)
    assert code == 0, (code, err[:300])
    assert_clean(code, err)


@pytest.fixture
def hub_db(tmp_path: Path) -> str:
    path = str(tmp_path / "hub.db")
    EventLog(path).close()
    return path


@pytest.mark.parametrize("stdout", HELP_EXIT)
def test_hub_listing_reports_the_state_of_stdout(hub_db: str, stdout: str) -> None:
    code, err = run(["hub", "review", "list", "--db", hub_db], stdout=stdout)
    assert code == HELP_EXIT[stdout], (code, err[:300])
    assert_clean(code, err)


@pytest.mark.parametrize("stderr", ["capture", "full", "closed"])
@pytest.mark.parametrize("stdout", HELP_EXIT)
def test_hub_missing_database_exits_2_in_every_stream_state(
    tmp_path: Path, stdout: str, stderr: str
) -> None:
    code, err = run(
        ["hub", "review", "list", "--db", str(tmp_path / "nope.db")], stdout=stdout, stderr=stderr
    )
    assert code == 2, (code, err[:300])
    assert_clean(code, err)
