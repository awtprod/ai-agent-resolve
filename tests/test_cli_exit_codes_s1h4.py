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
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

import pytest

from aar.hub.artifacts import ArtifactStore
from aar.hub.events import Actor, EventLog
from aar.hub.graph import WorkGraph
from aar.hub.registry import Registry
from aar.hub.reviews import request_review

CLI = "import sys; from aar.cli import main; sys.exit(main(sys.argv[1:]))"
# `aar hub console` would bind a port: the child replaces uvicorn.run with a no-op first.
CLI_NO_SERVE = "import sys, uvicorn; uvicorn.run = lambda *a, **k: None; " + CLI
HUMAN = Actor("human", "operator-1")
AGENT = Actor("agent", "agent-1")
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


def run(
    argv: list[str], stdout: str = "ok", stderr: str = "capture", *, program: str = CLI
) -> tuple[int, bytes]:
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
        [sys.executable, "-c", program, *argv],
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


# ---------------------------------------------------------------- the other hub subcommands
# Round 2 L4: H02, H03, H04 and H06 (a review command that ignores the result of printing),
# H08 and H09 (`hub serve` printing its refusals bare) and H10 (the console's read-only
# notice printed bare) survived the round-1 suite, which only drove `review list`.


@dataclass
class SeededHub:
    db: str
    artifacts: str
    project_id: str
    subject_id: str
    render_id: str
    review_id: str


@pytest.fixture
def seeded(tmp_path: Path) -> Iterator[SeededHub]:
    """A hub on disk with one project, one node under review and one open review."""
    db = tmp_path / "hub.db"
    log = EventLog(db)
    project = Registry(log).create_project("demo-show", HUMAN)
    subject = WorkGraph(log, project.id).add_node("op", AGENT, title="assemble first cut")
    store = ArtifactStore(tmp_path / "artifacts", log)
    render_id = store.put(b"render-bytes", actor=AGENT, mime="video/mp4", project_id=project.id)
    review = request_review(
        log, store, project.id, subject.id, [render_id], AGENT, title="first cut", fps=24.0
    )
    yield SeededHub(
        str(db), str(tmp_path / "artifacts"), project.id, subject.id, render_id, review.id
    )
    log.close()


def _review_argv(hub: SeededHub, command: str) -> list[str]:
    base = ["hub", "review", command, "--db", hub.db, "--project", hub.project_id]
    if command == "request":
        return [
            *base,
            *("--artifacts", hub.artifacts, "--subject", hub.subject_id),
            *("--render", hub.render_id, "--title", "second cut", "--actor", "agent:agent-1"),
        ]
    if command == "note":
        return [
            *base,
            *("--review", hub.review_id, "--frame", "48", "--text", "trim"),
            *("--actor", "agent:agent-1"),
        ]
    if command == "withdraw":
        return [
            *base,
            *("--review", hub.review_id, "--reason", "superseded", "--actor", "agent:agent-1"),
        ]
    assert command == "show"
    return [*base, hub.review_id]


@pytest.mark.parametrize("stdout", HELP_EXIT)
@pytest.mark.parametrize("command", ["request", "note", "withdraw", "show"])
def test_each_review_command_reports_the_state_of_stdout(
    seeded: SeededHub, command: str, stdout: str
) -> None:
    # H02 (request), H03 (note), H04 (withdraw), H06 (show): each returned 0 whatever happened
    # to its output, so a full or closed stdout looked like success.
    code, err = run(_review_argv(seeded, command), stdout=stdout)
    assert code == HELP_EXIT[stdout], (code, err[:300])
    assert_clean(code, err)


@pytest.mark.parametrize("stderr", ["capture", "full", "closed"])
def test_a_refused_remote_bind_is_a_clean_exit_2_in_every_stderr_state(
    seeded: SeededHub, stderr: str
) -> None:
    # H08: the refusal was a bare print to stderr, which on a full stderr ended in exit 120.
    argv = ["hub", "serve", "--db", seeded.db, "--artifacts", seeded.artifacts]
    code, err = run([*argv, "--host", "0.0.0.0"], stderr=stderr)
    assert code == 2, (code, err[:300])
    assert_clean(code, err)
    if stderr == "capture":
        assert err.decode().startswith("refusing to bind to '0.0.0.0'"), err


@pytest.mark.parametrize("stderr", ["capture", "full", "closed"])
def test_serve_with_a_missing_database_is_a_clean_exit_2_in_every_stderr_state(
    seeded: SeededHub, tmp_path: Path, stderr: str
) -> None:
    # H09: the bad-database message was a bare print to stderr.
    argv = ["hub", "serve", "--db", str(tmp_path / "nope.db"), "--artifacts", seeded.artifacts]
    code, err = run(argv, stderr=stderr)
    assert code == 2, (code, err[:300])
    assert_clean(code, err)
    if stderr == "capture":
        assert err.decode().startswith("aar hub serve: "), err


@pytest.mark.parametrize("stderr", ["capture", "full", "closed"])
def test_the_read_only_console_notice_never_changes_the_exit_code(
    seeded: SeededHub, stderr: str
) -> None:
    # H10: the notice was a bare print to stderr. The console is started with a stubbed
    # uvicorn.run, so this reaches the line after the notice and returns.
    argv = ["hub", "console", "--db", seeded.db, "--artifacts", seeded.artifacts]
    code, err = run(argv, stderr=stderr, program=CLI_NO_SERVE)
    assert code == 0, (code, err[:300])
    assert_clean(code, err)
    if stderr == "capture":
        assert err.decode() == "console is read-only: no --operator and no trusted headers\n"
