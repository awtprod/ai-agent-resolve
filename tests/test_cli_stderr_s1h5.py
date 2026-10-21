"""S1h5 item 5: a usage error says the same thing on stderr whatever state stdout is in.

``finalize`` only adds its one-line ``aar: stdout is closed; nothing written`` (or ``cannot write
output: ...``) to a *success*: there the line is the only way to learn why the exit code is 74.
A command that already failed (exit 2) has its own message, which is all the caller needs, and the
documented rule is that a non-zero code wins without comment. The S1h4 tests pinned the exit code
but not that stderr stays clean, so a mutant that printed the extra line for failures as well
(``prog if code == 0 else None`` -> ``prog``) survived with exit codes unchanged.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from test_cli_exit_codes_s1h4 import HELP_EXIT, USAGE_ARGV, run

pytestmark = pytest.mark.skipif(not Path("/dev/full").exists(), reason="needs /dev/full (Linux)")

# lines `finalize` / `emit` write for a failed or closed stdout; none may follow a usage error
EXTRA = ("stdout is closed", "cannot write output", "nothing written", "No space left")


@pytest.mark.parametrize("stdout", [s for s in HELP_EXIT if s != "ok"])
@pytest.mark.parametrize("argv", USAGE_ARGV)
def test_a_usage_error_prints_the_same_stderr_in_every_stdout_state(
    argv: list[str], stdout: str
) -> None:
    ok_code, ok_err = run(argv, stdout="ok")
    code, err = run(argv, stdout=stdout)
    assert (ok_code, code) == (2, 2)
    text = err.decode()
    assert text.startswith("usage: aar"), text
    assert text.rstrip("\n").splitlines()[-1].split(": error: ")[0].startswith("aar"), text
    assert not any(extra in text for extra in EXTRA), text
    # exactly what argparse prints when stdout is fine: nothing added, nothing dropped
    assert err == ok_err


@pytest.mark.parametrize("stdout", [s for s in HELP_EXIT if s != "ok"])
def test_a_drp_diff_input_error_prints_the_same_stderr_in_every_stdout_state(
    tmp_path: Path, stdout: str
) -> None:
    # the other exit-2 path: not a usage error but an unreadable input, through drp's own helpers
    missing = str(tmp_path / "missing.drp")
    argv = ["drp", "diff", missing, missing]
    ok_code, ok_err = run(argv, stdout="ok")
    code, err = run(argv, stdout=stdout)
    assert (ok_code, code) == (2, 2), err[:200]
    assert err.decode() == f"aar drp diff: cannot read {missing}: " + (
        f"[Errno 2] No such file or directory: '{missing}'\n"
    )
    assert err == ok_err
