"""S13f2 real-SubprocessRunner tests with shell stand-ins for ssh/scp in tmp_path (no network, no
host): LOW-6 stdin is /dev/null, MED-1 interrupts end to end, F2 typed error when ssh/scp are
missing, and F3 --timeout validation. Adapted from the validator's test_v3_realrunner."""

from __future__ import annotations

import json
import os
import signal
import sys
import time

import pytest
from host_support import make_config

from aar.cli import main
from aar.host import cli as host_cli
from aar.host.errors import LocalIOError
from aar.host.runner import SubprocessRunner
from aar.host.transport import EditHost

SSH_STANDIN = r"""#!{py}
import base64, json, os, sys
log = os.environ["STANDIN_LOG"]
argv = sys.argv[1:]
rec = {{"prog": "ssh", "argv": argv, "stdin": os.readlink("/proc/self/fd/0")}}
script = ""
if "-EncodedCommand" in argv:
    script = base64.b64decode(argv[argv.index("-EncodedCommand") + 1]).decode("utf-16-le")
rec["script"] = script
with open(log, "a") as fh:
    fh.write(json.dumps(rec) + "\n")
if "Get-FileHash" in script:
    print("AARSTAT 1 " + "A" * 64)
elif "Test-Path" in script and "Move-Item" not in script:
    print("AAREXISTS none")
elif "Remove-Item" in script:
    pass
else:
    print("ok")
"""

SCP_SLEEPER = r"""#!/bin/bash
echo "{\"prog\": \"scp\", \"argv\": \"$*\", \"pid\": $$}" >> "$STANDIN_LOG"
last="${@: -1}"
case "$last" in
  *:*) ;;                       # push: remote destination, nothing local to write
  *) printf 'par' > "$last" ;;  # pull: the partial a real scp leaves in SFTP mode
esac
exec sleep 30
"""


@pytest.fixture
def standins(tmp_path, monkeypatch):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "ssh").write_text(SSH_STANDIN.format(py=sys.executable))
    (bindir / "scp").write_text(SCP_SLEEPER)
    for f in ("ssh", "scp"):
        os.chmod(bindir / f, 0o755)
    log = tmp_path / "standin.log"
    monkeypatch.setenv("STANDIN_LOG", str(log))
    monkeypatch.setenv("PATH", f"{bindir}:/usr/bin:/bin")
    return log


def records(log):
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text().splitlines() if line.strip()]


def _alarm(seconds):
    def handler(signum, frame):
        raise KeyboardInterrupt

    old = signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    return old


def _clear(old):
    signal.setitimer(signal.ITIMER_REAL, 0)
    signal.signal(signal.SIGALRM, old)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    with open(f"/proc/{pid}/stat") as fh:
        return fh.read().split()[2] != "Z"


def test_low6_child_stdin_is_dev_null(tmp_path, standins) -> None:
    # make fd 0 a pipe first, so an inheriting child would show pipe:[...] not /dev/null
    r, w = os.pipe()
    saved = os.dup(0)
    os.dup2(r, 0)
    try:
        host = EditHost(make_config(tmp_path), SubprocessRunner())
        assert host.run_powershell("Get-Date").strip() == "ok"
    finally:
        os.dup2(saved, 0)
        for fd in (saved, r, w):
            os.close(fd)
    assert records(standins)[-1]["stdin"] == "/dev/null"


def test_low6_explicit_stdin_is_a_pipe(tmp_path, standins) -> None:
    res = SubprocessRunner().run(["ssh", "x"], stdin=b"data")
    assert res.returncode == 0
    assert records(standins)[-1]["stdin"].startswith("pipe:")


def test_med1_real_interrupt_during_scp_pull_removes_partial_and_kills_scp(tmp_path, standins):
    host = EditHost(make_config(tmp_path), SubprocessRunner(), token_factory=lambda: "T")
    old = _alarm(1.0)
    t0 = time.monotonic()
    try:
        with pytest.raises(KeyboardInterrupt):
            host.pull("E:\\Show\\x.mp4", str(tmp_path / "out.mp4"))
    finally:
        _clear(old)
    assert not list(tmp_path.glob("*.aarpart-*")), "partial left after a real interrupt"
    scp = [r for r in records(standins) if r["prog"] == "scp"]
    assert scp and not _alive(scp[-1]["pid"]), "scp stand-in still running"
    assert time.monotonic() - t0 < 10


def test_med1_real_interrupt_during_scp_push_issues_remote_remove(tmp_path, standins):
    src = tmp_path / "m.mov"
    src.write_bytes(b"m")
    host = EditHost(make_config(tmp_path), SubprocessRunner(), token_factory=lambda: "T")
    old = _alarm(1.0)
    try:
        with pytest.raises(KeyboardInterrupt):
            host.push(str(src), "E:\\Show\\out.mp4")
    finally:
        _clear(old)
    recs = records(standins)
    removes = [r for r in recs if r["prog"] == "ssh" and "Remove-Item" in r["script"]]
    assert removes and "out.mp4.aarpart-T" in removes[-1]["script"]
    assert [r["prog"] for r in recs] == ["ssh", "scp", "ssh"]


def test_push_overwrite_without_binaries_is_typed(tmp_path, monkeypatch):
    # F2: no ssh/scp on PATH -> scp FileNotFoundError, and the cleanup Remove-Item also fails;
    # the handler must still surface the typed LocalIOError, not a raw FileNotFoundError.
    empty = tmp_path / "empty-bin"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    src = tmp_path / "m.mov"
    src.write_bytes(b"m")
    host = EditHost(make_config(tmp_path), SubprocessRunner())
    with pytest.raises(LocalIOError):
        host.push(str(src), "E:\\Show\\out.mp4", overwrite=True)


def _cli_env(tmp_path, monkeypatch):
    for k, v in {
        "AAR_EDIT_SSH_ALIAS": "edit-pc",
        "AAR_EDIT_SSH_CONFIG": str(tmp_path / "ssh.cfg"),
        "AAR_EDIT_REMOTE_ROOTS": "E:\\Show",
        "AAR_EDIT_LOCAL_ROOTS": str(tmp_path),
    }.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(host_cli, "_default_runner", lambda: SubprocessRunner())


def test_cli_push_without_binaries_exit_one(tmp_path, monkeypatch, capsys) -> None:
    empty = tmp_path / "empty-bin"
    empty.mkdir()
    _cli_env(tmp_path, monkeypatch)
    monkeypatch.setenv("PATH", str(empty))
    src = tmp_path / "m.mov"
    src.write_bytes(b"m")
    code = main(["host", "push", str(src), "E:\\Show\\out.mp4", "--overwrite", "--json"])
    body = json.loads(capsys.readouterr().out)
    assert code == 1 and body["error"]["type"] == "local-io-error"


@pytest.mark.parametrize("raw", ["nan", "inf", "0", "-1", "1e300"])
def test_cli_ps_timeout_bad_values_exit_two(tmp_path, monkeypatch, raw) -> None:
    _cli_env(tmp_path, monkeypatch)
    script = tmp_path / "s.ps1"
    script.write_text("Get-Date\n")
    assert main(["host", "ps", str(script), "--timeout", raw, "--json"]) == 2


def test_cli_ps_timeout_good_value_runs(tmp_path, standins, monkeypatch) -> None:
    _cli_env(tmp_path, monkeypatch)
    # standins put ssh/scp first on PATH; _cli_env's SubprocessRunner uses it
    monkeypatch.setattr(host_cli, "_default_runner", lambda: SubprocessRunner())
    script = tmp_path / "s.ps1"
    script.write_text("Get-Date\n")
    assert main(["host", "ps", str(script), "--timeout", "30"]) == 0
