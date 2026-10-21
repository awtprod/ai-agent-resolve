"""S13f2 F2: typed runner errors, and the original error wins over a cleanup failure.
Fake runner only (real-runner versions are in test_host_s13f2_realrunner.py)."""

from __future__ import annotations

import errno
import os

import pytest
from host_support import ALIAS, FakeRunner, make_config, sha256_upper

from aar.host import transport
from aar.host.errors import HashMismatch, HostError, LocalIOError, SizeMismatch
from aar.host.transport import EditHost

REMOTE = "E:\\Show\\cut.mp4"
PAYLOAD = b"verified payload"


def leftovers(d) -> list[str]:
    return sorted(p.name for p in d.iterdir() if ".aarpart-" in p.name)


class StepRunner(FakeRunner):
    """Raise ``exc()`` on runner call number ``at``; with ``sticky`` every later call raises too.
    A pull scp writes a partial temp first (what a killed scp leaves)."""

    def __init__(self, at, exc, sticky=False, **kw):
        kw.setdefault("stat_result", (len(PAYLOAD), sha256_upper(PAYLOAD)))
        kw.setdefault("scp_payload", PAYLOAD)
        super().__init__(**kw)
        self.at, self.exc, self.sticky = at, exc, sticky
        self.raised = 0

    def run(self, argv, *, stdin=None, timeout=None):
        n = len(self.calls)
        if n == self.at or (self.sticky and n > self.at):
            self.calls.append(list(argv))
            if argv[0] == "scp" and not argv[-1].startswith(f"{ALIAS}:"):
                with open(argv[-1], "wb") as fh:
                    fh.write(PAYLOAD[:3])
            if argv[0] == "ssh":
                from host_support import decode_encoded_command

                self.scripts.append(decode_encoded_command(argv))
            self.raised += 1
            raise self.exc()
        return super().run(argv, stdin=stdin, timeout=timeout)


def kbi():
    return KeyboardInterrupt()


def oserr():
    return OSError(errno.EAGAIN, "Resource temporarily unavailable (fork)")


def sysexit():
    return SystemExit(3)


EXCS = {"KeyboardInterrupt": kbi, "OSError": oserr, "SystemExit": sysexit}


def _host(tmp_path, runner):
    return EditHost(make_config(tmp_path), runner, token_factory=lambda: "T")


@pytest.mark.parametrize("exc", list(EXCS))
@pytest.mark.parametrize("at", [0, 1], ids=["stat", "scp"])
def test_pull_runner_step_leaves_nothing(tmp_path, at, exc) -> None:
    host = _host(tmp_path, StepRunner(at, EXCS[exc]))
    with pytest.raises(BaseException) as info:
        host.pull(REMOTE, str(tmp_path / "out.mp4"))
    assert leftovers(tmp_path) == []
    assert not (tmp_path / "out.mp4").exists()
    if exc == "OSError":  # F2: a runner OSError is typed at the stat pre-check and around scp
        assert isinstance(info.value, LocalIOError)


def test_pull_runner_oserror_at_stat_is_typed(tmp_path) -> None:
    # inverted from the r3 probe: the pre-check OSError is now LocalIOError, not raw.
    host = _host(tmp_path, StepRunner(0, oserr))
    with pytest.raises(LocalIOError):
        host.pull(REMOTE, str(tmp_path / "out.mp4"))


@pytest.mark.parametrize("exc", list(EXCS))
@pytest.mark.parametrize("at", [0, 1, 2, 3], ids=["exists", "scp", "stat", "move"])
def test_push_runner_step(tmp_path, at, exc) -> None:
    src = tmp_path / "master.mov"
    src.write_bytes(PAYLOAD)
    runner = StepRunner(at, EXCS[exc])
    host = _host(tmp_path, runner)
    with pytest.raises(BaseException) as info:
        host.push(str(src), "E:\\Show\\out.mp4")
    removes = [s for s in runner.scripts if "Remove-Item" in s]
    if at == 0:  # failed at the exists pre-check: nothing uploaded, no remote temp
        assert removes == [] and [c[0] for c in runner.calls] == ["ssh"]
    else:
        assert removes and "out.mp4.aarpart-T" in removes[-1]
        if exc == "OSError":
            assert isinstance(info.value, LocalIOError)
        else:
            assert type(info.value).__name__ == exc


@pytest.mark.parametrize("exc", list(EXCS))
def test_push_overwrite_scp_step(tmp_path, exc) -> None:
    # kills N19/N20: the OSError handler and the remote-remove both run on the --overwrite path
    src = tmp_path / "master.mov"
    src.write_bytes(PAYLOAD)
    runner = StepRunner(0, EXCS[exc])
    host = _host(tmp_path, runner)
    with pytest.raises(BaseException) as info:
        host.push(str(src), "E:\\Show\\out.mp4", overwrite=True)
    assert [s for s in runner.scripts if "Remove-Item" in s]
    if exc == "OSError":
        assert isinstance(info.value, LocalIOError)


def test_push_runner_oserror_in_cleanup_does_not_mask_typed_error(tmp_path) -> None:
    src = tmp_path / "master.mov"
    src.write_bytes(PAYLOAD)
    runner = StepRunner(1, oserr, sticky=True)  # scp fails, then the cleanup ssh fails too
    host = _host(tmp_path, runner)
    with pytest.raises(LocalIOError):
        host.push(str(src), "E:\\Show\\out.mp4")
    assert runner.raised == 2  # scp, then the cleanup ssh


def test_push_interrupt_then_cleanup_oserror_keeps_the_interrupt(tmp_path) -> None:
    src = tmp_path / "master.mov"
    src.write_bytes(PAYLOAD)

    class R(StepRunner):
        def run(self, argv, *, stdin=None, timeout=None):
            n = len(self.calls)
            if n == 1:
                self.calls.append(list(argv))
                raise KeyboardInterrupt
            if n == 2:
                self.calls.append(list(argv))
                raise OSError(errno.ENOMEM, "Cannot allocate memory")
            return super().run(argv, stdin=stdin, timeout=timeout)

    with pytest.raises(KeyboardInterrupt):
        _host(tmp_path, R(99, kbi)).push(str(src), "E:\\Show\\out.mp4")


def test_push_hash_mismatch_then_cleanup_oserror_keeps_it(tmp_path) -> None:
    src = tmp_path / "master.mov"
    src.write_bytes(PAYLOAD)

    class R(StepRunner):
        def run(self, argv, *, stdin=None, timeout=None):
            if len(self.calls) == 3:  # 0 exists, 1 scp, 2 stat (mismatch) -> 3 cleanup Remove-Item
                self.calls.append(list(argv))
                raise OSError(errno.EMFILE, "Too many open files")
            return super().run(argv, stdin=stdin, timeout=timeout)

    runner = R(99, kbi, stat_result=(len(PAYLOAD), sha256_upper(b"different")))
    with pytest.raises(HashMismatch):
        _host(tmp_path, runner).push(str(src), "E:\\Show\\out.mp4")


@pytest.mark.parametrize("original", ["SizeMismatch", "KeyboardInterrupt", "LocalIOError"])
def test_pull_cleanup_unlink_error_keeps_the_original(tmp_path, monkeypatch, original) -> None:
    real_unlink = os.unlink

    def unlink(path, *a, **k):
        if ".aarpart-" in str(path):
            raise PermissionError(errno.EACCES, "denied")
        return real_unlink(path, *a, **k)

    if original == "SizeMismatch":
        runner = StepRunner(99, kbi, stat_result=(len(PAYLOAD) + 1, sha256_upper(PAYLOAD)))
    elif original == "KeyboardInterrupt":
        runner = StepRunner(1, kbi)
    else:
        runner = StepRunner(1, oserr)
    monkeypatch.setattr(transport.os, "unlink", unlink)
    want = {
        "SizeMismatch": SizeMismatch,
        "KeyboardInterrupt": KeyboardInterrupt,
        "LocalIOError": LocalIOError,
    }[original]
    with pytest.raises(want):
        _host(tmp_path, runner).pull(REMOTE, str(tmp_path / "out.mp4"))


def test_cleanup_failure_is_noted_on_the_original(tmp_path, monkeypatch) -> None:
    # F2: a swallowed cleanup error is recorded as a note on the surviving exception.
    real_unlink = os.unlink

    def unlink(path, *a, **k):
        if ".aarpart-" in str(path):
            raise PermissionError(errno.EACCES, "denied")
        return real_unlink(path, *a, **k)

    monkeypatch.setattr(transport.os, "unlink", unlink)
    runner = StepRunner(99, kbi, stat_result=(len(PAYLOAD) + 1, sha256_upper(PAYLOAD)))
    with pytest.raises(SizeMismatch) as info:
        _host(tmp_path, runner).pull(REMOTE, str(tmp_path / "out.mp4"))
    notes = getattr(info.value, "__notes__", [])
    assert any("cleanup" in n for n in notes), notes


def test_push_runner_oserror_is_typed(tmp_path) -> None:
    # kills N19/N20 directly: a push whose scp step raises OSError surfaces a typed HostError.
    assert issubclass(LocalIOError, HostError)
    src = tmp_path / "master.mov"
    src.write_bytes(PAYLOAD)
    runner = StepRunner(1, oserr)
    with pytest.raises(LocalIOError):
        _host(tmp_path, runner).push(str(src), "E:\\Show\\out.mp4")
