"""S13f2 F4/F5: _place_no_hardlink / _unlink_if_claim -- base exceptions, claims that are not
provably ours are kept and reported, the os.fstat-failure case, and the ctime_ns identity field."""

from __future__ import annotations

import errno
import os
from types import SimpleNamespace

import pytest
from host_support import FakeRunner, make_config, sha256_upper

from aar.host import transport
from aar.host.errors import LocalIOError
from aar.host.transport import EditHost, _place_no_hardlink, _unlink_if_claim

PAYLOAD = b"verified payload"
REMOTE = "E:\\Show\\cut.mp4"


def _no_link(monkeypatch):
    monkeypatch.setattr(
        transport.os, "link", lambda *a, **k: (_ for _ in ()).throw(OSError(errno.EPERM, "x"))
    )


def _host(tmp_path):
    runner = FakeRunner(stat_result=(len(PAYLOAD), sha256_upper(PAYLOAD)), scp_payload=PAYLOAD)
    return EditHost(make_config(tmp_path), runner, token_factory=lambda: "T")


def _claim(path):
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        return os.fstat(fd)
    finally:
        os.close(fd)


class _Custom(BaseException):
    pass


@pytest.mark.parametrize("exc", [KeyboardInterrupt, SystemExit, GeneratorExit, _Custom])
def test_any_base_exception_during_replace_removes_our_claim(tmp_path, monkeypatch, exc) -> None:
    _no_link(monkeypatch)
    monkeypatch.setattr(transport.os, "replace", lambda *a, **k: (_ for _ in ()).throw(exc()))
    with pytest.raises(exc):
        _host(tmp_path).pull(REMOTE, str(tmp_path / "out.mp4"))
    assert not (tmp_path / "out.mp4").exists()
    assert not [p for p in tmp_path.iterdir() if ".aarpart-" in p.name]


def _replace_then(monkeypatch, tmp_path, mutate):
    real_replace = os.replace
    dest = tmp_path / "out.mp4"

    def replace(src, dst):
        if str(dst) == str(dest):
            mutate(dest)
            raise OSError(errno.EIO, "io error")
        return real_replace(src, dst)

    monkeypatch.setattr(transport.os, "replace", replace)
    return dest


def test_claim_written_into_is_kept_and_reported(tmp_path, monkeypatch) -> None:
    _no_link(monkeypatch)
    dest = _replace_then(monkeypatch, tmp_path, lambda d: d.write_bytes(b"someone wrote here"))
    with pytest.raises(LocalIOError) as info:
        _host(tmp_path).pull(REMOTE, str(dest))
    assert dest.exists() and dest.read_bytes() == b"someone wrote here"
    assert str(dest) in str(info.value) or "left" in str(info.value)


def test_claim_replaced_by_symlink_is_kept(tmp_path, monkeypatch) -> None:
    _no_link(monkeypatch)
    dest = _replace_then(
        monkeypatch, tmp_path, lambda d: (os.unlink(d), os.symlink(str(d) + ".target", d))
    )
    with pytest.raises(LocalIOError):
        _host(tmp_path).pull(REMOTE, str(dest))
    assert os.path.lexists(dest)  # the planted symlink was not deleted


def test_lstat_failure_keeps_and_reports_the_claim(tmp_path, monkeypatch) -> None:
    _no_link(monkeypatch)
    dest = _replace_then(monkeypatch, tmp_path, lambda d: None)
    real_lstat = os.lstat
    monkeypatch.setattr(
        transport.os,
        "lstat",
        lambda p, *a, **k: (
            (_ for _ in ()).throw(PermissionError(errno.EACCES, "denied"))
            if str(p) == str(dest)
            else real_lstat(p, *a, **k)
        ),
    )
    with pytest.raises(LocalIOError) as info:
        _host(tmp_path).pull(REMOTE, str(dest))
    assert dest.exists() and dest.stat().st_size == 0
    assert str(dest) in str(info.value) or "left" in str(info.value)


def test_fstat_failure_leaves_a_reported_claim(tmp_path, monkeypatch) -> None:
    _no_link(monkeypatch)
    monkeypatch.setattr(
        transport.os, "fstat", lambda *a, **k: (_ for _ in ()).throw(OSError(errno.EIO, "io"))
    )
    dest = tmp_path / "out.mp4"
    with pytest.raises(LocalIOError) as info:
        _host(tmp_path).pull(REMOTE, str(dest))
    assert not dest.exists() or str(dest) in str(info.value)


def test_ctime_mismatch_alone_keeps_the_file(tmp_path) -> None:
    # F5: same dev+ino+size0 but a different ctime_ns is not ours.
    p = tmp_path / "c"
    real = _claim(p)
    fake = SimpleNamespace(st_dev=real.st_dev, st_ino=real.st_ino, st_ctime_ns=real.st_ctime_ns + 1)
    assert _unlink_if_claim(str(p), fake) is False
    assert p.exists()
    assert _unlink_if_claim(str(p), real) is True
    assert not p.exists()


def test_place_no_hardlink_success_and_existing(tmp_path) -> None:
    tmp = tmp_path / "t"
    tmp.write_bytes(b"x")
    dest = tmp_path / "d"
    _place_no_hardlink(str(tmp), str(dest))
    assert dest.read_bytes() == b"x" and not tmp.exists()
    tmp.write_bytes(b"y")
    with pytest.raises(Exception):  # noqa: B017  DestinationExists (no clobber)
        _place_no_hardlink(str(tmp), str(dest))
    assert dest.read_bytes() == b"x"


def test_own_failed_replace_keeps_claim_identity(tmp_path) -> None:
    # our own fstat->close->failed replace must not change (dev, ino, ctime_ns), so the claim is
    # still recognised and removed.
    dest = tmp_path / "d.mp4"
    claim = _claim(dest)
    missing = tmp_path / "nope"
    with pytest.raises(OSError):
        os.replace(missing, dest)  # ENOENT: our own failed replace
    now = os.lstat(dest)
    assert (now.st_dev, now.st_ino, now.st_ctime_ns) == (
        claim.st_dev,
        claim.st_ino,
        claim.st_ctime_ns,
    )
    assert _unlink_if_claim(str(dest), claim) is True
    assert not dest.exists()
