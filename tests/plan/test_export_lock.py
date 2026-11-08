"""aar exports to one destination are serialized by exclusive locks on ``<name>.aar-lock`` beside
each destination, the FCPXML's and the report's, taken in sorted order (the node job store's
portable helper), held for the whole pair write; a second export fails at
once with ``output_locked`` and touches nothing. Prism (PR #13) found the third check-then-act
race in the rollback; a lock, not more checks, closes it for aar writers."""

from __future__ import annotations

import os
import runpy
import subprocess
import sys
import threading
import types
from pathlib import Path
from typing import Any

import pytest
from export_support import GOLDEN_PLANS

import aar.plan.export.files as files
from aar.cli import main
from aar.node.jobs import file_lock
from aar.plan.export import ExportError
from aar.plan.export.files import lock_path, write_pair

WAIT_SECONDS = 20.0


def _pair(folder: Path, tag: str) -> list[tuple[Path, bytes]]:
    return [
        (folder / "cut.fcpxml", f"<{tag} fcpxml/>".encode()),
        (folder / "cut.report.json", f"{{{tag}}}".encode()),
    ]


def _listing(folder: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(folder.iterdir())}


class _Held:
    """An export (in a thread) held inside its first placement until released."""

    def __init__(
        self,
        folder: Path,
        monkeypatch: pytest.MonkeyPatch,
        tag: str = "first",
        pair: list[tuple[Path, bytes]] | None = None,
    ) -> None:
        self.inside = threading.Event()
        self.go = threading.Event()
        self.error: BaseException | None = None
        real = files._place_one
        owner = threading.current_thread()

        def held(tmp: Path, dest: Path, overwrite: bool) -> None:
            if threading.current_thread() is not owner and not self.inside.is_set():
                self.inside.set()
                assert self.go.wait(WAIT_SECONDS)
            real(tmp, dest, overwrite)

        monkeypatch.setattr(files, "_place_one", held)

        def run() -> None:
            try:
                write_pair(pair or _pair(folder, tag), overwrite=True)
            except BaseException as exc:  # reported to the test
                self.error = exc

        self.thread = threading.Thread(target=run)
        self.thread.start()
        assert self.inside.wait(WAIT_SECONDS), "the first export never reached its placement"

    def finish(self) -> None:
        self.go.set()
        self.thread.join(WAIT_SECONDS)
        assert not self.thread.is_alive()
        if self.error is not None:
            raise self.error


def test_two_concurrent_exports_one_wins_the_other_is_locked_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    held = _Held(tmp_path, monkeypatch)
    before = _listing(tmp_path)  # the first export's temporaries, its lock, nothing placed yet
    with pytest.raises(ExportError) as caught:
        write_pair(_pair(tmp_path, "second"), overwrite=True)
    assert caught.value.code == "output_locked"
    assert caught.value.details["lock"] == str(lock_path(tmp_path / "cut.fcpxml"))
    assert _listing(tmp_path) == before  # the loser touched nothing
    held.finish()
    after = _listing(tmp_path)
    assert after.pop("cut.fcpxml") == b"<first fcpxml/>"  # the winner's pair, intact
    assert after.pop("cut.report.json") == b"{first}"
    # The locks stay; no temporary is left.
    assert after == {"cut.fcpxml.aar-lock": b"", "cut.report.json.aar-lock": b""}


def test_exports_sharing_only_the_report_name_are_serialized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fix-check Q1: x.fcpxml and x.xml both write x.report.json. The lock covers the report
    too, so the second export is locked out and the winner's FCPXML keeps its own report."""
    report = tmp_path / "x.report.json"
    first = [(tmp_path / "x.fcpxml", b"<from fcpxml/>"), (report, b"{from fcpxml}")]
    second = [(tmp_path / "x.xml", b"<from xml/>"), (report, b"{from xml}")]
    held = _Held(tmp_path, monkeypatch, pair=first)
    before = _listing(tmp_path)
    with pytest.raises(ExportError) as caught:
        write_pair(second, overwrite=True)
    assert caught.value.code == "output_locked"
    assert caught.value.details["lock"] == str(tmp_path / "x.report.json.aar-lock")
    assert _listing(tmp_path) == before  # touched nothing (x.xml.aar-lock is free again)
    held.finish()
    assert (tmp_path / "x.fcpxml").read_bytes() == b"<from fcpxml/>"
    assert report.read_bytes() == b"{from fcpxml}"
    assert not (tmp_path / "x.xml").exists()
    # Afterwards the other one runs normally.
    write_pair(second, overwrite=True)
    assert report.read_bytes() == b"{from xml}"


def test_the_lock_is_released_after_success_and_after_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock = str(lock_path(tmp_path / "cut.fcpxml"))
    write_pair(_pair(tmp_path, "one"), overwrite=False)
    file_lock.release_file_lock(file_lock.acquire_file_lock(lock))  # free after success

    def failing(tmp: Path, dest: Path, overwrite: bool) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(files, "_place_one", failing)
    with pytest.raises(OSError, match="No space left"):
        write_pair(_pair(tmp_path, "two"), overwrite=True)
    file_lock.release_file_lock(file_lock.acquire_file_lock(lock))  # free after failure
    monkeypatch.undo()
    write_pair(_pair(tmp_path, "three"), overwrite=True)
    assert (tmp_path / "cut.fcpxml").read_bytes() == b"<three fcpxml/>"
    assert Path(lock).exists()


def test_a_refused_export_releases_the_lock_too(tmp_path: Path) -> None:
    (tmp_path / "cut.fcpxml").write_bytes(b"<old/>")
    with pytest.raises(ExportError) as caught:
        write_pair(_pair(tmp_path, "new"), overwrite=False)
    assert caught.value.code == "output_exists"
    lock = str(lock_path(tmp_path / "cut.fcpxml"))
    file_lock.release_file_lock(file_lock.acquire_file_lock(lock))


_HOLDER = """
import sys
from aar.node.jobs.file_lock import acquire_file_lock, release_file_lock
fd = acquire_file_lock(sys.argv[1])
print("locked", flush=True)
sys.stdin.read()
release_file_lock(fd)
"""


def test_another_process_holding_the_lock_locks_the_export_out(tmp_path: Path) -> None:
    """A second aar process (here a holder of the same lock) serializes against this one."""
    lock = lock_path(tmp_path / "cut.fcpxml")
    holder = subprocess.Popen(
        [sys.executable, "-c", _HOLDER, str(lock)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout is not None and holder.stdout.readline().strip() == "locked"
        with pytest.raises(ExportError) as caught:
            write_pair(_pair(tmp_path, "mine"), overwrite=False)
        assert caught.value.code == "output_locked"
        assert sorted(p.name for p in tmp_path.iterdir()) == ["cut.fcpxml.aar-lock"]
    finally:
        assert holder.stdin is not None
        holder.stdin.close()
        holder.wait(WAIT_SECONDS)
    assert holder.returncode == 0
    write_pair(_pair(tmp_path, "mine"), overwrite=False)
    assert (tmp_path / "cut.report.json").read_bytes() == b"{mine}"


def test_the_cli_exits_2_with_output_locked(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "cut.fcpxml"
    fd = file_lock.acquire_file_lock(str(lock_path(out)))
    try:
        argv = ["plan", "export", str(GOLDEN_PLANS["ccn_longform_v2"]), "--out", str(out)]
        assert main(argv) == 2
        err = capsys.readouterr().err
        assert "another aar export holds the lock" in err
        assert "or the file system has no file locks" in err
        assert sorted(p.name for p in tmp_path.iterdir()) == ["cut.fcpxml.aar-lock"]
    finally:
        file_lock.release_file_lock(fd)
    assert main(argv) == 0


# -- the Windows backend, simulated ------------------------------------------------------------


def _windows_lock_api(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, list[tuple[int, int]]]:
    """The helper loaded as on Windows, over a fake ``msvcrt`` whose byte locks are shared by
    every handle on one file (as Windows' are), keyed by the file's identity."""
    held: set[tuple[int, int]] = set()
    calls: list[tuple[int, int]] = []
    fake = types.ModuleType("msvcrt")
    fake.LK_NBLCK = 2  # type: ignore[attr-defined]
    fake.LK_UNLCK = 0  # type: ignore[attr-defined]

    def locking(fd: int, mode: int, size: int) -> None:
        st = os.fstat(fd)
        key = (st.st_dev, st.st_ino)
        calls.append((mode, size))
        if mode == fake.LK_NBLCK:  # type: ignore[attr-defined]
            if key in held:
                raise OSError(13, "Permission denied")  # msvcrt's EACCES for a held range
            held.add(key)
        else:
            held.discard(key)

    fake.locking = locking  # type: ignore[attr-defined]
    helper = Path(file_lock.__file__)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setitem(sys.modules, "msvcrt", fake)
    monkeypatch.setitem(sys.modules, "fcntl", None)
    namespace = runpy.run_path(str(helper))
    monkeypatch.undo()  # sys.platform and the modules back; the loaded helper keeps its backend
    assert namespace["_WINDOWS_BACKEND"] is True
    return types.SimpleNamespace(**namespace), calls


def test_the_windows_backend_serializes_exports_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api, calls = _windows_lock_api(monkeypatch)
    monkeypatch.setattr(files, "_lock_api", lambda: api)
    held = _Held(tmp_path, monkeypatch)
    with pytest.raises(ExportError) as caught:
        write_pair(_pair(tmp_path, "second"), overwrite=True)
    assert caught.value.code == "output_locked"
    held.finish()
    assert (tmp_path / "cut.fcpxml").read_bytes() == b"<first fcpxml/>"
    # The Windows helper locks a real byte of the lock file, and unlocks it on release.
    assert (tmp_path / "cut.fcpxml.aar-lock").read_bytes() == b"\0"
    # The first takes both locks, the second is refused the first lock, the first frees both.
    assert calls == [(2, 1), (2, 1), (2, 1), (0, 1), (0, 1)]
    write_pair(_pair(tmp_path, "third"), overwrite=True)  # free again
    assert calls[-4:] == [(2, 1), (2, 1), (0, 1), (0, 1)]
