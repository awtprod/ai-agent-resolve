"""S25c2 fix round 2: the placement's lock and staging on every path (N1, N5), the no-replace
rename where the system forbids it (N3), no subprocess at import (N4), and the verifier's rule
for every item column the writer sets (N2).

Every value is made up (``libwrite_support``)."""

from __future__ import annotations

import errno
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from libwrite_support import make_library

import aar.drp.libwrite.place as place
from aar.drp.items.rows import ITEM_COLUMNS
from aar.drp.libwrite import ProjectWriteError, place_project
from aar.drp.libwrite.place import STAGING
from aar.drp.libwrite.verify import DERIVED, FRESH, ITEM_RULES


def _projects(library: Path) -> Path:
    return library / "Resolve Projects" / "Users" / "guest" / "Projects"


def _top(library: Path) -> list[str]:
    return sorted(os.listdir(library / "Resolve Projects"))


def _write(data: bytes) -> Callable[[Path], None]:
    return lambda path: path.write_bytes(data)


# -- N1. the lock and the staging folder, on every path -----------------------------------------


def _failing_write(exc: BaseException) -> Callable[..., int]:
    def write(fd: int, data: bytes) -> int:
        raise exc

    return write


def test_a_lock_that_cannot_be_written_is_removed_with_the_staging_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    library = make_library(tmp_path / "lib")
    monkeypatch.setattr(place.os, "write", _failing_write(OSError(errno.ENOSPC, "No space")))
    with pytest.raises(ProjectWriteError) as info:
        place_project(library, "Full", _write(b"ours"))
    assert info.value.code == "library_unwritable"
    assert _top(library) == ["Users"]  # no lock, no staging folder
    monkeypatch.undo()
    # The next export of that name is not told another one is running.
    placed, _ = place_project(library, "Full", _write(b"ours"))
    assert placed.path.read_bytes() == b"ours"


def test_an_interrupt_while_the_lock_is_written_leaves_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    library = make_library(tmp_path / "lib")
    monkeypatch.setattr(place.os, "write", _failing_write(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        place_project(library, "Stopped", _write(b"ours"))
    assert _top(library) == ["Users"]


def test_a_lock_that_cannot_be_removed_after_a_failed_write_is_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    library = make_library(tmp_path / "lib")
    monkeypatch.setattr(place.os, "write", _failing_write(OSError(errno.ENOSPC, "No space")))
    monkeypatch.setattr(place, "_remove", lambda path: f"{path} could not be removed (injected)")
    with pytest.raises(ProjectWriteError) as info:
        place_project(library, "Full", _write(b"ours"))
    lock = library / "Resolve Projects" / STAGING / place._lock_name("Full")
    assert info.value.code == "library_unwritable"
    assert str(lock) in info.value.message and str(lock) in info.value.details["residue"]
    assert "delete it by hand" in info.value.message


def test_an_interrupt_between_making_the_lock_and_recording_it_names_the_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The window CPython leaves between ``os.open`` returning and the next statement: the
    empty file is named (it may be this export's), never deleted."""
    library = make_library(tmp_path / "lib")
    real_open = os.open

    def open_then_interrupt(path: Any, flags: int, mode: int = 0o777) -> int:
        os.close(real_open(path, flags, mode))
        raise KeyboardInterrupt

    monkeypatch.setattr(place.os, "open", open_then_interrupt)
    with pytest.raises(KeyboardInterrupt) as info:
        place_project(library, "Window", _write(b"ours"))
    lock = library / "Resolve Projects" / STAGING / place._lock_name("Window")
    assert lock.exists()
    assert str(lock) in info.value.residue  # type: ignore[attr-defined]
    assert any(str(lock) in note for note in info.value.__notes__)


@pytest.mark.parametrize("step", ["_listdir", "_real_folder", "_inside"])
def test_an_interrupt_right_after_the_staging_folder_is_made_removes_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, step: str
) -> None:
    library = make_library(tmp_path / "lib")
    real = getattr(place, step)

    def interrupted(path: Any, *args: Any, **kwargs: Any) -> Any:
        if Path(path).name == STAGING:
            raise KeyboardInterrupt
        return real(path, *args, **kwargs)

    monkeypatch.setattr(place, step, interrupted)
    with pytest.raises(KeyboardInterrupt):
        place_project(library, "Early", _write(b"ours"))
    assert _top(library) == ["Users"]


def test_an_interrupt_right_after_the_rename_says_the_project_is_placed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    library = make_library(tmp_path / "lib")
    real = place._move

    def moved_then_interrupted(folder: Path, target: Path) -> None:
        real(folder, target)
        raise KeyboardInterrupt

    monkeypatch.setattr(place, "_move", moved_then_interrupted)
    with pytest.raises(KeyboardInterrupt) as info:
        place_project(library, "Late", _write(b"ours"))
    assert (_projects(library) / "Late" / "Project.db").read_bytes() == b"ours"
    assert any("the project was placed at" in note for note in info.value.__notes__)
    assert _top(library) == ["Users"]  # lock and build folder removed


@pytest.mark.skipif(sys.platform == "win32", reason="an empty folder's permissions on POSIX")
def test_an_empty_staging_folder_that_cannot_be_removed_is_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    library = make_library(tmp_path / "lib")
    root = library / "Resolve Projects" / STAGING
    real_rmdir = Path.rmdir

    def rmdir(self: Path) -> None:
        if self == root:
            raise PermissionError(errno.EACCES, "Permission denied", str(self))
        real_rmdir(self)

    monkeypatch.setattr(Path, "rmdir", rmdir)
    placed, _ = place_project(library, "Kept", _write(b"ours"))
    assert any(str(root) in note and "empty" in note for note in placed.cleanup)


# -- N3. renameat2 refused by a syscall filter, or not there ------------------------------------


def _renameat2_failing(code: int) -> Callable[[], Callable[[bytes, bytes], tuple[int, int]]]:
    return lambda: lambda old, new: (-1, code)


@pytest.mark.parametrize(
    "code", [errno.EPERM, errno.ENOSYS, errno.EINVAL, errno.EOPNOTSUPP], ids=errno.errorcode.get
)
def test_a_no_replace_rename_that_is_not_available_here_falls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, code: int
) -> None:
    library = make_library(tmp_path / "lib")
    monkeypatch.setattr(place, "_renameat2", _renameat2_failing(code))
    placed, _ = place_project(library, "Fallback", _write(b"ours"))
    assert placed.path.read_bytes() == b"ours"


@pytest.mark.parametrize("code", [errno.EPERM, errno.ENOSYS], ids=errno.errorcode.get)
def test_the_fallback_never_replaces_a_folder_the_check_missed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, code: int
) -> None:
    library = make_library(tmp_path / "lib")
    theirs = _projects(library) / "Theirs"
    theirs.mkdir()
    (theirs / "Project.db").write_bytes(b"theirs")
    monkeypatch.setattr(place, "_renameat2", _renameat2_failing(code))
    monkeypatch.setattr(place, "_exists", lambda projects, name: None)  # a check that missed
    with pytest.raises(ProjectWriteError) as info:
        place_project(library, "Theirs", _write(b"ours"))
    assert info.value.code == "project_exists"
    assert (theirs / "Project.db").read_bytes() == b"theirs"
    assert _top(library) == ["Users"]


_FAKE_LIBC = """
import ctypes, errno, os, sys
from pathlib import Path

class Func:
    def __call__(self, *args):
        ctypes.set_errno({code})
        return -1

class Lib:
    def __init__(self, *args, **kwargs):
        self.renameat2 = Func()

ctypes.CDLL = Lib
import aar.drp.libwrite.place as place

library = Path(sys.argv[1])
(library / "Resolve Projects" / "Users" / "guest" / "Projects").mkdir(parents=True)
{body}
"""


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux's renameat2")
@pytest.mark.parametrize(
    ("code", "outcome"),
    [
        (errno.EPERM, "placed"),
        (errno.ENOSYS, "placed"),
        (errno.EINVAL, "placed"),
        (errno.EACCES, "library_unwritable"),
    ],
    ids=["EPERM", "ENOSYS", "EINVAL", "EACCES"],
)
def test_the_c_librarys_renameat2_refusing_falls_back_only_when_it_is_not_available(
    tmp_path: Path, code: int, outcome: str
) -> None:
    """The C library's own call set to fail with ``code`` (as a syscall filter makes it): the
    placement falls back for an error that means "not available here", refuses on another."""
    body = (
        "try:\n"
        "    placed, _ = place.place_project(library, 'Filtered', lambda p: p.write_bytes(b'ok'))\n"
        "    print('placed', placed.path.read_bytes().decode())\n"
        "except place.ProjectWriteError as exc:\n"
        "    print(exc.code)\n"
    )
    script = _FAKE_LIBC.format(code=code, body=body)
    out = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "lib")],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.split()[0] == outcome
    if outcome == "placed":
        assert out.stdout.split()[1] == "ok"
    staging = tmp_path / "lib" / "Resolve Projects"
    assert sorted(os.listdir(staging)) == ["Users"]


@pytest.mark.parametrize("code", [errno.EACCES, errno.EIO, errno.EXDEV], ids=errno.errorcode.get)
def test_a_real_error_of_the_no_replace_rename_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, code: int
) -> None:
    library = make_library(tmp_path / "lib")
    monkeypatch.setattr(place, "_renameat2", _renameat2_failing(code))
    with pytest.raises(ProjectWriteError) as info:
        place_project(library, "Refused", _write(b"ours"))
    assert info.value.code == "library_unwritable"
    assert list(_projects(library).iterdir()) == [] and _top(library) == ["Users"]


def test_a_c_library_without_renameat2_falls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import ctypes

    class NoSymbol:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __getattr__(self, name: str) -> Any:
            raise AttributeError(name)

    monkeypatch.setattr(ctypes, "CDLL", NoSymbol)
    monkeypatch.setattr(place, "_NOREPLACE", [])
    assert place._renameat2() is None
    library = make_library(tmp_path / "lib")
    placed, _ = place_project(library, "NoSymbol", _write(b"ours"))
    assert placed.path.read_bytes() == b"ours"


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux's renameat2")
def test_the_real_renameat2_refuses_an_existing_folder(tmp_path: Path) -> None:
    call = place._renameat2()
    if call is None:
        pytest.skip("this C library has no renameat2")
    ours, theirs = tmp_path / "ours", tmp_path / "theirs"
    ours.mkdir()
    theirs.mkdir()
    result, code = call(os.fsencode(ours), os.fsencode(theirs))
    assert (result, code) == (-1, errno.EEXIST)
    assert ours.is_dir() and theirs.is_dir()


def test_on_windows_there_is_no_renameat2(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(place.sys, "platform", "win32")
    monkeypatch.setattr(place, "_NOREPLACE", [])
    assert place._renameat2() is None


# -- N4. no subprocess at import -----------------------------------------------------------------


def test_importing_and_using_the_placement_starts_no_process(tmp_path: Path) -> None:
    code = (
        "import subprocess, ctypes.util\n"
        "def boom(*a, **k):\n"
        "    raise SystemExit('a process was started')\n"
        "subprocess.Popen.__init__ = boom\n"
        "ctypes.util.find_library = boom\n"
        "import aar.drp.libwrite.place as place\n"
        "place._renameat2()\n"
        "print('ok')\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120, check=False
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "ok"


# -- N5. a killed export's leftovers are named with the stale lock -------------------------------


def test_a_stale_lock_is_named_with_the_build_folder_a_killed_export_left(tmp_path: Path) -> None:
    library = make_library(tmp_path / "lib")
    root = library / "Resolve Projects" / STAGING
    root.mkdir()
    lock = root / place._lock_name("Killed")
    lock.write_bytes(b"0123 4567\n")
    build = root / "build-0011223344556677"
    (build / "project").mkdir(parents=True)
    with pytest.raises(ProjectWriteError) as info:
        place_project(library, "killed", _write(b"ours"))
    assert info.value.code == "project_exists"
    assert build.name in info.value.message
    assert str(build) in info.value.hint and str(lock) in info.value.hint
    assert "safe to delete when no export is running" in info.value.hint
    assert info.value.details["leftovers"] == [str(build)]
    assert lock.exists() and build.is_dir()  # reported, never deleted


# -- N2. a rule for every item column the writer sets ---------------------------------------------


def test_the_verifier_has_a_rule_for_every_column_of_the_writers_closed_list() -> None:
    """Derived from the writer's own list: a column added to ``ITEM_COLUMNS`` without a rule
    here fails this test (and the verifier refuses every project until it has one)."""
    assert set(ITEM_RULES) == set(ITEM_COLUMNS)
    assert len(ITEM_RULES) == len(ITEM_COLUMNS)
    for column, rule in ITEM_RULES.items():
        assert rule in (DERIVED, FRESH) or (isinstance(rule, tuple) and rule[0] == "constant"), (
            column
        )


def test_every_column_the_writer_leaves_for_a_later_edit_must_be_empty() -> None:
    from aar.drp.items.rows import LATER_COLUMNS

    assert {c for c, rule in ITEM_RULES.items() if rule == FRESH} == set(LATER_COLUMNS)
