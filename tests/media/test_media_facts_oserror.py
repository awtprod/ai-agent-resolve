"""``aar media facts``: no ``OSError`` about one path ever becomes a traceback.

Python 3.12 and 3.13 raise ``PermissionError`` from ``Path.is_dir()`` (and ``is_file()``,
``exists()``) for a path below a folder that cannot be entered, where 3.14 returns False. These
tests make the calls raise for chosen paths whatever this interpreter does, with ffprobe
replaced by a fixed answer, so they mean the same on every Python and file system: the file's
(or folder's) own error line or a typed refusal, the other files measured, never a traceback.
"""

from __future__ import annotations

import contextlib
import errno
import json
import ntpath
import os
import posixpath
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from aar.media import facts as facts_mod
from aar.media import tools
from aar.media.facts import main

PC = "E:\\Show 1"
STREAMS = {"format": {"duration": "1.0"}, "streams": [{"codec_type": "video", "index": 0}]}


def _probe(*_: Any, **__: Any) -> tools.RunResult:
    return tools.RunResult(("ffprobe",), 0, json.dumps(STREAMS).encode(), "", False, 0.0)


@pytest.fixture
def show(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """``show/footage/{a.mp4,b.mov,notes.txt}`` and ``show/sound/c.wav``, ffprobe faked."""
    root = tmp_path / "show"
    for rel in ("footage/a.mp4", "footage/b.mov", "footage/notes.txt", "sound/c.wav"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(b"media")
    monkeypatch.setattr(facts_mod, "run_tool", _probe)
    return root


def _pc(rel: str) -> str:
    return PC + "\\" + rel.replace("/", "\\")


def _go(
    root: Path, *names: str, capsys: pytest.CaptureFixture[str], out: Path | None = None
) -> tuple[int, list[dict[str, Any]], Path]:
    out = out or root.parent / "facts.jsonl"
    argv = [str(root / n) for n in names] + ["--out", str(out), "--pc-folder", f"{root}={PC}"]
    code = main(argv)
    captured = capsys.readouterr()
    assert "Traceback" not in captured.out + captured.err
    assert captured.err == ""
    return code, [json.loads(line) for line in captured.out.splitlines()], out


def _by_path(lines: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {line["path"]: line for line in lines if "path" in line}


def _deny_paths(
    monkeypatch: pytest.MonkeyPatch, denied: set[str], make: Callable[[str], OSError]
) -> None:
    """As Python 3.12 sees a path below a folder it cannot enter: ``Path.is_dir``,
    ``is_file``, ``exists`` and ``os.stat`` raise for it."""

    def guard(path: Any) -> None:
        name = os.fspath(path)
        if name in denied:
            raise make(name)

    for method in ("is_dir", "is_file", "exists"):
        original = getattr(Path, method)

        def patched(self: Path, *a: Any, _original: Any = original, **k: Any) -> Any:
            guard(self)
            return _original(self, *a, **k)

        monkeypatch.setattr(Path, method, patched)
    real_stat = os.stat

    def stat(path: Any, *a: Any, **k: Any) -> Any:
        guard(path)
        return real_stat(path, *a, **k)

    monkeypatch.setattr(os, "stat", stat)


ERRORS: list[Callable[[str], OSError]] = [
    lambda p: PermissionError(errno.EACCES, "Permission denied", p),
    lambda p: OSError(errno.EIO, "Input/output error", p),
    lambda p: OSError(errno.ENAMETOOLONG, "File name too long", p),
]


# --------------------------------------------------------------------------- simulated 3.12


@pytest.mark.parametrize("make", ERRORS, ids=["EACCES", "EIO", "ENAMETOOLONG"])
def test_a_named_file_below_a_folder_that_cannot_be_entered_is_its_own_line(
    show: Path,
    make: Callable[[str], OSError],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    locked = show / "locked" / "d.mp4"
    _deny_paths(monkeypatch, {str(locked)}, make)
    code, lines, out = _go(show, "footage/a.mp4", "locked/d.mp4", "sound/c.wav", capsys=capsys)
    assert code == 2
    printed = _by_path(lines)
    bad = printed[_pc("locked/d.mp4")]
    assert bad["ok"] is False and bad["error"]["code"] == "unreadable"
    assert make("").strerror in bad["error"]["message"]
    assert printed[_pc("footage/a.mp4")]["ok"] and printed[_pc("sound/c.wav")]["ok"]
    records = [json.loads(line) for line in out.read_text("utf-8").splitlines()]
    assert [r["path"] for r in records] == [
        _pc("footage/a.mp4"),
        _pc("locked/d.mp4"),
        _pc("sound/c.wav"),
    ]
    assert "ffprobe" in records[0] and "error" in records[1] and "ffprobe" in records[2]
    assert lines[-1]["failed"] == 1 and lines[-1]["files"] == 3


@pytest.mark.parametrize("make", ERRORS, ids=["EACCES", "EIO", "ENAMETOOLONG"])
def test_a_named_folder_that_cannot_be_told_is_its_own_line(
    show: Path,
    make: Callable[[str], OSError],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _deny_paths(monkeypatch, {str(show / "footage")}, make)
    code, lines, _ = _go(show, "footage", "sound/c.wav", capsys=capsys)
    assert code == 2
    printed = _by_path(lines)
    assert printed[_pc("footage")]["error"]["code"] == "unreadable"
    assert printed[_pc("sound/c.wav")]["ok"] is True
    assert lines[-1]["files"] == 2 and lines[-1]["failed"] == 1


class _Entry:
    """A folder entry whose kind cannot be told (``DirEntry`` methods raise)."""

    def __init__(self, entry: os.DirEntry[str], make: Callable[[str], OSError]) -> None:
        self._entry, self._make = entry, make
        self.name, self.path = entry.name, entry.path

    def is_dir(self, *, follow_symlinks: bool = True) -> bool:
        raise self._make(self.path)

    def is_file(self, *, follow_symlinks: bool = True) -> bool:
        raise self._make(self.path)

    def is_symlink(self) -> bool:
        raise self._make(self.path)


class _Scan:
    def __init__(self, it: Any, denied: set[str], make: Callable[[str], OSError]) -> None:
        self._it, self._denied, self._make = it, denied, make

    def __enter__(self) -> _Scan:
        return self

    def __exit__(self, *_: object) -> None:
        self._it.close()

    def __iter__(self) -> Iterator[Any]:
        for entry in self._it:
            yield _Entry(entry, self._make) if entry.name in self._denied else entry


@pytest.mark.parametrize("make", ERRORS, ids=["EACCES", "EIO", "ENAMETOOLONG"])
def test_an_entry_in_a_folder_that_cannot_be_told_is_measured_or_listed(
    show: Path,
    make: Callable[[str], OSError],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A media name is measured, so its own line says why it cannot be read; another name is
    listed as skipped ``unreadable``."""
    real_scandir = os.scandir
    denied = {"b.mov", "notes.txt"}
    monkeypatch.setattr(os, "scandir", lambda p: _Scan(real_scandir(p), denied, make))
    _deny_paths(monkeypatch, {str(show / "footage" / "b.mov")}, make)
    code, lines, _ = _go(show, "footage", capsys=capsys)
    assert code == 2
    printed = _by_path(lines)
    assert printed[_pc("footage/a.mp4")]["ok"] is True
    assert printed[_pc("footage/b.mov")]["error"]["code"] == "unreadable"
    assert lines[-1]["skipped"]["by_reason"] == {"unreadable": 1}
    assert lines[-1]["skipped"]["first"] == [
        {"path": str(show / "footage" / "notes.txt"), "reason": "unreadable"}
    ]


# --------------------------------------------------------------------------- the last guard


def test_an_oserror_no_step_handles_is_a_typed_refusal_naming_the_path(
    show: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def broken(*_: Any, **__: Any) -> Any:
        raise PermissionError(errno.EACCES, "Permission denied", "X:\\somewhere")

    monkeypatch.setattr(facts_mod, "targets", broken)
    code, lines, out = _go(show, "footage", capsys=capsys)
    assert code == 2 and not out.exists()
    assert lines == [{"ok": False, "error": lines[0]["error"]}]
    assert lines[0]["error"]["code"] == "system_error"
    assert "Permission denied" in lines[0]["error"]["message"]
    assert "X:\\\\somewhere" in lines[0]["error"]["message"]  # repr of the path


def test_a_programming_error_is_not_hidden(show: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(*_: Any, **__: Any) -> Any:
        raise TypeError("a bug")

    monkeypatch.setattr(facts_mod, "targets", broken)
    with pytest.raises(TypeError, match="a bug"):
        main([str(show / "footage"), "--out", str(show / "f.jsonl")])


def test_a_console_that_fails_still_gets_a_typed_refusal_on_stderr(
    show: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class Failing:
        def write(self, _: str) -> int:
            raise OSError(errno.EINVAL, "Invalid argument")

        def flush(self) -> None:
            raise OSError(errno.EINVAL, "Invalid argument")

    out = show.parent / "facts.jsonl"
    monkeypatch.setattr(facts_mod.sys, "stdout", Failing())
    code = main([str(show / "footage"), "--out", str(out), "--pc-folder", f"{show}={PC}"])
    err = capsys.readouterr().err
    assert code == 2 and "Traceback" not in err
    record = json.loads(err)
    assert record["error"]["code"] == "system_error"
    assert out.exists()  # the facts file was written before the console failed


# --------------------------------------------------------------------------- the table

# Every call in ``aar media facts`` that touches the file system or the environment, from
# argument parsing to the last write, with the outcome when it raises (the final message of
# S25d fix round 3 carries the same table with the exceptions each call raises on 3.12/3.13).


def _fail_when(
    monkeypatch: pytest.MonkeyPatch,
    owner: Any,
    name: str,
    exc: OSError,
    when: Callable[..., bool] = lambda *a, **k: True,
) -> None:
    original = getattr(owner, name)

    def patched(*a: Any, **k: Any) -> Any:
        if when(*a, **k):
            raise exc
        return original(*a, **k)

    monkeypatch.setattr(owner, name, patched)


def _about(rel: str) -> Callable[..., bool]:
    def when(*a: Any, **_: Any) -> bool:
        return bool(a) and os.fspath(a[0]).replace("\\", "/").endswith(rel)

    return when


EACCES = PermissionError(errno.EACCES, "Permission denied", "the-path")
EIO = OSError(errno.EIO, "Input/output error", "the-path")


def _inject(case: str, mp: pytest.MonkeyPatch, root: Path) -> None:
    out_dir = str(root.parent)
    if case == "Path.is_dir(named path)":
        _fail_when(mp, Path, "is_dir", EACCES, _about("footage"))
        _fail_when(mp, os, "stat", EACCES, _about("footage"))
    elif case == "Path.is_dir(out folder)":
        _fail_when(mp, Path, "is_dir", EACCES, lambda p: os.fspath(p) == out_dir)
    elif case == "os.path.lexists(out)":
        _fail_when(mp, os.path, "lexists", EIO, _about("facts.jsonl"))
    elif case == "os.scandir":
        _fail_when(mp, os, "scandir", EACCES)
    elif case == "iterating os.scandir":
        real = os.scandir

        class Broken:
            def __init__(self, p: Any) -> None:
                self.it = real(p)

            def __enter__(self) -> Broken:
                return self

            def __exit__(self, *_: object) -> None:
                self.it.close()

            def __iter__(self) -> Iterator[Any]:
                yield next(iter(self.it))
                raise EIO

        mp.setattr(os, "scandir", Broken)
    elif case == "DirEntry.is_dir/is_file/is_symlink":
        real_scandir = os.scandir
        mp.setattr(os, "scandir", lambda p: _Scan(real_scandir(p), {"b.mov"}, lambda q: EIO))
    elif case == "os.path.exists(dangling check)":
        _fail_when(mp, os.path, "exists", EIO, _about("footage/b.mov"))
        real_scandir = os.scandir

        class Linky:
            def __init__(self, p: Any) -> None:
                self.it = real_scandir(p)

            def __enter__(self) -> Linky:
                return self

            def __exit__(self, *_: object) -> None:
                self.it.close()

            def __iter__(self) -> Iterator[Any]:
                for entry in self.it:
                    if entry.name == "b.mov":
                        link = _Entry(entry, lambda q: EIO)
                        link.is_dir = lambda **_: False  # type: ignore[method-assign]
                        link.is_symlink = lambda: True  # type: ignore[method-assign]
                        yield link
                    else:
                        yield entry

        mp.setattr(os, "scandir", Linky)
    elif case == "abspath(_pc_path)":
        _fail_when(mp, posixpath, "abspath", EIO, _about("sound/c.wav"))
        _fail_when(mp, ntpath, "abspath", EIO, _about("sound/c.wav"))
    elif case == "os.path.lexists(at_root)":
        _fail_when(mp, os.path, "lexists", EIO, _about("footage/a.mp4"))
    elif case == "os.stat(before) ENOENT":
        _fail_when(
            mp, os, "stat", FileNotFoundError(errno.ENOENT, "gone", "p"), _about("sound/c.wav")
        )
    elif case == "os.stat(before) EACCES":
        _fail_when(mp, os, "stat", EACCES, _about("sound/c.wav"))
    elif case == "os.stat(after)":
        probed: set[str] = set()  # the second stat, once ffprobe has run

        def probe_then_arm(*a: Any, **k: Any) -> tools.RunResult:
            probed.add(a[1][-1])
            return _probe(*a, **k)

        mp.setattr(facts_mod, "run_tool", probe_then_arm)
        _fail_when(
            mp,
            os,
            "stat",
            EIO,
            lambda *a, **_: (
                any(u.endswith(os.fspath(a[0])) for u in probed) and _about("sound/c.wav")(*a)
            ),
        )
    elif case == "input_url(os.path.abspath)":
        _fail_when(mp, facts_mod, "input_url", EIO, _about("sound/c.wav"))
    elif case == "run_tool(other OSError)":
        _fail_when(mp, facts_mod, "run_tool", EIO)
    elif case == "subprocess.Popen":
        mp.setattr(facts_mod, "run_tool", tools.run_tool)
        mp.setenv("AAR_FFPROBE", os.path.abspath(sys.executable))
        _fail_when(mp, tools.subprocess, "Popen", OSError(errno.ENOMEM, "Out of memory"))
    elif case == "time.localtime/gmtime":
        _fail_when(mp, time, "localtime", OSError(errno.EINVAL, "Invalid argument"))
    elif case == "secrets.token_hex":
        _fail_when(mp, facts_mod.secrets, "token_hex", OSError(errno.EIO, "no randomness"))
    elif case == "os.open(temp)":
        _fail_when(mp, os, "open", EACCES, _about(".part"))
    elif case == "os.fsync":
        _fail_when(mp, os, "fsync", OSError(errno.ENOSPC, "No space left on device"))
    elif case == "os.link/os.rename EEXIST":
        exists = FileExistsError(errno.EEXIST, "File exists", "facts.jsonl")
        _fail_when(mp, os, "link", exists)
        _fail_when(mp, os, "rename", exists)
    elif case == "os.link/os.rename EPERM":
        _fail_when(mp, os, "link", OSError(errno.EPERM, "Operation not permitted"))
        _fail_when(mp, os, "rename", OSError(errno.EPERM, "Operation not permitted"))
    elif case == "os.unlink(temp)":
        _fail_when(mp, os, "unlink", EACCES, _about(".part"))
    else:  # pragma: no cover - a case named in TABLE but not injected
        raise AssertionError(case)


#: (call, names given, PC folder, expected exit, where the outcome shows, expected code)
TABLE: list[tuple[str, tuple[str, ...], str, int | str, str, str | None]] = [
    ("Path.is_dir(named path)", ("footage", "sound/c.wav"), PC, 2, "line:footage", "unreadable"),
    ("Path.is_dir(out folder)", ("footage",), PC, 2, "refusal", "output_unwritable"),
    ("os.path.lexists(out)", ("footage",), PC, 2, "refusal", "output_unwritable"),
    ("os.scandir", ("footage",), PC, 2, "refusal", "folder_unreadable"),
    ("iterating os.scandir", ("footage",), PC, 2, "refusal", "folder_unreadable"),
    ("DirEntry.is_dir/is_file/is_symlink", ("footage",), PC, 0, "line:footage/b.mov", None),
    ("os.path.exists(dangling check)", ("footage",), PC, 0, "line:footage/b.mov", None),
    ("abspath(_pc_path)", ("footage", "sound/c.wav"), PC, 2, "refusal", "path_unreadable"),
    ("os.path.lexists(at_root)", ("footage",), "E:\\", 2, "line:a.mp4", "unreadable"),
    ("os.stat(before) ENOENT", ("footage", "sound/c.wav"), PC, 2, "line:sound/c.wav", "missing"),
    (
        "os.stat(before) EACCES",
        ("footage", "sound/c.wav"),
        PC,
        2,
        "line:sound/c.wav",
        "unreadable",
    ),
    (
        "os.stat(after)",
        ("footage", "sound/c.wav"),
        PC,
        2,
        "line:sound/c.wav",
        "changed_while_measured",
    ),
    (
        "input_url(os.path.abspath)",
        ("footage", "sound/c.wav"),
        PC,
        2,
        "line:sound/c.wav",
        "unreadable",
    ),
    ("run_tool(other OSError)", ("footage",), PC, 2, "line:footage/a.mp4", "unreadable"),
    ("subprocess.Popen", ("footage",), PC, 2, "refusal", "tool_missing"),
    ("time.localtime/gmtime", ("footage",), PC, 2, "line:footage/a.mp4", "bad_time"),
    ("secrets.token_hex", ("footage",), PC, 2, "refusal", "output_unwritable"),
    ("os.open(temp)", ("footage",), PC, 2, "refusal", "output_unwritable"),
    ("os.fsync", ("footage",), PC, 2, "refusal", "output_unwritable"),
    ("os.link/os.rename EEXIST", ("footage",), PC, 2, "refusal", "output_exists"),
    (
        "os.link/os.rename EPERM",
        ("footage",),
        PC,
        0 if os.name == "posix" else 2,  # POSIX writes the name itself; Windows refuses
        "written" if os.name == "posix" else "refusal",
        None if os.name == "posix" else "output_unwritable",
    ),
    ("os.unlink(temp)", ("footage",), PC, 0, "left_behind", None),
]


@pytest.mark.parametrize(
    ("case", "names", "pc", "exit_code", "where", "code"), TABLE, ids=[row[0] for row in TABLE]
)
def test_each_call_that_raises_has_its_documented_outcome(
    case: str,
    names: tuple[str, ...],
    pc: str,
    exit_code: int,
    where: str,
    code: str | None,
    show: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out = show.parent / "facts.jsonl"
    argv = [str(show / n) for n in names] + ["--out", str(out), "--pc-folder", f"{show}={pc}"]
    if pc == "E:\\":  # files directly at a drive root: the at_root branch
        argv = [str(show / "footage"), "--out", str(out), "--pc-folder", f"{show / 'footage'}={pc}"]
    with monkeypatch.context() as mp:
        _inject(case, mp, show)
        got = main(argv)
    captured = capsys.readouterr()
    assert "Traceback" not in captured.out + captured.err and captured.err == ""
    lines = [json.loads(line) for line in captured.out.splitlines()]
    assert got == exit_code, lines
    if where == "refusal":
        assert len(lines) == 1 and lines[0]["ok"] is False, lines
        assert lines[0]["error"]["code"] == code
    elif where.startswith("line:"):
        rel = where.removeprefix("line:")
        path = (pc.rstrip("\\") + "\\" + rel.replace("/", "\\")) if pc != "E:\\" else pc + rel
        line = _by_path(lines)[path]
        assert (line.get("error") or {}).get("code") == code, line
        assert lines[-1]["files"] == len([x for x in lines if "path" in x])
    elif where == "left_behind":
        assert lines[-1]["ok"] is True and out.exists()
        assert [Path(p).name for p in lines[-1]["left_behind"]] == [
            p.name for p in show.parent.glob(".facts.jsonl.aar-facts-*.part")
        ]
    else:
        assert lines[-1]["ok"] is True and out.exists()


# --------------------------------------------------------------------------- the sweep

#: The system calls ``aar media facts`` may make, directly or through pathlib and os.path.
SWEPT: list[tuple[Any, str]] = [
    (os, "stat"),
    (os, "lstat"),
    (os, "scandir"),
    (os, "open"),
    (os, "fsync"),
    (os, "link"),
    (os, "rename"),
    (os, "replace"),
    (os, "unlink"),
    (os, "getcwd"),
    (os, "access"),
    (os.path, "exists"),
    (os.path, "lexists"),
    (os.path, "isdir"),
    (os.path, "isfile"),
    (os.path, "islink"),
    (os.path, "abspath"),
    (os.path, "realpath"),
    (Path, "stat"),
    (Path, "lstat"),
    (Path, "is_dir"),
    (Path, "is_file"),
    (Path, "exists"),
    (Path, "resolve"),
    (Path, "iterdir"),
    (time, "localtime"),
    (time, "gmtime"),
    (facts_mod, "run_tool"),
    (facts_mod, "input_url"),
]


@contextlib.contextmanager
def _sweep(fail_at: int | None) -> Iterator[list[int]]:
    """Patch every call in :data:`SWEPT`; the ``fail_at``-th call (counted across all of them,
    from 0) raises ``PermissionError``. Yields the call count so far."""
    count = [0]
    with pytest.MonkeyPatch.context() as mp:
        for owner, name in SWEPT:
            original = getattr(owner, name)

            def patched(*a: Any, _original: Any = original, _name: str = name, **k: Any) -> Any:
                n = count[0]
                count[0] += 1
                if n == fail_at:
                    raise PermissionError(errno.EACCES, f"Permission denied ({_name})", "the-path")
                return _original(*a, **k)

            mp.setattr(owner, name, patched)
        yield count


def test_any_system_call_that_raises_never_ends_in_a_traceback(
    show: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Each system call the run makes is made to raise in turn (whatever this Python calls
    underneath): the run ends with a file's error line or a step's typed refusal every time,
    never a traceback and never the last guard's ``system_error``. A call added later without
    handling of its own fails here."""

    def run(k: int | None) -> tuple[int, int, list[dict[str, Any]]]:
        folder = show.parent / f"out-{k}"
        folder.mkdir()
        argv = [str(show / "footage"), str(show / "sound" / "c.wav"), "--out"]
        argv += [str(folder / "facts.jsonl"), "--pc-folder", f"{show}={PC}"]
        with _sweep(k) as count:
            code = main(argv)
        captured = capsys.readouterr()
        assert "Traceback" not in captured.out + captured.err, (k, captured)
        assert captured.err == "", (k, captured.err)
        return code, count[0], [json.loads(line) for line in captured.out.splitlines()]

    code, calls, lines = run(None)
    assert code == 0 and lines[-1]["ok"] is True and lines[-1]["files"] == 3
    assert calls > 10, calls  # the sweep sees the run's calls
    for k in range(calls):
        code, _, lines = run(k)
        assert code in (0, 2), (k, lines)
        assert lines and all(isinstance(line, dict) for line in lines), k
        if code == 2:
            assert any(line.get("ok") is False for line in lines), (k, lines)
        # Handled where it happened, never by the last guard: a call added without its own
        # handling shows up here as ``system_error``.
        assert all((line.get("error") or {}).get("code") != "system_error" for line in lines), (
            k,
            lines,
        )
