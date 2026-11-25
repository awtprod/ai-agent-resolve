"""S25d fix round: ``aar media facts`` on real-world names, consoles and folders. Console output
is ASCII JSON whatever the console's encoding; the facts file is UTF-8, a name that is not
Unicode an error line of its own; what a folder holds that is not measured is listed; no media
at all is a refusal; the run has an overall deadline and an interrupted run keeps what it
measured; the byte cap holds for error lines; the output is never replaced, its temporary file
never left; Windows' file times and program lookup are guarded (simulated here)."""

from __future__ import annotations

import errno
import json
import os
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from media_support import FFMPEG, needs_ffmpeg

from aar.drp.mediapool.gate import load_facts
from aar.media import facts as facts_mod
from aar.media import tools
from aar.media.facts import Target, birth_ns, main, measure, measure_file, write_new

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
PC = "E:\\Show 1\\Footage"


def _media(path: Path) -> Path:
    assert FFMPEG is not None
    cmd = [FFMPEG, "-hide_banner", "-nostdin", "-v", "error", "-y", "-f", "lavfi"]
    cmd += ["-i", "testsrc2=size=160x90:rate=25:duration=0.5", "-c:v", "libx264"]
    cmd += ["-preset", "ultrafast", "-pix_fmt", "yuv420p", "-f", "mov", str(path)]
    subprocess.run(cmd, check=True, timeout=120)
    return path


def _run(*argv: str, capsys: pytest.CaptureFixture[str]) -> tuple[int, list[dict[str, Any]]]:
    code = main(list(argv))
    text = capsys.readouterr().out
    assert text.isascii(), text  # any console can show it
    return code, [json.loads(line) for line in text.splitlines()]


def _records(out: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in out.read_bytes().decode("utf-8").splitlines()]


@pytest.fixture
def clip(tmp_path: Path) -> Path:
    if FFMPEG is None:
        pytest.skip("needs ffmpeg")
    folder = tmp_path / "footage"
    folder.mkdir()
    return _media(folder / "a.mp4")


class _Any:
    def __eq__(self, other: object) -> bool:
        return True


ANY = _Any()


# --------------------------------------------------------------------------- F4: names, consoles


@needs_ffmpeg
def test_a_non_latin_name_on_a_cp1252_console(clip: Path, tmp_path: Path) -> None:
    """As on a Windows console whose code page cannot show the name: no crash, ASCII JSON on
    the console, the real name in the UTF-8 facts file."""
    name = "カメラ Ä 1.mp4"
    shutil.copy(clip, clip.with_name(name))
    out = tmp_path / "facts.jsonl"
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    env.update(PYTHONPATH=str(SRC_ROOT), PYTHONIOENCODING="cp1252", PYTHONUTF8="0")
    done = subprocess.run(
        [
            *(sys.executable, "-m", "aar.media.facts", str(clip.parent), "--out", str(out)),
            *("--pc-folder", f"{clip.parent}={PC}"),
        ],
        capture_output=True,
        timeout=120,
        env=env,
        cwd=tmp_path,
    )
    assert done.returncode == 0, done.stderr.decode("utf-8", "replace")
    assert done.stderr == b""
    printed = [json.loads(line) for line in done.stdout.decode("ascii").splitlines()]
    assert [line["path"] for line in printed[:2]] == [f"{PC}\\a.mp4", f"{PC}\\{name}"]
    assert printed[-1]["ok"] is True
    assert [r["path"] for r in _records(out)] == [f"{PC}\\a.mp4", f"{PC}\\{name}"]
    assert name.encode("utf-8") in out.read_bytes()
    assert sorted(load_facts(out)) == sorted(p.lower() for p in (f"{PC}\\a.mp4", f"{PC}\\{name}"))


@needs_ffmpeg
@pytest.mark.skipif(os.name != "posix", reason="POSIX names are bytes")
def test_a_name_that_is_not_unicode_is_its_own_error_line(
    clip: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = clip.parent
    try:
        fd = os.open(os.fsencode(folder) + b"/bad\xff name.mp4", os.O_WRONLY | os.O_CREAT, 0o644)
    except OSError:
        pytest.skip("this file system refuses names that are not UTF-8")
    os.write(fd, clip.read_bytes())
    os.close(fd)
    out = tmp_path / "facts.jsonl"
    code, lines = _run(
        str(folder), "--out", str(out), "--pc-folder", f"{folder}={PC}", capsys=capsys
    )
    assert code == 2
    records = _records(out)  # strict UTF-8
    bad = next(r for r in records if "bad" in r["path"])
    assert bad["path"] == f"{PC}\\bad\\udcff name.mp4"
    assert bad["error"]["code"] == "unrepresentable_name"
    assert any(r["path"] == f"{PC}\\a.mp4" and "ffprobe" in r for r in records)
    assert lines[-1]["failed"] == 1
    failures: dict[str, dict[str, str]] = {}
    load_facts(out, failures)  # the hub reads it, the bad name as "not measured"
    assert failures[bad["path"].lower()]["code"] == "unrepresentable_name"


# --------------------------------------------------------------------------- F5: folders


def test_what_a_folder_holds_that_is_not_measured_is_listed_and_no_media_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = tmp_path / "cards"
    folder.mkdir()
    pc = ("--pc-folder", f"{folder}={PC}")
    out = tmp_path / "facts.jsonl"
    # An empty folder.
    code, lines = _run(str(folder), "--out", str(out), *pc, capsys=capsys)
    assert code == 2 and lines[0]["error"]["code"] == "no_media"
    assert lines[0]["skipped"]["count"] == 0 and not out.exists()
    # Only files of other kinds, a subfolder and a dangling link.
    for name in ("A001.MKV", "b.avi", "c.mts", "still.jpg", "notes"):
        (folder / name).write_bytes(b"x")
    (folder / "sub").mkdir()
    (folder / "gone.mp4").symlink_to(folder / "nowhere.mp4")
    code, lines = _run(str(folder), "--out", str(out), *pc, capsys=capsys)
    assert code == 2 and lines == [{"ok": False, "error": lines[0]["error"], "skipped": ANY}]
    skipped = lines[0]["skipped"]
    assert skipped["count"] == 7
    assert skipped["by_reason"] == {
        "dangling_link": 1,
        "not_a_media_suffix:(none)": 1,
        "not_a_media_suffix:.avi": 1,
        "not_a_media_suffix:.jpg": 1,
        "not_a_media_suffix:.mkv": 1,
        "not_a_media_suffix:.mts": 1,
        "subfolder": 1,
    }
    assert {e["path"] for e in skipped["first"]} == {str(p) for p in folder.iterdir()}
    assert not out.exists()
    # A file named explicitly is measured whatever its suffix (here it is not media: an error).
    code, lines = _run(str(folder / "c.mts"), "--out", str(out), *pc, capsys=capsys)
    assert code == 2 and lines[0]["error"]["code"] in ("probe_failed", "tool_missing")


def test_an_unreadable_folder_is_a_typed_refusal(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = tmp_path / "locked"
    folder.mkdir()
    (folder / "a.mp4").write_bytes(b"x")
    out = tmp_path / "facts.jsonl"

    def refuse(path: Any) -> Any:
        raise PermissionError(errno.EACCES, "Permission denied", str(path))

    monkeypatch.setattr(facts_mod.os, "scandir", refuse)
    code, lines = _run(
        str(folder), "--out", str(out), "--pc-folder", f"{folder}={PC}", capsys=capsys
    )
    assert code == 2 and lines[0]["error"]["code"] == "folder_unreadable"
    assert "Permission denied" in lines[0]["error"]["message"] and not out.exists()


def test_the_run_has_a_deadline_and_says_what_it_did_not_measure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every probe hangs (a stand-in ffprobe that sleeps): the first is stopped at the run's
    deadline, not the per-file timeout, and the rest are listed ``deadline``; the file is
    written."""
    if os.name != "posix":
        pytest.skip("the stand-in ffprobe is a shell script")
    tool = tmp_path / "bin" / "ffprobe"
    tool.parent.mkdir()
    tool.write_text("#!/bin/sh\nexec sleep 60\n")
    tool.chmod(0o755)
    monkeypatch.setenv("AAR_FFPROBE", str(tool))
    folder = tmp_path / "cards"
    folder.mkdir()
    for name in ("a.mp4", "b.mp4", "c.mp4"):
        (folder / name).write_bytes(b"not probed")
    out = tmp_path / "facts.jsonl"
    started = time.monotonic()
    code, lines = _run(
        str(folder),
        *("--out", str(out), "--pc-folder", f"{folder}={PC}", "--deadline", "1"),
        capsys=capsys,
    )
    assert time.monotonic() - started < 30
    assert code == 2
    codes = [line["error"]["code"] for line in lines[:3]]
    assert codes == ["timeout", "deadline", "deadline"]
    assert [r["error"]["code"] for r in _records(out)] == codes


def test_the_deadline_by_the_clock(tmp_path: Path) -> None:
    ticks = iter([0.0, 10.0, 20.0, 30.0])
    files = [Target(tmp_path / f"{n}.mp4", f"E:\\{n}.mp4") for n in "abc"]
    result = measure(files, deadline_s=15.0, clock=lambda: next(ticks))
    assert [e["error"]["code"] for e in result.summary] == ["missing", "deadline", "deadline"]
    assert result.data.count(b"\n") == 3 and not result.interrupted


@needs_ffmpeg
def test_an_interrupted_run_writes_what_it_measured(
    clip: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ("b.mp4", "c.mp4"):
        shutil.copy(clip, clip.with_name(name))
    real = facts_mod.run_tool
    calls: list[int] = []

    def ctrl_c_on_the_second(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        if len(calls) == 2:
            raise KeyboardInterrupt
        return real(*args, **kwargs)

    monkeypatch.setattr(facts_mod, "run_tool", ctrl_c_on_the_second)
    out = tmp_path / "facts.jsonl"
    code, lines = _run(
        str(clip.parent), "--out", str(out), "--pc-folder", f"{clip.parent}={PC}", capsys=capsys
    )
    assert code == 130 and lines[-1]["interrupted"] is True and lines[-1]["ok"] is False
    assert [line.get("error", {}).get("code") for line in lines[:3]] == [
        None,
        "interrupted",
        "interrupted",
    ]
    records = _records(out)
    assert "ffprobe" in records[0] and records[1]["error"]["code"] == "interrupted"
    assert len(calls) == 2  # nothing probed after the interrupt


# --------------------------------------------------------------------------- caps and output


@needs_ffmpeg
def test_the_byte_cap_holds_for_error_lines_and_exactly_at_the_limit(
    clip: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    shutil.copy(clip, clip.with_name("b.mp4"))
    pc = ("--pc-folder", f"{clip.parent}={PC}")
    whole = tmp_path / "whole.jsonl"
    assert _run(str(clip.parent), "--out", str(whole), *pc, capsys=capsys)[0] == 0
    size = len(whole.read_bytes())
    # Exactly the size: everything fits.
    exact = tmp_path / "exact.jsonl"
    code, lines = _run(
        str(clip.parent), "--out", str(exact), "--max-bytes", str(size), *pc, capsys=capsys
    )
    assert code == 0 and exact.read_bytes() == whole.read_bytes()
    # One byte less: the second file is listed output_cap (its error line fits).
    short = tmp_path / "short.jsonl"
    code, lines = _run(
        str(clip.parent), "--out", str(short), "--max-bytes", str(size - 1), *pc, capsys=capsys
    )
    assert code == 2 and lines[1]["error"]["code"] == "output_cap"
    assert len(short.read_bytes()) <= size - 1
    # One byte: nothing fits, not even an error line; the console still lists every file.
    tiny = tmp_path / "tiny.jsonl"
    code, lines = _run(str(clip.parent), "--out", str(tiny), "--max-bytes", "1", *pc, capsys=capsys)
    assert code == 2 and tiny.read_bytes() == b""
    assert [line["in_file"] for line in lines[:2]] == [False, False]
    assert lines[-1]["not_in_file"] == 2
    # Exactly --max-files files is fine; one more is refused.
    code, lines = _run(
        str(clip.parent),
        "--out",
        str(tmp_path / "two.jsonl"),
        "--max-files",
        "2",
        *pc,
        capsys=capsys,
    )
    assert code == 0
    code, lines = _run(
        str(clip.parent),
        "--out",
        str(tmp_path / "one.jsonl"),
        "--max-files",
        "1",
        *pc,
        capsys=capsys,
    )
    assert code == 2 and lines[0]["error"]["code"] == "too_many_files"


@pytest.mark.skipif(os.name != "posix", reason="the hard-link fallback is POSIX's")
def test_without_hard_links_the_output_is_still_never_replaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "facts.jsonl"
    theirs = b"another run's facts\n"

    def no_links(src: Any, dst: Any) -> None:
        raise OSError(errno.EPERM, "Operation not permitted")

    def racing(real: Any, index: int) -> Any:
        """``real``, but another run creates the output just before the call that would give
        ours its name (after any check for it)."""

        def call(*args: Any, **kwargs: Any) -> Any:
            if Path(os.fsdecode(args[index])) == out and not out.exists():
                out.write_bytes(theirs)
            return real(*args, **kwargs)

        return call

    with monkeypatch.context() as patch:
        patch.setattr(facts_mod.os, "link", no_links)
        patch.setattr(facts_mod.os, "rename", racing(os.rename, 1))
        patch.setattr(facts_mod.os, "open", racing(os.open, 0))
        with pytest.raises(FileExistsError):
            write_new(out, b"ours\n")
    assert out.read_bytes() == theirs
    assert sorted(p.name for p in tmp_path.iterdir()) == ["facts.jsonl"]  # no .part left
    # Without a racer: written under its name, the temporary file gone.
    other = tmp_path / "other.jsonl"
    monkeypatch.setattr(
        facts_mod.os, "link", lambda *a: (_ for _ in ()).throw(OSError(errno.EPERM, "no"))
    )
    write_new(other, b"ours\n")
    assert other.read_bytes() == b"ours\n"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["facts.jsonl", "other.jsonl"]


def test_a_failed_write_leaves_no_partial_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def full(fd: int) -> None:
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(facts_mod.os, "fsync", full)
    with pytest.raises(OSError):
        write_new(tmp_path / "facts.jsonl", b"x\n")
    assert list(tmp_path.iterdir()) == []
    # ... and in the no-hard-link fallback, which writes the name itself.
    monkeypatch.undo()
    real_fsync = os.fsync
    calls: list[int] = []

    def full_second_time(fd: int) -> None:
        calls.append(fd)
        if len(calls) == 2:
            raise OSError(errno.ENOSPC, "No space left on device")
        real_fsync(fd)

    monkeypatch.setattr(facts_mod.os, "fsync", full_second_time)
    monkeypatch.setattr(
        facts_mod.os, "link", lambda *a: (_ for _ in ()).throw(OSError(errno.EPERM, "no"))
    )
    with pytest.raises(OSError):
        write_new(tmp_path / "facts.jsonl", b"x\n")
    assert list(tmp_path.iterdir()) == []


@needs_ffmpeg
def test_a_file_edited_in_place_to_the_same_size_is_changed_while_measured(
    clip: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = facts_mod.run_tool
    size = clip.stat().st_size

    def edit_while_probing(*args: Any, **kwargs: Any) -> Any:
        result = real(*args, **kwargs)
        data = bytearray(clip.read_bytes())
        data[-1] ^= 0xFF
        with clip.open("r+b") as f:  # in place: same file, same size
            f.write(bytes(data))
        st = clip.stat()
        os.utime(clip, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
        return result

    monkeypatch.setattr(facts_mod, "run_tool", edit_while_probing)
    record = measure_file(Target(clip, "E:\\a.mp4"))
    assert clip.stat().st_size == size
    assert record["error"]["code"] == "changed_while_measured"


# --------------------------------------------------------------------------- Windows, simulated


@needs_ffmpeg
def test_file_times_windows_cannot_convert_are_error_lines_not_crashes(
    clip: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = Target(clip, "E:\\a.mp4")
    # Before 1970: Windows' localtime raises OSError (simulated; Linux converts it).
    os.utime(clip, ns=(0, -86_400 * 1_000_000_000))
    real_localtime = time.localtime

    def windows_localtime(seconds: float | None = None) -> time.struct_time:
        if seconds is not None and seconds < 0:
            raise OSError(errno.EINVAL, "Invalid argument")
        return real_localtime(seconds)

    with monkeypatch.context() as patch:
        patch.setattr(facts_mod.time, "localtime", windows_localtime)
        record = measure_file(target)
    assert record["error"]["code"] == "bad_time", record
    # Past the year 9999 (real on every system): no date the facts can hold.
    far = 253_402_300_800 * 1_000_000_000
    os.utime(clip, ns=(far, far))
    record = measure_file(target)
    assert record["error"]["code"] == "bad_time" and "10000" in record["error"]["message"]
    # A creation time that cannot be written is left out; the file is still measured.
    os.utime(clip, ns=(1_700_000_000 * 10**9, 1_700_000_000 * 10**9))
    notes: list[str] = []
    with monkeypatch.context() as patch:
        patch.setattr(facts_mod, "birth_ns", lambda st: far)
        record = measure_file(target, notes=notes)
    assert "error" not in record and "ctime_utc" not in record
    assert notes and "creation time is left out" in notes[0]  # said, not swallowed
    with monkeypatch.context() as patch:
        patch.setattr(facts_mod, "birth_ns", lambda st: 1_600_000_000 * 10**9)
        record = measure_file(target)
    assert record["ctime_utc"] == "2020-09-13T12:26:40.0000000Z"


def test_the_creation_time_is_read_where_each_system_keeps_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ns = 1_600_000_000_123_456_700
    assert birth_ns(SimpleNamespace(st_birthtime_ns=ns, st_ctime_ns=1)) == ns  # Windows 3.12+
    assert birth_ns(SimpleNamespace(st_birthtime=1_600_000_000.5)) == 1_600_000_000_500_000_000
    assert birth_ns(SimpleNamespace(st_birthtime_ns=0)) is None  # unknown
    assert birth_ns(SimpleNamespace(st_birthtime=float("nan"))) is None
    linux = SimpleNamespace(st_ctime_ns=ns)  # Linux: st_ctime is a status change, never used
    monkeypatch.setattr(facts_mod, "_WINDOWS", False)
    assert birth_ns(linux) is None
    monkeypatch.setattr(facts_mod, "_WINDOWS", True)
    monkeypatch.setattr(facts_mod, "sys", SimpleNamespace(version_info=(3, 11, 9)))
    assert birth_ns(linux) == ns  # Windows before 3.12: st_ctime is the creation time
    monkeypatch.setattr(facts_mod, "sys", SimpleNamespace(version_info=(3, 12, 0)))
    assert birth_ns(linux) is None  # 3.12+ has st_birthtime_ns; st_ctime is deprecated there


@pytest.mark.skipif(os.name != "posix", reason="simulates Windows' lookup with POSIX paths")
def test_on_windows_ffprobe_is_never_taken_from_the_current_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media = tmp_path / "media"
    media.mkdir()
    planted = media / "ffprobe.exe"  # a program left in the folder being measured
    planted.write_text("#!/bin/sh\necho planted\n")
    real = tmp_path / "bin" / "ffprobe.exe"
    real.parent.mkdir()
    real.write_text("#!/bin/sh\necho real\n")
    for program in (planted, real):
        program.chmod(program.stat().st_mode | stat.S_IXUSR)
    monkeypatch.chdir(media)
    monkeypatch.setattr(tools, "_WINDOWS", True)
    monkeypatch.delenv("AAR_FFPROBE", raising=False)
    monkeypatch.setenv("PATH", os.pathsep.join([".", "", "media", str(real.parent)]))
    assert tools.resolve_tool("ffprobe") == str(real)
    monkeypatch.setenv("PATH", os.pathsep.join([".", "", "media"]))
    with pytest.raises(tools.MediaToolMissing):
        tools.resolve_tool("ffprobe")
    # An override must be absolute or a bare name looked up the same way.
    monkeypatch.setenv("PATH", str(real.parent))
    for relative in ("./ffprobe.exe", "media/ffprobe.exe"):
        monkeypatch.setenv("AAR_FFPROBE", relative)
        with pytest.raises(tools.MediaToolMissing):
            tools.resolve_tool("ffprobe")
    monkeypatch.setenv("AAR_FFPROBE", "ffprobe.exe")
    assert tools.resolve_tool("ffprobe") == str(real)
    monkeypatch.setenv("AAR_FFPROBE", str(real))
    assert tools.resolve_tool("ffprobe") == str(real)
