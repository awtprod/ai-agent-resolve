"""S25d: ``aar media facts``, the edit PC's facts command. Its output must parse with the hub's
facts model (:mod:`aar.drp.mediapool.facts`); a file it cannot measure is listed with its
error, never skipped; its caps hold; the same files give the same bytes; it runs from ``src``
with the standard library only, and it never changes the media."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from media_support import FFMPEG, needs_ffmpeg

from aar.drp.mediapool.errors import MediaPoolError
from aar.drp.mediapool.facts import facts_from_json
from aar.drp.mediapool.gate import load_facts
from aar.drp.mediapool.kinds import kind_of_facts
from aar.media import facts as facts_mod
from aar.media.facts import Target, local_text, main, targets, utc_text

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
PC = "E:\\Show 1\\Footage"


def _media(path: Path, *, seconds: float = 1.0, audio: bool = True, timecode: bool = True) -> Path:
    assert FFMPEG is not None
    cmd = [FFMPEG, "-hide_banner", "-nostdin", "-v", "error", "-y"]
    cmd += ["-f", "lavfi", "-i", f"testsrc2=size=320x180:rate=30000/1001:duration={seconds}"]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=d={seconds}:r=48000", "-c:a", "pcm_s16le"]
    if timecode:
        cmd += ["-timecode", "01:00:00:00"]
    cmd += ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-f", "mov"]
    subprocess.run([*cmd, str(path)], check=True, timeout=120)
    return path


@pytest.fixture
def show(tmp_path: Path) -> Path:
    folder = tmp_path / "footage"
    folder.mkdir()
    _media(folder / "B.mp4")
    _media(folder / "a.mov", audio=False, timecode=False)
    (folder / "broken.mov").write_bytes(b"this is not media")
    (folder / "notes.txt").write_text("not a media name")
    (folder / ".hidden.mp4").write_bytes(b"x")
    return folder


def _run(*argv: str, capsys: pytest.CaptureFixture[str]) -> tuple[int, list[dict]]:
    code = main(list(argv))
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    return code, lines


def _digests(folder: Path) -> dict[str, tuple[str, int]]:
    return {
        p.name: (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
        for p in sorted(folder.iterdir())
    }


@needs_ffmpeg
def test_the_facts_parse_with_the_hub_model_and_a_bad_file_is_listed(
    show: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    before = _digests(show)
    out = tmp_path / "facts.jsonl"
    code, lines = _run(str(show), "--out", str(out), "--pc-folder", f"{show}={PC}", capsys=capsys)
    assert code == 2  # one file could not be measured
    assert [(line["path"], line["ok"]) for line in lines[:3]] == [
        (f"{PC}\\a.mov", True),
        (f"{PC}\\B.mp4", True),
        (f"{PC}\\broken.mov", False),
    ]
    assert lines[2]["error"]["code"] == "probe_failed"
    assert lines[-1] == {
        "failed": 1,
        "files": 3,
        "ok": False,
        "repeated_paths_dropped": 0,
        "written": str(out),
    }
    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [r["path"] for r in records] == [f"{PC}\\a.mov", f"{PC}\\B.mp4", f"{PC}\\broken.mov"]
    assert records[2] == {
        "path": f"{PC}\\broken.mov",
        "exists": True,
        "error": records[2]["error"],
    }
    # The two measured lines are facts the hub accepts, each file with its own measurements.
    mp4 = facts_from_json(records[1])
    assert mp4.path == f"{PC}\\B.mp4" and mp4.size == (show / "B.mp4").stat().st_size
    assert mp4.mtime_utc_ns // 100 == (show / "B.mp4").stat().st_mtime_ns // 100
    assert [s.codec_type for s in mp4.streams if s.codec_type != "data"] == ["video", "audio"]
    kind_of_facts(mp4)  # the writer can classify it (an h264 .mp4 with 16-bit PCM)
    mov = facts_from_json(records[0])
    assert [s.codec_type for s in mov.streams] == ["video"]
    # The hub's loader keeps the measured files and records the failure under its path.
    failures: dict[str, dict[str, str]] = {}
    loaded = load_facts(out, failures)
    assert sorted(loaded) == [f"{PC}\\a.mov".lower(), f"{PC}\\b.mp4".lower()]
    assert failures[f"{PC}\\broken.mov".lower()]["code"] == "probe_failed"
    # Read-only on the media: every byte and modification time as before.
    assert _digests(show) == before


@needs_ffmpeg
def test_the_output_is_the_same_bytes_every_run(
    show: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    outs = []
    for n, order in enumerate(([show / "B.mp4", show / "a.mov"], [show / "a.mov", show / "B.mp4"])):
        out = tmp_path / f"facts{n}.jsonl"
        code, _ = _run(
            *map(str, order), "--out", str(out), "--pc-folder", f"{show}={PC}", capsys=capsys
        )
        assert code == 0
        outs.append(out.read_bytes())
    assert outs[0] == outs[1]
    assert outs[0].count(b"\n") == 2


@needs_ffmpeg
def test_the_caps_hold(show: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pc = ("--pc-folder", f"{show}={PC}")
    # Too many files: refused before any is measured, nothing written.
    out = tmp_path / "many.jsonl"
    code, lines = _run(str(show), "--out", str(out), "--max-files", "2", *pc, capsys=capsys)
    assert (
        code == 2
        and lines == [{"ok": False, "error": lines[0]["error"]}]
        and lines[0]["error"]["code"] == "too_many_files"
    )
    assert not out.exists()
    # The output cap: the file that would pass it, and every later one, listed output_cap.
    out = tmp_path / "capped.jsonl"
    code, lines = _run(str(show), "--out", str(out), "--max-bytes", "300", *pc, capsys=capsys)
    assert code == 2
    assert [line.get("error", {}).get("code") for line in lines[:3]] == [
        "output_cap",
        "output_cap",
        "output_cap",
    ]
    assert len(out.read_bytes()) < 1000
    # A probe that runs past its timeout is killed and listed.
    out = tmp_path / "slow.jsonl"
    code, lines = _run(
        str(show / "B.mp4"), "--out", str(out), "--timeout", "1e-9", *pc, capsys=capsys
    )
    assert code == 2 and lines[0]["error"]["code"] == "timeout"
    # An existing output is never overwritten.
    code, lines = _run(str(show / "B.mp4"), "--out", str(out), *pc, capsys=capsys)
    assert code == 2 and lines[0]["error"]["code"] == "output_exists"


def test_odd_files_are_listed_and_paths_need_a_pc_name(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "empty.mp4").write_bytes(b"")
    if not hasattr(os, "mkfifo"):
        pytest.skip("needs a FIFO")
    os.mkfifo(tmp_path / "pipe.mov")  # never opened: ffprobe would wait on it
    out = tmp_path / "facts.jsonl"
    pc = ("--pc-folder", f"{tmp_path}={PC}")
    names = ("gone.mp4", "empty.mp4", "pipe.mov")
    code, lines = _run(*(str(tmp_path / n) for n in names), "--out", str(out), *pc, capsys=capsys)
    assert code == 2
    errors = {line["path"].rsplit("\\", 1)[1]: line["error"]["code"] for line in lines[:3]}
    assert errors == {"empty.mp4": "empty", "pipe.mov": "not_a_file", "gone.mp4": "missing"}
    gone = next(json.loads(x) for x in out.read_text().splitlines() if "gone" in x)
    assert gone["exists"] is False
    # Without a PC name for the file (not on Windows, no --pc-folder): refused, nothing written.
    code, lines = _run(str(tmp_path / "empty.mp4"), "--out", str(tmp_path / "x"), capsys=capsys)
    assert code == 2 and lines[0]["error"]["code"] == "not_a_pc_path"


def test_pc_paths_and_times() -> None:
    files, repeated = targets(
        [Path("/media/show/Proxy/A.mp4"), Path("/media/show/proxy2/b.mov")],
        [("/media/show", "E:\\Show"), ("/media/show/proxy2", "\\\\nas\\cams\\P2")],
    )
    # in the order of the PC paths, case folded; the longest --pc-folder that holds a file wins
    assert [t.pc_path for t in files] == ["\\\\nas\\cams\\P2\\b.mov", "E:\\Show\\Proxy\\A.mp4"]
    assert repeated == 0
    files, repeated = targets(
        [Path("/media/show/A.mp4"), Path("/media/show/A.mp4")], [("/media/show", "E:\\Show")]
    )
    assert len(files) == 1 and repeated == 1
    assert utc_text(1_773_392_706_123_456_789) == "2026-03-13T09:05:06.1234567Z"
    text = local_text(1_773_392_706)
    from aar.drp.mediapool.facts import parse_local_time

    parse_local_time(text)  # the hub's parser takes it (weekday, padding, English names)
    assert isinstance(Target(Path("x"), "E:\\x"), Target)


def test_the_hub_loader_refuses_bad_error_lines_and_never_blocks(tmp_path: Path) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"path": "E:\\x.mp4", "error": "no"}) + "\n")
    with pytest.raises(MediaPoolError) as caught:
        load_facts(bad)
    assert caught.value.code == "bad_facts"
    extra = tmp_path / "extra.jsonl"
    line = {"path": "E:\\x.mp4", "exists": True, "error": {"code": "x", "message": "m"}, "y": 1}
    extra.write_text(json.dumps(line) + "\n")
    with pytest.raises(MediaPoolError):
        load_facts(extra)
    if hasattr(os, "mkfifo"):
        fifo = tmp_path / "facts.fifo"
        os.mkfifo(fifo)
        with pytest.raises(MediaPoolError) as caught:
            load_facts(fifo)
        assert "regular" in caught.value.message


_RUN_AND_LIST_MODULES = """
import json, runpy, sys
listing, sys.argv = sys.argv[1], ["aar.media.facts", *sys.argv[2:]]
try:
    runpy.run_module("aar.media.facts", run_name="__main__", alter_sys=True)
except SystemExit as exc:
    code = exc.code
else:
    code = 0
with open(listing, "w") as f:
    json.dump(sorted(name for name, mod in sys.modules.items() if mod is not None), f)
sys.exit(code)
"""


@needs_ffmpeg
def test_runs_from_src_with_the_standard_library_only(show: Path, tmp_path: Path) -> None:
    """As on the edit PC: ``python -S -m aar.media.facts`` with only ``PYTHONPATH=src``."""
    listing = tmp_path / "modules.json"
    out = tmp_path / "facts.jsonl"
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    env["PYTHONPATH"] = str(SRC_ROOT)
    done = subprocess.run(
        [
            *(sys.executable, "-S", "-c", _RUN_AND_LIST_MODULES, str(listing)),
            *(str(show / "B.mp4"), "--out", str(out), "--pc-folder", f"{show}={PC}"),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
        cwd=tmp_path,
    )
    assert done.returncode == 0, done.stderr
    modules = json.loads(listing.read_text())
    imported = {name.split(".")[0] for name in modules}
    assert sorted(imported - set(sys.stdlib_module_names) - {"aar", "__main__"}) == []
    assert "pydantic" not in modules and "aar.drp" not in modules
    assert facts_from_json(out.read_text(encoding="utf-8").splitlines()[0]).path.endswith("B.mp4")


def test_the_cli_names_the_command() -> None:
    from aar.cli import main as aar_main

    assert aar_main(["media", "facts", "--help"]) == 0
    assert facts_mod.MAX_LINE_BYTES == 1024 * 1024
