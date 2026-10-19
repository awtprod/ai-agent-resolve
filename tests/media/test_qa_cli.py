"""``aar qa check``: exit codes, text and JSON output, options and input errors.

Live ffmpeg runs on tiny fixtures from ``media_support`` (a stereo 1 kHz sine at peak P dBFS
measures P LUFS and P dBTP). Tests that stop at an input error never reach ffmpeg, so they are
not marked ``needs_ffmpeg``.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import pytest
from media_support import amp, gapped, needs_ffmpeg, plan_dict, render, sine, write_json

from aar.cli import main

SHORT_DB = -14.5  # the ccn short loudness target
CHECK_IDS = [
    "decode",
    "loudness",
    "clipping",
    "longest-silence",
    "size-cap",
    "duration",
    "join-quiet",
    "join-retranscribe",
    "caption-coverage",
]
CHECK_KEYS = {
    "id",
    "status",
    "message",
    "skip_reason",
    "measured",
    "target",
    "details",
    "incomplete",
}
LINE = re.compile(r"^(PASS|WARN|FAIL|SKIP)\s+([a-z-]+): ", re.MULTILINE)


def run(capsys: pytest.CaptureFixture[str], *argv: object) -> tuple[int, str, str]:
    """Run ``aar qa check ARGV`` and return (exit code, stdout, stderr)."""
    code = main(["qa", "check", *(str(a) for a in argv)])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def report_json(capsys: pytest.CaptureFixture[str], *argv: object) -> tuple[int, dict[str, Any]]:
    """Run with ``--json`` and return (exit code, report); the code must follow ``ok``."""
    code, out, err = run(capsys, *argv, "--json")
    assert err == ""
    report = json.loads(out)
    assert code == (0 if report["ok"] else 1)
    return code, report


def error_json(capsys: pytest.CaptureFixture[str], *argv: object) -> dict[str, Any]:
    """Run with ``--json`` expecting an input error (exit 2); return the ``error`` object."""
    code, out, err = run(capsys, *argv, "--json")
    assert code == 2
    assert err == ""
    body = json.loads(out)
    assert body["ok"] is False
    assert set(body["error"]) == {"type", "message", "details"}
    return body["error"]


def checks_by_id(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {c["id"]: c for c in report["checks"]}


def write_profile(path: Path, *, name: str = "x", **qa: float) -> Path:
    """A profile with the four required [qa] keys (ccn's values unless overridden), no rules."""
    keys = {
        "longform_lufs": -22.0,
        "short_lufs": -14.5,
        "lufs_tolerance": 0.5,
        "review_mp4_max_mb": 20,
        **qa,
    }
    lines = [f'name = "{name}"', "[qa]", *(f"{key} = {value}" for key, value in keys.items())]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def make_plan(
    folder: Path,
    *,
    sources: list[dict[str, Any]] | None = None,
    segments: list[dict[str, Any]] | None = None,
    captions: list[dict[str, Any]] | None = None,
    profile: str = "ccn",
) -> Path:
    """A 50 fps plan: by default one 100-frame (2.0 s) segment of ``src.wav``, source id "a"."""
    data = plan_dict(
        sources or [{"id": "a", "media": "src.wav"}],
        segments or [{"track": 1, "source": "a", "src_in": 0, "src_out": 100}],
        captions=captions,
    )
    data["profile"] = profile
    return write_json(folder / "plan.json", data)


def plain_file(tmp_path: Path, name: str = "render.mp4") -> Path:
    """A file that exists but is not media; input errors are raised before it is read."""
    path = tmp_path / name
    path.write_bytes(b"not media\n")
    return path


@pytest.fixture(scope="module")
def short_wav(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A 2 s stereo sine at the short target: -14.5 LUFS, -14.5 dBTP."""
    return sine(tmp_path_factory.mktemp("qa-cli") / "short.wav", SHORT_DB)


# ---------------------------------------------------------------- reports


@needs_ffmpeg
def test_a_short_at_target_passes_and_every_check_has_a_line(
    capsys: pytest.CaptureFixture[str], short_wav: Path
):
    code, out, err = run(capsys, short_wav, "--kind", "short")
    assert (code, err) == (0, "")
    lines = out.splitlines()
    assert lines[0] == (
        f"qa {short_wav} (short, profile ccn): PASS: 4 pass, 0 warn, 0 fail, 5 skipped"
    )
    assert lines[1] == (
        "not checked or not fully checked: size-cap, duration, join-quiet, join-retranscribe, "
        "caption-coverage"
    )
    # No fail or warn, so no detail lines: the head, the unchecked line, one line per check.
    assert len(lines) == 11
    assert LINE.findall(out) == [
        ("PASS", "decode"),
        ("PASS", "loudness"),
        ("PASS", "clipping"),
        ("PASS", "longest-silence"),
        ("SKIP", "size-cap"),
        ("SKIP", "duration"),
        ("SKIP", "join-quiet"),
        ("SKIP", "join-retranscribe"),
        ("SKIP", "caption-coverage"),
    ]
    assert "PASS  loudness: Integrated -14.5 LUFS (target -14.5 ± 0.5)" in out


@needs_ffmpeg
def test_a_short_level_fails_the_longform_target(
    capsys: pytest.CaptureFixture[str], short_wav: Path
):
    code, out, err = run(capsys, short_wav, "--kind", "longform")
    assert (code, err) == (1, "")
    assert out.splitlines()[0].startswith(f"qa {short_wav} (longform, profile ccn): FAIL: ")
    (loudness_line,) = [ln for ln in out.splitlines() if ln.startswith("FAIL  loudness:")]
    assert "7.5 LU above the -22 target" in loudness_line

    code, report = report_json(capsys, short_wav, "--kind", "longform")
    assert code == 1
    assert (report["kind"], report["profile"], report["file"]) == (
        "longform",
        "ccn",
        str(short_wav),
    )
    assert report["status"] == "fail"
    assert report["ok"] is False
    assert report["counts"] == {"pass": 3, "warn": 0, "fail": 1, "skipped": 5}
    loudness = checks_by_id(report)["loudness"]
    assert loudness["status"] == "fail"
    assert loudness["measured"]["integrated_lufs"] == pytest.approx(-14.5, abs=0.05)
    assert loudness["measured"]["true_peak_dbtp"] == pytest.approx(-14.5, abs=0.05)
    assert loudness["target"] == {
        "integrated_lufs": -22.0,
        "tolerance_lu": 0.5,
        "true_peak_max_dbtp": -1.0,
    }
    # Only loudness fails: the level is fine for the other checks.
    assert [c["status"] for c in report["checks"] if c["id"] != "loudness"].count("fail") == 0


@needs_ffmpeg
@pytest.mark.parametrize("kind", ["short", "longform"])
def test_json_report_has_the_nine_checks_in_order(
    capsys: pytest.CaptureFixture[str], short_wav: Path, kind: str
):
    _, report = report_json(capsys, short_wav, "--kind", kind)
    assert set(report) == {
        "file",
        "kind",
        "profile",
        "status",
        "ok",
        "counts",
        "unchecked",
        "media",
        "checks",
    }
    assert report["unchecked"] == CHECK_IDS[4:]
    assert report["kind"] == kind
    assert report["media"]["size_bytes"] == short_wav.stat().st_size
    assert [c["id"] for c in report["checks"]] == CHECK_IDS
    for entry in report["checks"]:
        assert set(entry) == CHECK_KEYS
        assert (entry["skip_reason"] is None) == (entry["status"] != "skipped")
    by_id = checks_by_id(report)
    size_cap = by_id["size-cap"]
    assert size_cap["status"] == "skipped"
    assert size_cap["skip_reason"] == "not-applicable"
    assert size_cap["target"] == {"max_mb": 20}
    assert by_id["duration"]["skip_reason"] == "no-plan"
    assert by_id["join-quiet"]["skip_reason"] == "no-plan"
    assert by_id["join-retranscribe"]["skip_reason"] == "no-plan"
    assert by_id["caption-coverage"]["skip_reason"] == "no-plan"


@needs_ffmpeg
@pytest.mark.parametrize(
    ("audio_dbfs", "code"),
    [(-20, 1), (-22, 0)],
    ids=["two-LU-hot", "on-the-longform-target"],
)
def test_review_renders_get_the_size_cap_and_the_longform_loudness_target(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, audio_dbfs: int, code: int
):
    path = render(tmp_path / "review.mp4", frames=30, audio=f"{amp(audio_dbfs)}*sin(2*PI*1000*t)")
    got, report = report_json(capsys, path, "--kind", "review")
    assert got == code
    by_id = checks_by_id(report)
    size_cap = by_id["size-cap"]
    assert size_cap["status"] == "pass"
    assert size_cap["skip_reason"] is None
    assert size_cap["measured"]["size_bytes"] == path.stat().st_size
    assert size_cap["measured"]["size_mb"] < 1
    assert size_cap["target"] == {"max_mb": 20, "max_bytes": 20_000_000}
    loudness = by_id["loudness"]
    assert loudness["target"]["integrated_lufs"] == -22.0
    assert loudness["measured"]["integrated_lufs"] == pytest.approx(audio_dbfs, abs=0.3)
    assert loudness["status"] == ("pass" if code == 0 else "fail")
    assert [s["kind"] for s in report["media"]["streams"]] == ["video", "audio"]


@needs_ffmpeg
def test_review_size_cap_comes_from_the_profile(capsys: pytest.CaptureFixture[str], tmp_path: Path):
    path = render(tmp_path / "review.mp4", frames=30)
    assert path.stat().st_size > 1000
    profile = write_profile(tmp_path / "tiny.toml", review_mp4_max_mb=0.001)
    code, out, _ = run(capsys, path, "--kind", "review", "--profile", profile)
    assert code == 1
    (line,) = [ln for ln in out.splitlines() if ln.startswith("FAIL  size-cap:")]
    assert "over the 0.001 MB cap" in line

    _, report = report_json(capsys, path, "--kind", "review", "--profile", profile)
    size_cap = checks_by_id(report)["size-cap"]
    assert size_cap["status"] == "fail"
    assert size_cap["target"] == {"max_mb": 0.001, "max_bytes": 1000}
    assert report["profile"] == "x"


# ---------------------------------------------------------------- --silence-db


@needs_ffmpeg
@pytest.mark.parametrize("flag", [["--silence-db", "-36"], ["--silence-db=-36"]])
def test_silence_db_moves_the_threshold(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, flag: list[str]
):
    # A 1 s pause at -39 dBFS between two 0.5 s tones: audible-ish at -42, silent at -36.
    path = gapped(
        tmp_path / "gap.wav", before_s=0.5, gap_s=1.0, after_s=0.5, peak_dbfs=-10, gap_dbfs=-39
    )
    _, default = report_json(capsys, path, "--kind", "short")
    silence = checks_by_id(default)["longest-silence"]
    assert silence["status"] == "pass"
    assert silence["measured"]["silences"] == 0
    assert silence["measured"]["longest_silence_s"] == 0.0
    assert silence["target"] == {"silence_db": -42.0, "min_s": 0.2}

    _, lowered = report_json(capsys, path, "--kind", "short", *flag)
    silence = checks_by_id(lowered)["longest-silence"]
    assert silence["status"] == "pass"  # ccn sets no longest_silence_warn_s
    assert silence["measured"]["silences"] == 1
    assert silence["measured"]["longest_silence_s"] == pytest.approx(1.0, abs=0.01)
    assert silence["measured"]["start_s"] == pytest.approx(0.5, abs=0.01)
    assert silence["target"] == {"silence_db": -36.0, "min_s": 0.2}
    assert "-36 dB" in silence["message"]


@pytest.mark.parametrize(
    "flag",
    [["--silence-db", "0"], ["--silence-db", "5"], ["--silence-db", "nan"], ["--silence-db=-inf"]],
    ids=["zero", "positive", "nan", "minus-inf"],
)
def test_silence_db_must_be_a_finite_negative_number(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, flag: list[str]
):
    path = plain_file(tmp_path)
    code, out, err = run(capsys, path, "--kind", "short", *flag)
    assert (code, out) == (2, "")
    assert err == "error: --silence-db must be a negative number of dB\n"

    error = error_json(capsys, path, "--kind", "short", *flag)
    assert error == {
        "type": "invalid-argument",
        "message": "--silence-db must be a negative number of dB",
        "details": [],
    }


# ---------------------------------------------------------------- the render file


def test_a_missing_file_is_exit_two(capsys: pytest.CaptureFixture[str], tmp_path: Path):
    missing = tmp_path / "nope.mp4"
    code, out, err = run(capsys, missing, "--kind", "short")
    assert (code, out) == (2, "")
    assert err == f"error: file not found: {missing}\n"

    error = error_json(capsys, missing, "--kind", "short")
    assert error["type"] == "FileNotFoundError"
    assert error["message"] == f"file not found: {missing}"
    assert error["details"] == []


def test_a_directory_is_not_a_render(capsys: pytest.CaptureFixture[str], tmp_path: Path):
    code, out, err = run(capsys, tmp_path, "--kind", "short")
    assert (code, out) == (2, "")
    assert err == f"error: not a regular file: {tmp_path}\n"

    assert error_json(capsys, tmp_path, "--kind", "short")["type"] == "NotAFile"


@pytest.mark.parametrize(
    "argv",
    [[], ["--kind", "bogus"], ["--kind"], ["--kind", "short", "--bogus"]],
    ids=["no-kind", "bad-kind", "kind-without-value", "unknown-option"],
)
def test_usage_errors_are_exit_two(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, argv: list[str]
):
    code, out, err = run(capsys, plain_file(tmp_path), *argv)
    assert (code, out) == (2, "")
    assert "usage: aar" in err
    assert "error: " in err


@needs_ffmpeg
@pytest.mark.parametrize("content", [b"this is not a media file\n", b""], ids=["text", "empty"])
def test_an_unreadable_render_fails_decode_and_skips_the_rest(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, content: bytes
):
    path = tmp_path / "broken.mp4"
    path.write_bytes(content)
    code, out, err = run(capsys, path, "--kind", "short")
    assert (code, err) == (1, "")
    assert out.splitlines()[0] == (
        f"qa {path} (short, profile ccn): FAIL: 0 pass, 0 warn, 1 fail, 8 skipped"
    )
    assert LINE.findall(out)[:2] == [("FAIL", "decode"), ("SKIP", "loudness")]

    _, report = report_json(capsys, path, "--kind", "short")
    assert report["status"] == "fail"
    assert report["media"] is None
    assert report["counts"] == {"pass": 0, "warn": 0, "fail": 1, "skipped": 8}
    by_id = checks_by_id(report)
    assert by_id["decode"]["status"] == "fail"
    assert by_id["decode"]["measured"]["readable"] is False
    assert by_id["decode"]["skip_reason"] is None
    expected = {
        "loudness": "render-unreadable",
        "clipping": "render-unreadable",
        "longest-silence": "render-unreadable",
        "size-cap": "render-unreadable",
        "duration": "render-unreadable",
        "join-quiet": "no-plan",
        "join-retranscribe": "render-unreadable",
        "caption-coverage": "no-plan",
    }
    for check_id, reason in expected.items():
        assert (by_id[check_id]["status"], by_id[check_id]["skip_reason"]) == ("skipped", reason)


# ---------------------------------------------------------------- --plan and --transcript


@needs_ffmpeg
def test_a_plan_enables_the_duration_and_join_checks(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, short_wav: Path
):
    sine(tmp_path / "src.wav", -10)
    plan = make_plan(tmp_path)  # one 2.0 s segment: no cuts
    _, report = report_json(capsys, short_wav, "--kind", "short", "--plan", plan)
    assert report["status"] == "pass"
    assert report["profile"] == "ccn"  # the plan's own profile field
    by_id = checks_by_id(report)
    duration = by_id["duration"]
    assert duration["status"] == "pass"
    assert duration["measured"]["measured_from"] == "audio stream"
    assert duration["measured"]["render_s"] == pytest.approx(2.0, abs=0.001)
    assert duration["measured"]["diff_frames"] == pytest.approx(0.0, abs=0.05)
    assert duration["target"] == {"plan_frames": 100, "plan_s": 2.0, "tolerance_frames": 1}
    join_quiet = by_id["join-quiet"]
    assert join_quiet["status"] == "pass"
    assert join_quiet["measured"] == {"cuts": 0}
    assert join_quiet["skip_reason"] is None
    for check_id in ("join-retranscribe", "caption-coverage"):
        assert by_id[check_id]["status"] == "skipped"
        assert by_id[check_id]["skip_reason"] == "no-transcript"
    assert report["counts"] == {"pass": 6, "warn": 0, "fail": 0, "skipped": 3}


@needs_ffmpeg
def test_a_render_of_the_wrong_length_fails_the_duration_check(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    short = sine(tmp_path / "one-second.wav", SHORT_DB, seconds=1.0)
    plan = make_plan(tmp_path)
    code, out, _ = run(capsys, short, "--kind", "short", "--plan", plan)
    assert code == 1
    (line,) = [ln for ln in out.splitlines() if ln.startswith("FAIL  duration:")]
    assert "50.0 frames shorter than the plan" in line

    _, report = report_json(capsys, short, "--kind", "short", "--plan", plan)
    duration = checks_by_id(report)["duration"]
    assert duration["status"] == "fail"
    assert duration["measured"]["diff_frames"] == pytest.approx(-50.0, abs=0.05)


@needs_ffmpeg
def test_join_quiet_reads_plan_sources_next_to_the_plan_not_the_cwd(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    short_wav: Path,
):
    # src.wav: 1.0 s of tone, 0.5 s of digital silence, 1.0 s of tone. The plan cuts from
    # source seconds 0-1.0 to 1.24-2.24, so the incoming src_in sits inside the pause.
    plan_dir = tmp_path / "plan-dir"
    plan_dir.mkdir()
    gapped(plan_dir / "src.wav", before_s=1.0, gap_s=0.5, after_s=1.0, peak_dbfs=-10)
    segments = [
        {"track": 1, "source": "a", "src_in": 0, "src_out": 50},
        {"track": 1, "source": "a", "src_in": 62, "src_out": 112},
    ]
    plan = make_plan(plan_dir, segments=segments)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    code, report = report_json(capsys, short_wav, "--kind", "short", "--plan", plan)
    assert code == 0
    join_quiet = checks_by_id(report)["join-quiet"]
    assert join_quiet["status"] == "pass"
    assert join_quiet["measured"] == {"cuts": 1, "failed": 0, "unchecked": 0}
    (join,) = join_quiet["details"]
    assert (join["join"], join["source"], join["status"]) == (1, "a", "pass")
    assert join["record_s"] == 1.0
    assert join["record_tc"] == "00:00:01:00"
    assert join["src_in_s"] == pytest.approx(1.24)
    assert join["longest_quiet_ms"] >= 150
    assert join["quiet_ms_at_cut"] >= 150
    assert join["run_at_cut"]["start_s"] <= 1.24 <= join["run_at_cut"]["end_s"]


@needs_ffmpeg
def test_a_cut_in_the_middle_of_speech_fails_join_quiet(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, short_wav: Path
):
    sine(tmp_path / "src.wav", -10, seconds=2.5)  # no pause anywhere
    segments = [
        {"track": 1, "source": "a", "src_in": 0, "src_out": 50},
        {"track": 1, "source": "a", "src_in": 62, "src_out": 112},
    ]
    plan = make_plan(tmp_path, segments=segments)

    code, out, err = run(capsys, short_wav, "--kind", "short", "--plan", plan)
    assert (code, err) == (1, "")
    (line,) = [ln for ln in out.splitlines() if ln.startswith("FAIL  join-quiet:")]
    assert "1 of 1 cuts do not start in a pause of 60 ms below -40 dBFS" in line
    assert "no 60 ms pause within ±0.1 s of src_in, so re-cut it by ear" in line
    assert (
        "      join 1 at 00:00:01:00: fail (join 1 (record 00:00:01:00, src_in frame 62): no "
        "60 ms pause within ±0.1 s of src_in, so re-cut it by ear)"
    ) in out.splitlines()

    _, report = report_json(capsys, short_wav, "--kind", "short", "--plan", plan)
    join_quiet = checks_by_id(report)["join-quiet"]
    assert join_quiet["status"] == "fail"
    assert join_quiet["measured"] == {"cuts": 1, "failed": 1, "unchecked": 0}
    assert join_quiet["details"][0]["longest_quiet_ms"] == 0.0


@needs_ffmpeg
def test_plan_sources_that_cannot_be_read_warn_and_are_listed_unchecked(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, short_wav: Path
):
    segments = [
        {"track": 1, "source": "a", "src_in": 0, "src_out": 50},
        {"track": 1, "source": "a", "src_in": 62, "src_out": 112},
    ]
    plan = make_plan(tmp_path, segments=segments)  # src.wav is not next to the plan
    code, report = report_json(capsys, short_wav, "--kind", "short", "--plan", plan)
    assert code == 0  # not checked is a warning, not a failure
    assert report["status"] == "warn"
    join_quiet = checks_by_id(report)["join-quiet"]
    assert join_quiet["status"] == "warn"
    assert join_quiet["skip_reason"] is None
    assert join_quiet["incomplete"] is True
    assert join_quiet["measured"] == {
        "cuts": 1,
        "failed": 0,
        "unchecked": 1,
        "unchecked_reason": "source-unreadable",
    }
    assert "join-quiet" in report["unchecked"]
    code, out, _ = run(capsys, short_wav, "--kind", "short", "--plan", plan)
    assert code == 0
    assert out.splitlines()[0].endswith("WARN: 5 pass, 1 warn, 0 fail, 3 skipped")
    assert "join-quiet" in out.splitlines()[1]


@needs_ffmpeg
@pytest.mark.parametrize(
    ("caption_out", "code", "uncovered"),
    [(100, 0, 0), (50, 1, 1)],
    ids=["covered", "second-word-uncovered"],
)
def test_a_transcript_enables_caption_coverage(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    short_wav: Path,
    caption_out: int,
    code: int,
    uncovered: int,
):
    captions = [{"lines": ["hello world"], "record_in": 0, "record_out": caption_out}]
    plan = make_plan(tmp_path, captions=captions)
    # A bare list of words in recording seconds: "hello" mid 0.35 s, "world" mid 1.4 s.
    transcript = write_json(
        tmp_path / "words.json",
        [
            {"text": "hello", "start_s": 0.2, "end_s": 0.5},
            {"text": "world", "start_s": 1.2, "end_s": 1.6},
        ],
    )
    got, report = report_json(
        capsys, short_wav, "--kind", "short", "--plan", plan, "--transcript", transcript
    )
    assert got == code
    by_id = checks_by_id(report)
    coverage = by_id["caption-coverage"]
    assert coverage["status"] == ("pass" if code == 0 else "fail")
    assert coverage["measured"] == {
        "kept_words": 2,
        "captions": 1,
        "uncovered": uncovered,
        "segments": 1,
        "segments_unchecked": 0,
    }
    assert [d["word"] for d in coverage["details"]] == ["world"] * uncovered
    # The CLI passes no transcriber, so the re-transcription check says so instead of passing.
    assert by_id["join-retranscribe"]["status"] == "skipped"
    assert by_id["join-retranscribe"]["skip_reason"] == "no-transcriber"


def _missing(tmp_path: Path) -> Path:
    return tmp_path / "nope.json"


def _directory(tmp_path: Path) -> Path:
    return tmp_path


def _not_json(tmp_path: Path) -> Path:
    path = tmp_path / "plan.json"
    path.write_text("{not json", encoding="utf-8")
    return path


def _not_utf8(tmp_path: Path) -> Path:
    path = tmp_path / "plan.json"
    path.write_bytes(b"\xff\xfe{")
    return path


@pytest.mark.parametrize(
    ("make", "error_type"),
    [
        (_missing, "FileNotFoundError"),
        (_directory, "IsADirectoryError"),
        (_not_json, "invalid-input"),
        (_not_utf8, "UnicodeDecodeError"),
    ],
    ids=["missing", "directory", "not-json", "not-utf8"],
)
def test_an_unreadable_plan_is_exit_two(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, make: Any, error_type: str
):
    render_file = plain_file(tmp_path)
    plan = make(tmp_path)
    code, out, err = run(capsys, render_file, "--kind", "short", "--plan", plan)
    assert (code, out) == (2, "")
    assert err.startswith("error: ")

    assert error_json(capsys, render_file, "--kind", "short", "--plan", plan)["type"] == error_type


def test_an_invalid_plan_lists_each_problem(capsys: pytest.CaptureFixture[str], tmp_path: Path):
    render_file = plain_file(tmp_path)
    plan = write_json(tmp_path / "plan.json", {"version": 1})

    error = error_json(capsys, render_file, "--kind", "short", "--plan", plan)
    assert error["type"] == "invalid-input"
    assert error["message"] == "input failed validation"
    assert {"loc": ["project"], "msg": "Field required", "type": "missing"} in error["details"]

    code, out, err = run(capsys, render_file, "--kind", "short", "--plan", plan)
    assert (code, out) == (2, "")
    assert err.startswith("error: input failed validation\n")
    assert "  project: Field required\n" in err


@pytest.mark.parametrize(
    ("content", "error_type", "text"),
    [
        ("{not json", "JSONDecodeError", "Expecting property name"),
        (
            json.dumps([{"text": "a", "start_s": 3, "end_s": 1}]),
            "invalid-input",
            "input failed validation",
        ),
    ],
    ids=["not-json", "word-ends-before-it-starts"],
)
def test_a_bad_transcript_is_exit_two(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, content: str, error_type: str, text: str
):
    render_file = plain_file(tmp_path)
    transcript = tmp_path / "words.json"
    transcript.write_text(content, encoding="utf-8")
    code, out, err = run(capsys, render_file, "--kind", "short", "--transcript", transcript)
    assert (code, out) == (2, "")
    assert text in err

    error = error_json(capsys, render_file, "--kind", "short", "--transcript", transcript)
    assert error["type"] == error_type


# ---------------------------------------------------------------- --profile


def test_a_missing_profile_file_is_exit_two(capsys: pytest.CaptureFixture[str], tmp_path: Path):
    render_file = plain_file(tmp_path)
    missing = tmp_path / "nope.toml"
    code, out, err = run(capsys, render_file, "--kind", "short", "--profile", missing)
    assert (code, out) == (2, "")
    assert err == f"error: profile file not found: {missing}\n"

    error = error_json(capsys, render_file, "--kind", "short", "--profile", missing)
    assert error["type"] == "ProfileError"


def test_an_unknown_profile_name_is_exit_two(capsys: pytest.CaptureFixture[str], tmp_path: Path):
    code, out, err = run(capsys, plain_file(tmp_path), "--kind", "short", "--profile", "nope")
    assert (code, out) == (2, "")
    assert err == "error: no built-in rule profile named 'nope'\n"


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ('name = "x"\n[qa]\nlongform_lufs = -22\n', "short_lufs"),
        ('name = "x"\n[qa\n', "invalid TOML"),
    ],
    ids=["missing-required-key", "bad-toml"],
)
def test_an_invalid_profile_file_is_exit_two(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, text: str, message: str
):
    profile = tmp_path / "bad.toml"
    profile.write_text(text, encoding="utf-8")
    code, out, err = run(capsys, plain_file(tmp_path), "--kind", "short", "--profile", profile)
    assert (code, out) == (2, "")
    assert err.startswith(f"error: {profile}: ")
    assert message in err


@needs_ffmpeg
def test_a_minimal_profile_file_loads_and_sets_the_targets(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, short_wav: Path
):
    profile = write_profile(tmp_path / "x.toml")  # four [qa] keys, no rules
    code, report = report_json(capsys, short_wav, "--kind", "short", "--profile", profile)
    assert code == 0
    assert report["profile"] == "x"
    assert checks_by_id(report)["loudness"]["target"]["integrated_lufs"] == -14.5

    stricter = write_profile(tmp_path / "quiet.toml", name="quiet", short_lufs=-20)
    code, report = report_json(capsys, short_wav, "--kind", "short", "--profile", stricter)
    assert code == 1
    assert report["profile"] == "quiet"
    loudness = checks_by_id(report)["loudness"]
    assert loudness["status"] == "fail"
    assert loudness["target"]["integrated_lufs"] == -20.0


@needs_ffmpeg
def test_the_profile_defaults_to_the_plans_and_the_option_overrides_it(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, short_wav: Path
):
    sine(tmp_path / "src.wav", -10)
    profile = write_profile(tmp_path / "x.toml")
    plan = make_plan(tmp_path, profile=str(profile))
    _, report = report_json(capsys, short_wav, "--kind", "short", "--plan", plan)
    assert report["profile"] == "x"
    _, report = report_json(
        capsys, short_wav, "--kind", "short", "--plan", plan, "--profile", "ccn"
    )
    assert report["profile"] == "ccn"


# ---------------------------------------------------------------- ffmpeg override, file names


def test_a_bad_tool_override_is_exit_two(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    # check() resolves ffprobe first, then ffmpeg; the file is never read, so it need not be media.
    path = plain_file(tmp_path)
    monkeypatch.setenv("AAR_FFPROBE", str(tmp_path / "no-ffprobe"))
    monkeypatch.setenv("AAR_FFMPEG", str(tmp_path / "no-ffmpeg"))
    code, out, err = run(capsys, path, "--kind", "short")
    assert (code, out) == (2, "")
    assert err.startswith("error: AAR_FF")
    assert "is not an executable program" in err

    error = error_json(capsys, path, "--kind", "short")
    assert error["type"] == "MediaToolMissing"
    assert "AAR_FF" in error["message"]


@needs_ffmpeg
def test_a_bad_ffmpeg_override_alone_is_exit_two(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    short_wav: Path,
):
    monkeypatch.setenv("AAR_FFMPEG", str(tmp_path / "nope"))
    code, out, err = run(capsys, short_wav, "--kind", "short")
    assert (code, out) == (2, "")
    assert err.startswith("error: AAR_FFMPEG=")
    assert str(tmp_path / "nope") in err
    assert error_json(capsys, short_wav, "--kind", "short")["type"] == "MediaToolMissing"


@needs_ffmpeg
@pytest.mark.parametrize("name", ["-dash.wav", "space ünïcødé.wav", "colon:name.wav"])
def test_awkward_absolute_file_names_work(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, name: str
):
    path = sine(tmp_path / name, SHORT_DB)
    code, out, err = run(capsys, str(path), "--kind", "short")
    assert (code, err) == (0, "")
    assert out.splitlines()[0].startswith(f"qa {path} (short, profile ccn): PASS: ")

    _, report = report_json(capsys, str(path), "--kind", "short")
    assert report["file"] == str(path)
    assert report["media"]["size_bytes"] == path.stat().st_size
    assert checks_by_id(report)["decode"]["status"] == "pass"
    assert checks_by_id(report)["loudness"]["measured"]["integrated_lufs"] == pytest.approx(
        SHORT_DB, abs=0.05
    )


@needs_ffmpeg
def test_a_relative_dash_name_needs_the_double_dash_separator(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    sine(tmp_path / "-dash.wav", SHORT_DB)
    monkeypatch.chdir(tmp_path)
    # argparse reads a bare "-dash.wav" as an option: a usage error, exit 2.
    code, out, err = run(capsys, "--kind", "short", "-dash.wav")
    assert (code, out) == (2, "")
    assert "usage: aar qa check" in err
    # After "--" it is the file, and ffmpeg still gets it as a file: URL, not an option.
    code, out, err = run(capsys, "--kind", "short", "--", "-dash.wav")
    assert (code, err) == (0, "")
    assert out.splitlines()[0].startswith("qa -dash.wav (short, profile ccn): PASS: ")


# ---------------------------------------------------------------- regressions found in review


@pytest.mark.skipif(not hasattr(os, "geteuid") or os.geteuid() == 0, reason="root reads anything")
def test_an_unreadable_parent_directory_is_exit_two(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o000)
    try:
        code, out, err = run(capsys, locked / "x.wav", "--kind", "short")
        assert (code, out) == (2, "")
        assert err.startswith("error: cannot read ")
        error = error_json(capsys, locked / "x.wav", "--kind", "short")
        assert error["type"] == "PermissionError"
    finally:
        locked.chmod(0o700)


@needs_ffmpeg
@pytest.mark.skipif(sys.platform == "win32", reason="POSIX byte file names")
def test_json_report_for_a_file_name_that_is_not_utf8(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    name = os.fsdecode(b"bad\xff.wav")  # a lone surrogate once decoded
    path = sine(tmp_path / name, SHORT_DB, seconds=1.0)
    code, report = report_json(capsys, path, "--kind", "short")
    assert code == 0
    assert report["file"] == str(path)
    code, out, _ = run(capsys, path, "--kind", "short")
    assert code == 0 and "bad\\udcff.wav" in out.splitlines()[0]


def test_help_says_how_to_pass_a_dash_name(capsys: pytest.CaptureFixture[str]):
    assert main(["qa", "check", "--help"]) == 0
    text = " ".join(capsys.readouterr().out.split())
    assert (
        "A relative name that starts with '-' needs '--' before it or './' in front of it." in text
    )
