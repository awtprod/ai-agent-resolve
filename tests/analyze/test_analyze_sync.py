"""``analyze.sync`` end to end on synthetic recordings of one event (WAV files through ffmpeg).

Every clip is at most 60 s. Tolerances:

- offsets on clean input: 0.5 ms (the brief), and 30 us where the input is clean and the
  offset is a fraction of a sample at the 8 kHz analysis rate (125 us), so the parabolic
  refinement is what gets it there;
- drift: +-10 ppm. Inside one 6 s consistency window 100 ppm moves the alignment by 0.6 ms
  (4.8 samples), so each window's peak lands anywhere in that smear: about 0.17 ms rms. A line
  through ~22 windows over ~45 s then has a slope error of about 2 ppm (1 sigma); 10 ppm is 5
  sigma and still tells 100 ppm from 0 or from 90 and 110.
"""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from analyze_support import FFMPEG, Source, lowpass, needs_ffmpeg, tone, write_wav

from aar.analyze import SyncSettings, sync_files
from aar.cli import main

pytestmark = needs_ffmpeg

NTSC = Fraction(24000, 1001)


@pytest.fixture(scope="module")
def event() -> Source:
    return Source(seconds=75.0, seed=3)


def sync_arrays(
    tmp_path: Path,
    ref: np.ndarray,
    other: np.ndarray,
    *,
    fps: Fraction = NTSC,
    ref_origin: int = 0,
    **settings: Any,
) -> dict[str, Any]:
    a = write_wav(tmp_path / "ref.wav", ref)
    b = write_wav(tmp_path / "other.wav", other)
    report = sync_files(a, [b], fps=fps, ref_origin=ref_origin, settings=SyncSettings(**settings))
    json.dumps(report, allow_nan=False)  # strict JSON, always
    [pair] = report["pairs"]
    assert report["ok"] == (pair["status"] == "ok")
    return pair


def assert_no_number(pair: dict[str, Any]) -> None:
    """A pair that is not confident never carries an offset or an origin."""
    for key in ("offset_s", "offset_frames", "suggested_recording_origin"):
        assert key not in pair


# ---------------------------------------------------------------- offsets


@pytest.mark.parametrize("offset", [2.0, -3.5, 1.2345, 1.23456, -0.00731])
def test_offsets_positive_negative_and_sub_sample(tmp_path, event, offset):
    ref = event.take(5.0, 50.0)
    other = event.take(5.0 + offset, 45.0)
    pair = sync_arrays(tmp_path, ref, other)
    assert pair["status"] == "ok"
    assert pair["offset_s"] == pytest.approx(offset, abs=0.0005)
    assert pair["offset_s"] == pytest.approx(offset, abs=30e-6)
    frames = pair["offset_frames"]
    assert frames == round(offset * 24000 / 1001)
    assert pair["suggested_recording_origin"] == frames
    expected_error_ms = (frames * 1001 / 24000 - pair["offset_s"]) * 1000
    assert pair["rounding_error_ms"] == pytest.approx(expected_error_ms, abs=1e-3)
    assert abs(pair["rounding_error_ms"]) <= 1000 / 2 * 1001 / 24000 + 1e-6
    confidence = pair["confidence"]
    assert confidence["agreement"] == 1.0
    assert confidence["median_ratio"] >= 10
    assert pair["drift"]["measured"] and abs(pair["drift"]["ppm"]) < 2


def test_sign_convention_b_frame_zero_at_a_two_seconds(tmp_path, event):
    # B's frame 0 is at A's 2.0 s, so B.recording_origin = A.recording_origin + 2.0 * fps.
    pair = sync_arrays(
        tmp_path, event.take(1.0, 40.0), event.take(3.0, 30.0), fps=Fraction(25), ref_origin=1000
    )
    assert pair["status"] == "ok"
    assert pair["offset_s"] == pytest.approx(2.0, abs=30e-6)
    assert pair["offset_frames"] == 50
    assert pair["suggested_recording_origin"] == 1050
    assert pair["rounding_error_ms"] == pytest.approx(0.0, abs=0.03)
    # Swapping the files flips the sign.
    swapped = sync_arrays(
        tmp_path, event.take(3.0, 30.0), event.take(1.0, 40.0), fps=Fraction(25), ref_origin=1050
    )
    assert swapped["offset_s"] == pytest.approx(-2.0, abs=30e-6)
    assert swapped["suggested_recording_origin"] == 1000


@pytest.mark.parametrize("variant", ["quiet", "noisy", "lowpassed", "inverted"])
def test_a_different_microphone_still_syncs(tmp_path, event, variant):
    other = event.take(4.25, 45.0)
    rng = np.random.default_rng(7)
    if variant == "quiet":
        other = other * 0.05  # -26 dB
    elif variant == "noisy":
        rms = float(np.sqrt(np.mean(other**2)))
        other = other + rng.standard_normal(len(other)) * rms  # 0 dB signal-to-noise
    elif variant == "lowpassed":
        other = lowpass(other, 1000.0)
    else:
        other = -other
    pair = sync_arrays(tmp_path, event.take(3.0, 50.0), other)
    assert pair["status"] == "ok", pair.get("reason")
    assert pair["offset_s"] == pytest.approx(1.25, abs=0.0005)
    # Anchors outside the other file find chance peaks of either sign; the matches agree.
    matched = [w for w in pair["coarse"]["windows"] if w.get("ratio", 0) >= 2]
    assert len(matched) >= 8
    assert {w["polarity"] for w in matched} == ({-1} if variant == "inverted" else {1})


def test_a_loud_shared_hum_does_not_hide_the_match(tmp_path, event):
    # A steady 1 kHz whine 20 dB above the room in both files: its correlation repeats every
    # millisecond. PHAT weighting gives it a few bins out of thousands, so the room still wins.
    hum = tone(80.0, 1000.0, peak=0.5)
    ref = event.take(3.0, 50.0) * 0.05 + hum[: 50 * 16000]
    other = event.take(4.5, 45.0) * 0.05 + hum[int(1.5 * 16000) : int(1.5 * 16000) + 45 * 16000]
    pair = sync_arrays(tmp_path, ref, other)
    assert pair["status"] == "ok", pair.get("reason")
    assert pair["offset_s"] == pytest.approx(1.5, abs=0.0005)


# ---------------------------------------------------------------- overlap and range


def test_partial_overlap(tmp_path, event):
    pair = sync_arrays(tmp_path, event.take(0.5, 40.0), event.take(25.5, 35.0))
    assert pair["status"] == "ok"
    assert pair["offset_s"] == pytest.approx(25.0, abs=30e-6)
    assert pair["overlap_s"] == pytest.approx(15.0, abs=0.01)
    assert pair["overlap_ref_s"] == [pytest.approx(25.0, abs=0.01), pytest.approx(40.0, abs=0.01)]
    # Anchors that fall outside the other file are not eligible, so they do not count against it.
    confidence = pair["confidence"]
    assert confidence["windows_eligible"] < 12 and confidence["agreement"] == 1.0


def test_no_overlap_when_a_file_is_shorter_than_five_seconds(tmp_path, event):
    pair = sync_arrays(tmp_path, event.take(1.0, 40.0), event.take(10.0, 3.0))
    assert pair["status"] == "no-overlap"
    assert pair["overlap_s"] == pytest.approx(3.0, abs=0.01)
    assert "less than the 5 s" in pair["hint"]
    assert_no_number(pair)


def test_disjoint_audio_is_low_confidence_with_candidates(tmp_path, event):
    # No shared audio cannot be told from audio that matches badly: never a number.
    pair = sync_arrays(tmp_path, event.take(0.5, 30.0), event.take(40.0, 30.0))
    assert pair["status"] == "low-confidence"
    assert_no_number(pair)
    assert pair["candidates"] and all(c["median_ratio"] < 2 for c in pair["candidates"])
    assert "best candidate" in pair["reason"] and pair["hint"]


def test_an_offset_beyond_max_offset_is_not_found(tmp_path, event):
    ref, other = event.take(1.0, 55.0), event.take(21.0, 40.0)  # true offset +20 s
    pair = sync_arrays(tmp_path, ref, other, max_offset_s=10.0)
    assert pair["status"] == "low-confidence"
    assert_no_number(pair)
    assert all(abs(c["offset_s"]) <= 10.0 + 1e-6 for c in pair["candidates"])
    wider = sync_arrays(tmp_path, ref, other, max_offset_s=30.0)
    assert wider["status"] == "ok" and wider["offset_s"] == pytest.approx(20.0, abs=30e-6)


# ---------------------------------------------------------------- the important negatives


def test_a_pure_tone_is_low_confidence(tmp_path):
    signal = tone(60.0, 440.0)
    pair = sync_arrays(tmp_path, signal[: 50 * 16000], signal[2 * 16000 :])
    assert pair["status"] == "low-confidence"
    assert_no_number(pair)
    assert all(c["median_ratio"] < 1.5 for c in pair["candidates"])


def test_silence_is_low_confidence(tmp_path, event):
    # Digital silence through GCC-PHAT itself is covered in test_analyze_sync_unit.
    pair = sync_arrays(tmp_path, event.take(1.0, 40.0), np.zeros(30 * 16000))
    assert pair["status"] == "low-confidence"
    assert_no_number(pair)
    assert "other file is silent" in pair["reason"]
    assert pair["level_dbfs"]["other"] is None


def test_a_silent_reference_is_low_confidence(tmp_path, event):
    pair = sync_arrays(tmp_path, np.zeros(30 * 16000), event.take(1.0, 40.0))
    assert pair["status"] == "low-confidence" and "reference is silent" in pair["reason"]
    assert_no_number(pair)


# ---------------------------------------------------------------- drift


def test_100_ppm_drift_is_measured(tmp_path, event):
    pair = sync_arrays(tmp_path, event.take(2.0, 55.0), event.take(3.0, 50.0, drift_ppm=100.0))
    assert pair["status"] == "ok"  # 100 ppm over 50 s is 5 ms, well under a frame
    drift = pair["drift"]
    assert drift["measured"]
    assert drift["ppm"] == pytest.approx(100.0, abs=10.0)
    assert drift["over_overlap_ms"] == pytest.approx(5.0, abs=0.6)
    assert drift["residual_rms_ms"] < 0.5
    assert drift["used"] >= 20 and drift["outliers"] == 0
    # The offset is the line's value mid-overlap: about 1 s + 100 ppm * 25 s.
    assert pair["offset_s"] == pytest.approx(1.0025, abs=0.0005)


def test_drift_beyond_a_frame_is_reported_not_corrected(tmp_path, event):
    pair = sync_arrays(
        tmp_path, event.take(2.0, 55.0), event.take(3.0, 50.0, drift_ppm=400.0), fps=Fraction(60)
    )
    assert pair["status"] == "drift"
    drift = pair["drift"]
    assert drift["ppm"] == pytest.approx(400.0, abs=10.0)
    assert drift["over_overlap_ms"] > 1000 / 60
    assert "ppm" in pair["hint"] and "does not correct" in pair["hint"]
    # Confident, so the best single offset is given, with the hint.
    assert pair["offset_s"] == pytest.approx(1.01, abs=0.001)
    assert pair["suggested_recording_origin"] == pair["offset_frames"]
    # The same pair at 24p moves 20 ms, under one 41.7 ms frame: ok.
    assert (
        sync_arrays(tmp_path, event.take(2.0, 55.0), event.take(3.0, 50.0, drift_ppm=400.0))[
            "status"
        ]
        == "ok"
    )


# ---------------------------------------------------------------- failures and the CLI


def _video_only(path: Path) -> Path:
    import subprocess

    assert FFMPEG is not None
    subprocess.run(
        [FFMPEG, "-hide_banner", "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i",
         "testsrc=size=64x36:rate=25:duration=1", "-c:v", "libx264", "-preset", "ultrafast",
         str(path)],
        check=True, timeout=60,
    )  # fmt: skip
    return path


def run_cli(capsys: pytest.CaptureFixture[str], *argv: object) -> tuple[int, dict[str, Any]]:
    code = main(["analyze", "sync", *(str(a) for a in argv)])
    out = capsys.readouterr().out
    return code, json.loads(out)


def test_cli_exit_codes_and_out_file(tmp_path, event, capsys):
    ref = write_wav(tmp_path / "ref.wav", event.take(5.0, 40.0))
    good = write_wav(tmp_path / "good.wav", event.take(7.0, 30.0))
    flat = write_wav(tmp_path / "flat.wav", tone(30.0))
    out = tmp_path / "sync.json"
    code, report = run_cli(
        capsys, "--ref", ref, "--other", good, "--fps", "24000/1001", "--ref-origin", "100",
        "--out", out,
    )  # fmt: skip
    assert code == 0 and report["ok"]
    assert json.loads(out.read_text(encoding="utf-8")) == report
    [pair] = report["pairs"]
    assert pair["suggested_recording_origin"] == 100 + 48
    assert report["sign"].startswith("offset_s is where the other file's first sample falls")
    code, report = run_cli(capsys, "--ref", ref, "--other", good, "--other", flat, "--fps", "25")
    assert code == 1 and not report["ok"]
    assert [p["status"] for p in report["pairs"]] == ["ok", "low-confidence"]


def test_cli_media_that_does_not_decode_is_a_failed_pair(tmp_path, event, capsys):
    ref = write_wav(tmp_path / "ref.wav", event.take(5.0, 40.0))
    junk = tmp_path / "junk.wav"
    junk.write_bytes(b"RIFF this is not audio" * 50)
    silent_video = _video_only(tmp_path / "video.mp4")
    code, report = run_cli(
        capsys, "--ref", ref, "--other", junk, "--other", silent_video, "--fps", "25"
    )
    assert code == 1
    junk_pair, video_pair = report["pairs"]
    assert junk_pair["status"] == "failed" and junk_pair["error"]["type"] == "decode-failed"
    assert video_pair["status"] == "failed" and video_pair["error"]["type"] == "no-audio"
    assert_no_number(junk_pair)
    # A reference that does not decode fails every pair, still exit 1.
    code, report = run_cli(capsys, "--ref", junk, "--other", ref, "--fps", "25")
    assert code == 1 and report["pairs"][0]["error"]["type"] == "decode-failed"
    assert report["reference"]["error"]["type"] == "decode-failed"


@pytest.mark.parametrize(
    ("argv", "kind"),
    [
        (["--other", "{good}", "--ref", "{missing}", "--fps", "25"], "input-missing"),
        (["--ref", "{good}", "--other", "{missing}", "--fps", "25"], "input-missing"),
        (["--ref", "{good}", "--other", "{good}", "--fps", "0"], "invalid-argument"),
        (["--ref", "{good}", "--other", "{good}", "--fps", "x/2"], "invalid-argument"),
        (["--ref", "{good}", "--other", "{good}", "--fps", "25", "--max-offset-s", "nan"],
         "invalid-argument"),
        (["--ref", "{good}", "--other", "{good}", "--fps", "25", "--min-agreement", "2"],
         "invalid-argument"),
        (["--ref", "{good}", "--other", "{good}", "--fps", "25", "--out", "{tmp}/no/such.json"],
         "invalid-argument"),
    ],
)  # fmt: skip
def test_cli_input_errors_exit_2(tmp_path, capsys, argv, kind):
    good = write_wav(tmp_path / "good.wav", tone(6.0))
    names = {"good": good, "missing": tmp_path / "missing.wav", "tmp": tmp_path}
    code, body = run_cli(capsys, *(a.format(**names) for a in argv))
    assert code == 2
    assert body["ok"] is False and body["error"]["type"] == kind


def test_cli_missing_ffmpeg_exits_2(tmp_path, capsys, monkeypatch):
    good = write_wav(tmp_path / "good.wav", tone(6.0))
    monkeypatch.setenv("AAR_FFPROBE", str(tmp_path / "nope"))
    code, body = run_cli(capsys, "--ref", good, "--other", good, "--fps", "25")
    assert code == 2 and body["error"]["type"] == "tool-missing"
