"""Live ffmpeg runs: probe, one-pass audio measurement, full decode, PCM windows, clip count."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest
from media_support import aeval, gapped, needs_ffmpeg, render, sine, square, truncate

from aar.media import (
    MediaError,
    MediaInputError,
    count_clipped,
    decode_check,
    measure_audio,
    parse_astats,
    parse_ebur128,
    parse_silencedetect,
    probe,
    quiet_runs,
    read_pcm,
    run_tool,
)
from aar.media.tools import input_url

pytestmark = needs_ffmpeg

LEVEL = 10 ** (-0.01 / 20)


def measure(path: Path, silence_db: float = -42.0):
    return measure_audio(path, silence_db=silence_db, silence_min_s=0.2, timeout_s=60)


def test_probe_mp4_streams(tmp_path):
    path = render(tmp_path / "r.mp4", frames=60)
    info = probe(path)
    assert info.size_bytes == path.stat().st_size
    assert info.format and "mp4" in info.format
    video, audio = info.video, info.audio
    assert video is not None and audio is not None
    assert video.codec == "h264" and audio.codec == "aac"
    assert video.fps == Fraction(30000, 1001)
    assert (video.width, video.height) == (160, 90)
    assert video.duration_s == pytest.approx(60 * 1001 / 30000, abs=1e-5)
    assert (audio.sample_rate, audio.channels) == (48000, 2)
    assert info.duration_s == pytest.approx(2.002, abs=0.03)


@pytest.mark.parametrize("name", ["-starts-with-dash.wav", "space and ünïcødé ✓.wav"])
def test_hostile_names_probe_and_measure(tmp_path, name):
    path = sine(tmp_path / name, -20.0, seconds=1.0)
    info = probe(path)
    assert info.audio is not None and info.audio.duration_s == pytest.approx(1.0, abs=1e-3)
    assert measure(path).loudness.integrated_lufs == pytest.approx(-20.0, abs=0.1)


def test_relative_dash_name_is_not_an_option(tmp_path, monkeypatch):
    sine(tmp_path / "-y.wav", -20.0, seconds=0.5)
    monkeypatch.chdir(tmp_path)
    assert probe("-y.wav").audio is not None


def test_not_media_zero_length_and_missing(tmp_path):
    text = tmp_path / "notes.mp4"
    text.write_text("this is not media\n", encoding="utf-8")
    with pytest.raises(MediaError) as caught:
        probe(text)
    assert caught.value.exit_code == 1
    assert "Invalid data" in caught.value.stderr_tail
    empty = tmp_path / "empty.mp4"
    empty.write_bytes(b"")
    with pytest.raises(MediaError, match="empty file"):
        probe(empty)
    with pytest.raises(MediaInputError):
        probe(tmp_path / "missing.mp4")


def test_measure_sine_matches_its_level(tmp_path):
    m = measure(sine(tmp_path / "s.wav", -6.0))
    assert m.loudness.integrated_lufs == pytest.approx(-6.0, abs=0.1)
    assert m.loudness.true_peak_dbtp == pytest.approx(-6.0, abs=0.1)
    assert m.loudness.sample_peak_dbfs == pytest.approx(-6.0, abs=0.1)
    assert m.stats.peak_level_dbfs == pytest.approx(-6.0, abs=0.01)
    assert m.silences == []
    assert not m.log_truncated


def test_measure_finds_a_known_gap(tmp_path):
    path = gapped(tmp_path / "g.wav", before_s=0.5, gap_s=1.0, after_s=0.5, peak_dbfs=-20)
    m = measure(path)
    assert len(m.silences) == 1
    gap = m.silences[0]
    assert gap.start_s == pytest.approx(0.5, abs=0.002)
    assert gap.duration() == pytest.approx(1.0, abs=0.002)


def test_silence_threshold_is_the_callers(tmp_path):
    # A -39 dBFS tone is silence at a -36 dB venue threshold but not at -42.
    path = gapped(
        tmp_path / "v.wav", before_s=0.5, gap_s=1.0, after_s=0.5, peak_dbfs=-10, gap_dbfs=-39
    )
    assert measure(path, -42.0).silences == []
    venue = measure(path, -36.0).silences
    assert len(venue) == 1 and venue[0].duration() == pytest.approx(1.0, abs=0.002)


@pytest.mark.parametrize(("gap_db", "silent"), [(-42.5, True), (-41.5, False)])
def test_silence_level_either_side_of_the_threshold(tmp_path, gap_db, silent):
    path = gapped(
        tmp_path / "t.wav", before_s=0.5, gap_s=1.0, after_s=0.5, peak_dbfs=-10, gap_dbfs=gap_db
    )
    assert bool(measure(path, -42.0).silences) is silent


def test_live_stderr_feeds_the_pure_parsers(tmp_path):
    path = sine(tmp_path / "s.wav", -20.0, seconds=1.0)
    result = run_tool(
        "ffmpeg",
        [
            *("-hide_banner", "-nostdin", "-nostats", "-i", input_url(path), "-af"),
            "astats=measure_perchannel=none,silencedetect=n=-42dB:d=0.2,ebur128=peak=true",
            *("-f", "null", "-"),
        ],
        timeout_s=60,
    )
    assert parse_ebur128(result.stderr).integrated_lufs == pytest.approx(-20.0, abs=0.1)
    assert parse_astats(result.stderr).peak_level_dbfs == pytest.approx(-20.0, abs=0.01)
    assert parse_silencedetect(result.stderr) == []


def test_measure_on_a_file_without_audio_fails_with_tool_message(tmp_path):
    path = render(tmp_path / "v.mp4", frames=30, audio="none")
    with pytest.raises(MediaError) as caught:
        measure(path)
    assert caught.value.exit_code not in (None, 0)
    assert caught.value.stderr_tail


def test_decode_clean_and_truncated(tmp_path):
    ok = render(tmp_path / "ok.mp4", frames=60)
    clean = decode_check(ok, timeout_s=60)
    assert clean.ok and clean.exit_code == 0 and clean.errors == []
    cut = decode_check(truncate(ok, tmp_path / "cut.mp4"), timeout_s=60)
    assert not cut.ok
    assert cut.exit_code != 0
    assert cut.error_lines >= 1
    assert len(cut.errors) <= 5


def test_read_pcm_window_is_sample_accurate(tmp_path):
    path = gapped(tmp_path / "g.wav", before_s=0.5, gap_s=0.06, after_s=0.44)
    pcm = read_pcm(path, start_s=0.42, duration_s=0.2, channels=2)
    assert pcm.channels == 2 and pcm.rate == 48000
    assert pcm.frames == 9600
    assert pcm.start_s == 0.42
    best = quiet_runs(pcm, threshold_dbfs=-40).longest
    assert best is not None
    assert best.start_s == pytest.approx(0.5, abs=0.0002)
    assert best.duration_s * 1000 == pytest.approx(60, abs=0.25)


def test_read_pcm_clamps_a_negative_start_and_stops_at_the_end(tmp_path):
    path = sine(tmp_path / "s.wav", -6.0, seconds=0.5)
    head = read_pcm(path, start_s=-0.1, duration_s=0.2, channels=2)
    assert head.start_s == 0.0 and head.frames == 9600
    tail = read_pcm(path, start_s=0.45, duration_s=0.2, channels=2)
    assert tail.frames == pytest.approx(2400, abs=2)
    past = read_pcm(path, start_s=0.6, duration_s=0.2, channels=2)
    assert past.frames == 0


def test_count_clipped_square_and_sine(tmp_path):
    clipped = count_clipped(
        square(tmp_path / "sq.wav", channels=1),
        level=LEVEL,
        channels=1,
        sample_rate=48000,
        timeout_s=60,
    )
    assert clipped.clipped == 48000 == clipped.scanned
    assert clipped.first_s == 0.0
    clean = count_clipped(
        sine(tmp_path / "s.wav", -6.0, channels=2),
        level=LEVEL,
        channels=2,
        sample_rate=48000,
        timeout_s=60,
    )
    assert clean.clipped == 0 and clean.first_s is None
    assert clean.scanned == 2 * 2 * 48000


def test_count_clipped_places_the_first_over(tmp_path):
    # One full-scale sample at n = 12000 (0.25 s) on an otherwise quiet tone.
    path = aeval(
        tmp_path / "spike.wav",
        f"if(eq(n\\,12000)\\,1\\,{10 ** (-20 / 20)}*sin(2*PI*1000*t))",
        seconds=0.5,
        channels=2,
    )
    count = count_clipped(path, level=LEVEL, channels=2, sample_rate=48000, timeout_s=60)
    assert count.clipped == 2  # both channels
    assert count.first_s == pytest.approx(0.25)


def test_nan_samples_are_counted_by_astats(tmp_path):
    path = aeval(tmp_path / "nan.wav", "if(lt(t\\,0.25)\\,0/0\\,0.1)", seconds=0.5)
    m = measure(path)
    assert m.stats.non_finite == 12000  # 12000 per channel: 0.25 s at 48 kHz
