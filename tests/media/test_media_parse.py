"""The ffmpeg text parsers, on captured stderr (samples/, from capture.py) and on edge cases."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from aar.media import ParseError, Silence, parse_astats, parse_ebur128, parse_silencedetect
from aar.media.probe import parse_probe_json, parse_rational

SAMPLES = Path(__file__).parent / "samples"


def sample(name: str) -> str:
    return (SAMPLES / name).read_text(encoding="utf-8")


def test_chain_sample_gives_all_three_measurements():
    text = sample("chain_gap.txt")
    loud = parse_ebur128(text)
    assert loud.integrated_lufs == -21.7
    assert loud.integrated_threshold_lufs == -31.9
    assert loud.lra_lu == 0.0
    assert loud.sample_peak_dbfs == -20.0
    assert loud.true_peak_dbtp == -20.0
    stats = parse_astats(text)
    assert stats.peak_level_dbfs == pytest.approx(-19.999205)
    assert stats.get("Number of samples") == 120000
    assert stats.non_finite == 0
    assert parse_silencedetect(text) == [Silence(0.699979, 1.70004)]


def test_digital_silence_parses_minus_inf_and_the_gate_floor():
    text = sample("chain_digital_silence.txt")
    loud = parse_ebur128(text)
    assert loud.integrated_lufs == -70.0
    assert loud.sample_peak_dbfs == -math.inf
    assert loud.true_peak_dbtp == -math.inf
    assert parse_astats(text).peak_level_dbfs == -math.inf
    silences = parse_silencedetect(text)
    assert len(silences) == 1
    assert silences[0].start_s == 0.0
    assert silences[0].end_s == 1.0


def test_astats_reads_only_the_overall_section():
    stats = parse_astats(sample("astats_nan_perchannel.txt"))
    # Channel 1 printed 24000 NaNs; Overall prints the per-channel average.
    assert stats.get("Number of NaNs") == 12000
    assert stats.non_finite == 12000
    assert stats.peak_level_dbfs == -math.inf
    assert "Flat factor" not in stats.overall  # only in the per-channel sections


def test_ebur128_summary_is_found_after_per_frame_lines():
    loud = parse_ebur128(sample("ebur128_framelog.txt"))
    assert loud.integrated_lufs == pytest.approx(-21.1)
    assert loud.true_peak_dbtp == pytest.approx(-18.1)
    assert loud.sample_peak_dbfs is None  # peak=true only


def test_ebur128_takes_the_last_summary():
    first = sample("chain_digital_silence.txt")
    second = sample("chain_gap.txt")
    assert parse_ebur128(first + second).integrated_lufs == -21.7


@pytest.mark.parametrize(
    "text",
    ["", "no summary here\n", "[Parsed_ebur128_0 @ 0x1] Summary:\n\n  Integrated loudness:\n"],
)
def test_ebur128_without_a_complete_summary_raises(text):
    with pytest.raises(ParseError):
        parse_ebur128(text)


def test_astats_without_overall_raises():
    with pytest.raises(ParseError):
        parse_astats(
            "[Parsed_astats_0 @ 0x1] Channel: 1\n[Parsed_astats_0 @ 0x1] Peak level dB: -3\n"
        )


def test_silencedetect_open_silence_at_end_and_negative_start():
    # Handwritten: ffmpeg 6.1 closes a silence at EOF itself, older builds may not.
    text = (
        "[silencedetect @ 0xa] silence_start: -0.00002\n"
        "[silencedetect @ 0xa] silence_end: 0.5 | silence_duration: 0.50002\n"
        "[silencedetect @ 0xa] silence_start: 2.25\n"
        "[silencedetect @ 0xa] silence_start: 2.5\n"
    )
    found = parse_silencedetect(text)
    assert found == [Silence(0.0, 0.5), Silence(2.25, None)]
    assert found[1].duration(3.0) == pytest.approx(0.75)
    assert found[1].duration(None) is None


def test_silencedetect_ignores_other_lines():
    assert parse_silencedetect("silence_start: 1\n[other @ 0x1] silence_start: 2\n") == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("30000/1001", (30000, 1001)),
        ("25/1", (25, 1)),
        ("50", (50, 1)),
        ("0/0", None),
        ("", None),
        ("abc", None),
        (None, None),
        ("-25/1", None),
    ],
)
def test_parse_rational(text, expected):
    got = parse_rational(text)
    assert (None if got is None else (got.numerator, got.denominator)) == expected


def test_parse_probe_json_tolerates_missing_and_junk_fields():
    info = parse_probe_json(
        {
            "format": {"format_name": "mov", "duration": "N/A", "bit_rate": "x"},
            "streams": [
                {
                    "index": 0,
                    "codec_type": "video",
                    "avg_frame_rate": "0/0",
                    "r_frame_rate": "25/1",
                },
                {"index": 1, "codec_type": "video", "disposition": {"attached_pic": 1}},
                {"index": 2, "codec_type": "audio", "sample_rate": "48000", "channels": 2},
                "junk",
            ],
        },
        "x.mov",
        10,
    )
    assert info.duration_s is None and info.bit_rate is None
    assert info.video is not None and info.video.index == 0
    assert info.video.fps is not None and float(info.video.fps) == 25.0
    assert info.audio is not None and info.audio.sample_rate == 48000
    assert info.to_dict()["streams"][0]["fps"] == "25/1"
