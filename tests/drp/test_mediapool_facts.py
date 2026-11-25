"""S25c1: the measured facts are untrusted, and every derived value has one rule (pinned here).

Each formula test fails if the rule changes (floor versus round, which rate, which time). The
values are synthetic; the rules' evidence is the corpus gate (``test_mediapool_gate``).
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from fractions import Fraction

import mediapool_support as ms
import pytest

from aar.drp.mediapool import formulas as fm
from aar.drp.mediapool.errors import MediaPoolError
from aar.drp.mediapool.facts import MAX_FACTS_BYTES, facts_from_json

CREATED_US = int(datetime(2026, 3, 20, 11, 22, 33, tzinfo=UTC).timestamp()) * 1_000_000


def refused(data: object, code: str = "bad_facts") -> MediaPoolError:
    with pytest.raises(MediaPoolError) as caught:
        facts_from_json(data)
    assert caught.value.code == code
    return caught.value


# --------------------------------------------------------------------------- hostile facts


@pytest.mark.parametrize(
    "change",
    [
        {"stream_width": "3840"},  # a number sent as text
        {"stream_sample_rate": 48000},  # text sent as a number
        {"stream_sample_rate": "48k"},
        {"stream_channels": 0},
        {"stream_channels": 65},
        {"stream_duration_ts": -1},
        {"stream_duration_ts": 2**62},
        {"stream_time_base": "one/48000"},
        {"stream_codec_name": "pcm\x00s24le"},
        {"stream_tags": {"title": "x" * 5000}},  # an enormous tag value
        {"stream_tags": {f"k{i}": "v" for i in range(300)}},
        {"stream_side_data_list": [{"rotation": [90]}]},
        {"size": -1},
        {"size": "2880044"},
        {"size": 2**53},
        {"path": "relative\\take.wav"},
        {"path": "X:/Show/take.wav"},
        {"path": "X:\\Show\\..\\take.wav"},
        {"path": "X:\\Show\\"},
        {"path": "X:\\Show\\ta\x07ke.wav"},
        {"mtime_local": "Mon Mar 3 04:05:06 2026"},  # the 3rd was a Tuesday
        {"mtime_local": "2026-03-03 04:05:06"},
        {"mtime_local": 1794191444},
        {"mtime_utc": "2026-11-09 03:30:44"},
        {"ctime_utc": 5},
        {"exists": False},
        {"exists": "yes"},
        {"surprise": 1},
        {"ffprobe": None},
        {"ffprobe": {"format": {}}},  # missing streams
        {"ffprobe": {"streams": [], "format": None}},
        {"ffprobe": {"streams": "none", "format": {}}},
    ],
)
def test_malformed_facts_are_refused(change: dict[str, object]) -> None:
    refused(ms.wav_facts(**change))


def test_too_many_streams_and_duplicate_indexes_are_refused() -> None:
    data = ms.wav_facts()
    stream = data["ffprobe"]["streams"][0]
    data["ffprobe"]["streams"] = [dict(stream, index=i) for i in range(65)]
    refused(data)
    data["ffprobe"]["streams"] = [stream, dict(stream)]
    refused(data)


def test_facts_text_is_bounded_and_must_be_an_object() -> None:
    refused("x" * (MAX_FACTS_BYTES + 1))
    refused("[1, 2]")
    refused("{not json")
    refused(json.dumps(ms.wav_facts())[:-5])


def test_good_facts_read() -> None:
    facts = facts_from_json(json.dumps(ms.wav_facts()))
    assert (facts.folder, facts.name, facts.extension) == (ms.PC_FOLDER, ms.WAV_NAME, "wav")
    assert facts.mtime_local == datetime(2026, 3, 3, 4, 5, 6)
    assert facts.mtime_utc_ns % 1_000_000_000 == 123_456_700  # 7 digits kept exactly
    assert facts.ctime_utc_ns is not None
    unc = facts_from_json(ms.wav_facts(path="\\\\server\\share\\Show\\take.wav"))
    assert unc.folder == "\\\\server\\share\\Show"


def test_unknown_ffprobe_fields_are_ignored_not_trusted() -> None:
    data = ms.wav_facts()
    data["ffprobe"]["streams"][0]["new_in_some_ffprobe"] = {"deep": [1, 2]}
    assert facts_from_json(data).streams[0].index == 0


# --------------------------------------------------------------------------- formulas


def facts(**change: object):
    return facts_from_json(ms.mp4_facts(**change))


def test_modification_time_text_pads_the_day_with_a_space() -> None:
    assert fm.qt_local_time(datetime(2026, 3, 3, 4, 5, 6)) == "Tue Mar  3 04:05:06 2026"
    assert fm.qt_local_time(datetime(2026, 3, 13, 9, 5, 3)) == "Fri Mar 13 09:05:03 2026"


def test_rate_prefers_the_timecode_track_over_a_variable_video_rate() -> None:
    data = ms.mp4_facts(video_r_frame_rate="30/1", video_avg_frame_rate="1234567/41152")
    data["ffprobe"]["streams"][2].update(codec_tag_string="tmcd", avg_frame_rate="30000/1001")
    assert fm.video_rate(facts_from_json(data)) == Fraction(30000, 1001)


def test_constant_rate_without_timecode_track_is_the_video_rate() -> None:
    assert fm.video_rate(facts()) == Fraction(24000, 1001)


def test_variable_rate_without_timecode_track_is_refused() -> None:
    f = facts(video_r_frame_rate="30/1", video_avg_frame_rate="1234567/41152")
    with pytest.raises(MediaPoolError) as caught:
        fm.video_rate(f)
    assert caught.value.code == "vfr_without_timecode"


@pytest.mark.parametrize("rate", ["12/1", "1000000/1", "2997/100"])
def test_a_rate_resolve_was_not_seen_to_store_is_refused(rate: str) -> None:
    with pytest.raises(MediaPoolError) as caught:
        fm.video_rate(facts(video_r_frame_rate=rate, video_avg_frame_rate=rate))
    assert caught.value.code == "not_exact"


@pytest.mark.parametrize("rate", ["0/0", "30000/0", "-30/1"])
def test_zero_and_negative_rates_are_refused(rate: str) -> None:
    with pytest.raises(MediaPoolError) as caught:
        fm.video_rate(facts(video_r_frame_rate=rate, video_avg_frame_rate=rate))
    assert caught.value.code == "cannot_derive"


def test_frame_count_is_duration_times_rate_not_nb_frames() -> None:
    f = facts(video_nb_frames="9999")
    assert fm.video_frames(f, Fraction(24000, 1001)) == 3000


def test_a_frame_count_that_is_not_whole_is_refused() -> None:
    f = facts(video_duration_ts=3003001)
    with pytest.raises(MediaPoolError) as caught:
        fm.video_frames(f, Fraction(24000, 1001))
    assert caught.value.code == "not_exact"
    with pytest.raises(MediaPoolError):
        fm.video_frames(facts(video_time_base="1/0"), Fraction(24000, 1001))


def test_start_seconds_count_at_the_nominal_rate_and_divide_by_the_double_rate() -> None:
    rate = Fraction(24000, 1001)
    # 02:10:20:12 is frame 7820 * 24 + 12 = 187692 (nominal 24 for 23.976).
    assert fm.start_seconds("02:10:20:12", rate) == 187692 / float(rate)
    assert fm.start_seconds(None, rate) == 0.0
    # Dividing by the double is not the exactly rounded quotient for every frame number; this
    # one shows the difference, so the test fails if the rule changes.
    frames = next(n for n in range(1, 10_000) if n / float(rate) != float(Fraction(n) / rate))
    seconds, ff = divmod(frames, 24)
    label = f"{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}:{ff:02d}"
    assert fm.start_seconds(label, rate) == frames / float(rate) != float(Fraction(frames) / rate)


def test_drop_frame_start_uses_the_drop_frame_count() -> None:
    rate = Fraction(30000, 1001)
    assert fm.start_seconds("01:00:00;00", rate) == 107892 / float(rate)
    with pytest.raises(MediaPoolError):
        fm.start_seconds("25:00:00:00", rate)


def test_start_timecode_comes_from_any_stream_that_has_one() -> None:
    assert fm.start_timecode(facts()) == "02:10:20:12"
    data = ms.mp4_facts()
    data["ffprobe"]["streams"][2]["tags"] = {}
    assert fm.start_timecode(facts_from_json(data)) is None


def test_audio_samples_round_down() -> None:
    # 1/14112000 s per tick at 44.1 kHz is 1/320 sample per tick: 320 * 1000 + 200 ticks is
    # 1000.625 samples, so rounding to the nearest would give 1001.
    data = ms.wav_facts(
        stream_time_base="1/14112000", stream_sample_rate="44100", stream_duration_ts=320_200
    )
    stream = facts_from_json(data).streams[0]
    assert fm.audio_samples(stream) == 1000
    assert math.floor(320_200 * 44100 / 14112000) == 1000


def test_audio_without_duration_or_with_absurd_rate_is_refused() -> None:
    for change in (
        {"stream_duration_ts": None},
        {"stream_sample_rate": "5"},
        {"stream_duration_ts": 0},
    ):
        with pytest.raises(MediaPoolError):
            fm.audio_samples(facts_from_json(ms.wav_facts(**change)).streams[0])


@pytest.mark.parametrize(
    ("codec", "name", "depth"),
    [
        ("pcm_s16le", "Linear PCM", 1),
        ("pcm_s16be", "Linear PCM", 1),
        ("pcm_s24le", "Linear PCM", 2),
        ("aac", "AAC", 3),
        ("mp3", "MP3", 3),
    ],
)
def test_audio_codec_name_and_bit_depth(codec: str, name: str, depth: int) -> None:
    stream = facts_from_json(ms.wav_facts(stream_codec_name=codec)).streams[0]
    assert (fm.audio_codec_name(stream), fm.audio_bit_depth(stream)) == (name, depth)


def test_unknown_audio_codec_is_refused() -> None:
    stream = facts_from_json(ms.wav_facts(stream_codec_name="pcm_f32le")).streams[0]
    with pytest.raises(MediaPoolError):
        fm.audio_codec_name(stream)


def test_channel_layout_is_named_or_absent() -> None:
    read = lambda **c: fm.channel_layout(facts_from_json(ms.wav_facts(**c)).streams[0])  # noqa: E731
    assert read() is None
    assert read(stream_channel_layout="stereo") == 2
    assert read(stream_channel_layout="mono", stream_channels=1) == 1
    with pytest.raises(MediaPoolError):
        read(stream_channel_layout="5.1")
    with pytest.raises(MediaPoolError):
        read(stream_channel_layout="mono")  # two channels


def test_creation_time_is_the_tag_else_the_files_creation_time_never_its_modification() -> None:
    assert fm.creation_us(facts()) == CREATED_US
    wav = facts_from_json(ms.wav_facts())
    assert fm.creation_us(wav) == wav.ctime_utc_ns // 1000 != wav.mtime_utc_ns // 1000
    with pytest.raises(MediaPoolError):
        fm.creation_us(facts_from_json(ms.wav_facts(ctime_utc=None)))


@pytest.mark.parametrize(
    ("w", "h", "pix", "size"),
    [
        (3840, 2160, "yuv420p", 3840 * 2160 * 3 // 2 + 8192),
        (1920, 1013, "yuv420p", 1920 * 1014 * 3 // 2 + 8192),  # an odd height counts one more line
        (1920, 1080, "yuv420p10le", 1920 * 1080 * 3 + 8192),
        (1920, 1080, "yuv422p10le", 1920 * 1080 * 8 // 3 + 8192),
        (1920, 1080, "yuva444p12le", 1920 * 1080 * 8 + 8192),
    ],
)
def test_frame_size(w: int, h: int, pix: str, size: int) -> None:
    bpp = fm.PICTURE_CLASSES[pix][1]
    assert fm.frame_size(w, h, bpp) == size


def test_unknown_pixel_format_is_refused() -> None:
    f = facts(video_pix_fmt="yuv444p16le")
    with pytest.raises(MediaPoolError):
        fm.picture_class(f.streams[0])


def test_rotation_transposes_the_resolution() -> None:
    plain = facts().streams[0]
    assert fm.resolution(plain) == (3840, 2160, False)
    rotated = facts(
        video_side_data_list=[{"side_data_type": "Display Matrix", "rotation": -90}]
    ).streams[0]
    assert fm.resolution(rotated) == (2160, 3840, True)
    upside = facts(video_side_data_list=[{"rotation": 180}]).streams[0]
    assert fm.resolution(upside) == (3840, 2160, False)
