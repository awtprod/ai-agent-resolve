"""Source.start_timecode in the plan producer (S18 round 4): the stub's own value, else the media's
``timecode`` tag (video stream, then any stream such as a tmcd track, then the container) parsed at
the video stream's rate with drop-frame handled. Resolve links an FCPXML clip to its file only when
the asset's start equals the file's start timecode (import test rerun, 2026-11-08)."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from fractions import Fraction
from pathlib import Path

import pytest
from export_support import GOLDEN_PLANS, ccn, export, load_plan
from produce_support import ffmpeg, needs_ffmpeg

from aar.media import MediaError
from aar.media.probe import parse_probe_json, timecode_frames
from aar.media.quicktime import QuickTimeTimecode, read_timecode
from aar.plan.export import ExportError
from aar.plan.export.timebase import parse_time
from aar.plan.model import EditPlan
from aar.plan.produce import build as build_module
from aar.plan.produce.audio import SourceAudio, open_audio
from aar.plan.produce.build import Probed, StartTimecode, probe_sources, source_start_timecodes
from aar.plan.produce.errors import ProduceInputError
from qt_support import TMCD_AT, Movie, co64, retime_tmcd, stsd, tmcd_entry

FILM = Fraction(24000, 1001)
NTSC = Fraction(30000, 1001)
NTSC60 = Fraction(60000, 1001)


def _stub(*sources: dict[str, object], fps: str = "24/1") -> EditPlan:
    return EditPlan.model_validate(
        {
            "project": "P",
            "timeline": "T",
            "kind": "longform",
            "fps": fps,
            "resolution": {"width": 1920, "height": 1080},
            "profile": "ccn",
            "sources": list(sources),
            "segments": [],
        }
    )


def _source(**extra: object) -> dict[str, object]:
    return {"id": "cam", "media": "cam.mov", **extra}


# -- parsing a timecode ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "fps", "frames"),
    [
        ("00:00:00:00", Fraction(24), 0),
        ("01:00:00:00", Fraction(24), 86400),
        ("01:00:00:00", FILM, 86400),  # non-drop counts at the nominal 24
        ("01:00:00:00", Fraction(25), 90000),
        ("01:00:00:00", NTSC, 108000),  # 29.97 non-drop: nominal 30
        ("00:00:01:05", Fraction(24), 29),
        ("23:59:59:23", Fraction(24), 24 * 86400 - 1),
        # drop-frame (';' or '.'): frame numbers 0-1 skipped each minute but every tenth
        ("00:00:59;29", NTSC, 1799),
        ("00:01:00;02", NTSC, 1800),
        ("00:10:00;00", NTSC, 17982),
        ("01:00:00;00", NTSC, 107892),
        ("01:00:00.00", NTSC, 107892),
        ("00:01:00;04", NTSC60, 3600),  # 59.94 drops 0-3
        ("01:00:00;00", NTSC60, 215784),
    ],
)
def test_timecode_frames(text: str, fps: Fraction, frames: int) -> None:
    assert timecode_frames(text, fps) == frames


@pytest.mark.parametrize(
    ("text", "fps", "why"),
    [
        ("01:00:00", Fraction(24), "not a HH:MM:SS:FF timecode"),
        ("1h", Fraction(24), "not a HH:MM:SS:FF timecode"),
        ("00:00:00:24", Fraction(24), "frame 24"),
        ("00:60:00:00", Fraction(24), "minute 60"),
        ("00:00:60:00", Fraction(24), "second 60"),
        ("24:00:00:00", Fraction(24), "hour 24"),
        ("01:00:00;00", Fraction(24), "drop-frame"),
        ("01:00:00;00", Fraction(25), "drop-frame"),
        ("00:01:00;00", NTSC, "dropped"),
        ("00:01:00;01", NTSC, "dropped"),
        ("00:01:00;03", NTSC60, "dropped"),
    ],
)
def test_a_bad_timecode_is_refused(text: str, fps: Fraction, why: str) -> None:
    with pytest.raises(ValueError, match=why):
        timecode_frames(text, fps)


# -- reading the tag ---------------------------------------------------------------------------


def _info(video_tc: str | None, data_tc: str | None, format_tc: str | None) -> dict[str, object]:
    video: dict[str, object] = {"index": 0, "codec_type": "video", "avg_frame_rate": "24/1"}
    data: dict[str, object] = {"index": 2, "codec_type": "data", "codec_tag_string": "tmcd"}
    if video_tc:
        video["tags"] = {"timecode": video_tc}
    if data_tc:
        data["tags"] = {"timecode": data_tc}
    streams = [video, {"index": 1, "codec_type": "audio"}, data]
    fmt = {"tags": {"timecode": format_tc}} if format_tc else {}
    return {"streams": streams, "format": fmt}


@pytest.mark.parametrize(
    ("video_tc", "data_tc", "format_tc", "want"),
    [
        ("01:00:00:00", "02:00:00:00", "03:00:00:00", "01:00:00:00"),  # the video stream's
        (None, "02:00:00:00", "03:00:00:00", "02:00:00:00"),  # then a tmcd (any) stream's
        (None, None, "03:00:00:00", "03:00:00:00"),  # then the container's
        (None, None, None, None),
    ],
)
def test_the_probe_reads_the_timecode_tag(
    video_tc: str | None, data_tc: str | None, format_tc: str | None, want: str | None
) -> None:
    info = parse_probe_json(_info(video_tc, data_tc, format_tc), "x.mov", 1)
    assert info.timecode == want
    assert info.to_dict()["timecode"] == want


# -- precedence --------------------------------------------------------------------------------

HOUR = Probed(None, "01:00:00:00", Fraction(24))


TC24 = Fraction(24)


@pytest.mark.parametrize(
    ("stub_tc", "probed", "want", "warns"),
    [
        (None, HOUR, StartTimecode(86400, TC24), []),  # the probe's tag, at the media's rate
        (86400, HOUR, StartTimecode(86400), []),  # agree: no warning (the stub gave no rate)
        (0, HOUR, StartTimecode(0), ["start-timecode-mismatch"]),  # the stub wins, and says so
        (90000, None, StartTimecode(90000), []),  # the stub alone
        (None, Probed(None, None, TC24), None, []),  # no tag: unset
        (None, None, None, []),  # nothing probed (a pool clip, say): unset
        (None, Probed(None, "junk", TC24), None, ["start-timecode-unread"]),
        (3, Probed(None, "junk", TC24), StartTimecode(3), ["start-timecode-unread"]),
    ],
)
def test_start_timecode_precedence(
    stub_tc: int | None, probed: Probed | None, want: StartTimecode | None, warns: list[str]
) -> None:
    stub = _stub(_source(**({"start_timecode": stub_tc} if stub_tc is not None else {})))
    timecodes, warnings = source_start_timecodes(stub, {"cam": probed} if probed else {})
    assert timecodes == ({"cam": want} if want is not None else {})
    assert [w.code for w in warnings] == warns
    assert all(w.source == "cam" for w in warnings)


def test_a_drop_frame_tag_is_counted_at_the_stream_rate() -> None:
    stub = _stub(_source(), fps="30000/1001")
    timecodes, warnings = source_start_timecodes(stub, {"cam": Probed(None, "01:00:00;00", NTSC)})
    assert timecodes == {"cam": StartTimecode(107892, NTSC)} and warnings == []


def test_an_audio_only_tag_is_counted_at_the_plan_rate() -> None:
    stub = _stub(_source(media="cam.wav"), fps="30000/1001")
    timecodes, warnings = source_start_timecodes(stub, {"cam": Probed(None, "01:00:00:00", None)})
    assert timecodes == {"cam": StartTimecode(108000)} and warnings == []  # the plan's rate


def test_another_stream_rate_comes_with_its_rate() -> None:
    """Fix-check Q3: counted at the media's own rate (25 fps here), and that rate is set, so the
    export converts with it."""
    stub = _stub(_source())
    timecodes, warnings = source_start_timecodes(
        stub, {"cam": Probed(None, "01:00:00:00", Fraction(25))}
    )
    assert timecodes == {"cam": StartTimecode(90000, Fraction(25))} and warnings == []


def test_a_stub_timecode_without_its_rate_on_other_rate_media_is_refused() -> None:
    stub = _stub(_source(start_timecode=90000))
    with pytest.raises(ProduceInputError) as caught:
        source_start_timecodes(stub, {"cam": Probed(None, None, Fraction(25))})
    assert caught.value.code == "start-timecode-rate-required"
    assert caught.value.exit_code == 2
    # With its rate it is fine, and so is media at the plan's rate.
    given = _stub(_source(start_timecode=90000, start_timecode_rate={"num": 25}))
    assert source_start_timecodes(given, {"cam": Probed(None, None, Fraction(25))}) == (
        {"cam": StartTimecode(90000, Fraction(25))},
        [],
    )
    assert source_start_timecodes(stub, {"cam": Probed(None, None, TC24)})[0] == {
        "cam": StartTimecode(90000)
    }


# -- a real file -------------------------------------------------------------------------------


@pytest.fixture
def timecoded(tmp_path: Path) -> Path:
    """A 160x90 24 fps clip with sound and an embedded start timecode of 01:00:00:00."""
    out = tmp_path / "tc.mov"
    ffmpeg(
        "-f", "lavfi", "-i", "testsrc=size=160x90:rate=24", "-f", "lavfi", "-i", "sine=d=1",
        "-t", "1", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-timecode", "01:00:00:00", str(out),
    )  # fmt: skip
    return out


@pytest.mark.parametrize("key", ["timecode", "TIMECODE", "TimeCode"])
def test_the_tag_name_is_read_in_any_case(key: str) -> None:
    """Fix-check Q2: ffprobe reports Matroska's tag as TIMECODE, on the container."""
    data = {
        "streams": [{"index": 0, "codec_type": "video"}],
        "format": {"tags": {key: "01:00:00:00"}},
    }
    assert parse_probe_json(data, "x.mkv", 1).timecode == "01:00:00:00"


@needs_ffmpeg
def test_a_matroska_file_gives_its_start_timecode(tmp_path: Path) -> None:
    out = tmp_path / "tc.mkv"
    ffmpeg(
        "-f", "lavfi", "-i", "testsrc=size=160x90:rate=24", "-f", "lavfi", "-i", "sine=d=1",
        "-t", "1", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-timecode", "01:00:00:00", str(out),
    )  # fmt: skip
    stub = _stub(_source(media=str(out)))
    probed, warnings = probe_sources(stub, {}, None)
    assert warnings == [] and probed["cam"].timecode == "01:00:00:00"
    assert source_start_timecodes(stub, probed)[0] == {"cam": StartTimecode(86400, TC24)}


@needs_ffmpeg
def test_a_timecoded_file_gives_its_start_timecode(timecoded: Path) -> None:
    stub = _stub(_source(media=str(timecoded)))
    audio = open_audio(stub, ["cam"], None)
    assert (audio["cam"].timecode, audio["cam"].fps) == ("01:00:00:00", Fraction(24))
    # A source of the cut: open_audio's probe; a source the cut does not use: probed on its own.
    for given in (audio, {}):
        probed, warnings = probe_sources(stub, given, None)
        assert warnings == []
        assert source_start_timecodes(stub, probed) == ({"cam": StartTimecode(86400, TC24)}, [])


@needs_ffmpeg
def test_a_25_fps_file_at_01_00_00_00_exports_at_3600s_in_a_24_fps_plan(tmp_path: Path) -> None:
    """Fix-check Q3 end to end: the probe reads 90000 at 25 fps and the producer sets the rate,
    so the export of a 24 fps plan starts the asset exactly at 3600 s."""
    out = tmp_path / "pal.mov"
    ffmpeg(
        "-f", "lavfi", "-i", "testsrc=size=160x90:rate=25", "-f", "lavfi", "-i", "sine=d=1",
        "-t", "1", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-timecode", "01:00:00:00", str(out),
    )  # fmt: skip
    stub = _stub(_source(media=str(out)))  # a 24 fps plan
    probed, _ = probe_sources(stub, {}, None)
    timecodes, warnings = source_start_timecodes(stub, probed)
    assert timecodes == {"cam": StartTimecode(90000, Fraction(25))} and warnings == []
    data = load_plan(GOLDEN_PLANS["tc_24"]).model_dump(mode="json")
    for source in data["sources"]:
        source["start_timecode"] = timecodes["cam"].frames
        source["start_timecode_rate"] = {"num": 25, "den": 1}
    result = export(EditPlan.model_validate(data), ccn())
    starts = {a.get("start") for a in ET.fromstring(result.fcpxml).iter("asset")}
    assert starts == {"3600s"}


# -- S18a: the tmcd reader -----------------------------------------------------------------------

#: BRAW's video stream averages 2953125/98536 fps (one 1002-tick sample in 1575 at 1/30000): what
#: ffprobe gives as its rate, on no frame grid. Its tmcd track says 30000/1001.
BRAW_AVG = Fraction(2953125, 98536)


def _tmcd(
    frames: int = 521420, rate: Fraction = NTSC, drop: bool = False, per_second: int | None = None
) -> QuickTimeTimecode:
    per_second = round(rate) if per_second is None else per_second
    return QuickTimeTimecode(
        frames, rate, drop, True, 2 | drop, rate.numerator, rate.denominator, per_second, 4096
    )


BRAW_TAG = "04:49:40:20"


def test_the_tmcd_track_is_the_fallback_when_the_probe_has_no_timecode() -> None:
    stub = _stub(_source(media="A001.braw"), fps="30000/1001")
    # No video (ffprobe could not open it): the tmcd track's rate, snapped; with video: the
    # video's rate, snapped.
    for seen, rate_origin in (
        (Probed(None, tmcd=_tmcd()), None),
        (Probed((4112, 2176), None, BRAW_AVG, _tmcd()), "media probe"),
    ):
        timecodes, warnings = source_start_timecodes(stub, {"cam": seen})
        assert timecodes == {"cam": StartTimecode(521420, NTSC)} and warnings == []
        chosen = timecodes["cam"]
        assert (chosen.origin, chosen.rate_origin, chosen.drop_frame) == (
            "tmcd reader",
            rate_origin,
            False,
        )


def test_the_braw_rate_is_the_video_rate_snapped() -> None:
    """The CCN BRAW on ffprobe 6.1: the tag reads 04:49:40:20 and the video stream averages
    2953125/98536 fps (dev used that as the rate: on no 29.97 frame). Snapped, it is 30000/1001,
    with or without the tmcd track."""
    stub = _stub(_source(media="A001.braw"), fps="30000/1001")
    for seen in (
        Probed((4112, 2176), BRAW_TAG, BRAW_AVG, _tmcd()),
        Probed((4112, 2176), BRAW_TAG, BRAW_AVG),
        Probed((4112, 2176), BRAW_TAG, BRAW_AVG, None, None, NTSC),  # with r_frame_rate
    ):
        timecodes, warnings = source_start_timecodes(stub, {"cam": seen})
        assert timecodes == {"cam": StartTimecode(521420, NTSC)} and warnings == []
        chosen = timecodes["cam"]
        assert (chosen.origin, chosen.rate_origin) == ("media probe", None)
        assert chosen.to_dict(NTSC)["rate_from"] == "media probe"
        assert (Fraction(chosen.frames) / NTSC * NTSC).denominator == 1


TMCD_SPELLINGS = [
    pytest.param(NTSC, Fraction(2997, 100), 108000, id="29.97+tmcd-2997/100"),
    pytest.param(NTSC, Fraction(30), 108000, id="29.97+tmcd-30/1"),
    pytest.param(NTSC, NTSC, 108000, id="29.97+tmcd-30000/1001"),
    pytest.param(FILM, Fraction(24), 86400, id="23.976+tmcd-24/1"),
    pytest.param(Fraction(25), Fraction(25), 90000, id="25+tmcd-25/1"),
]


@pytest.mark.parametrize(("stream", "tmcd_rate", "frames"), TMCD_SPELLINGS)
def test_a_tmcd_rate_spelled_differently_takes_the_streams_rate(
    stream: Fraction, tmcd_rate: Fraction, frames: int
) -> None:
    """The validator's regression (round 1 took the tmcd rate verbatim, off the plan's grid):
    the tmcd track only says how many frames a second it counts; the rate is the stream's,
    snapped, so 01:00:00:00 lands exactly on a frame of a plan at the stream's rate."""
    stub = _stub(_source(), fps=f"{stream.numerator}/{stream.denominator}")
    for tag in ("01:00:00:00", None):
        seen = Probed((64, 48), tag, stream, _tmcd(frames, tmcd_rate))
        timecodes, warnings = source_start_timecodes(stub, {"cam": seen})
        assert timecodes == {"cam": StartTimecode(frames, stream)} and warnings == []
        on_plan = Fraction(frames) / timecodes["cam"].rate * stream  # type: ignore[operator]
        assert on_plan == frames  # exactly on the plan's frame grid


def test_a_plain_hour_with_its_tmcd_track_warns_nothing() -> None:
    stub = _stub(_source(), fps="30000/1001")
    seen = Probed((64, 48), "01:00:00:00", NTSC, _tmcd(108000))
    assert source_start_timecodes(stub, {"cam": seen}) == ({"cam": StartTimecode(108000, NTSC)}, [])


@pytest.mark.parametrize(
    ("seen", "why"),
    [
        pytest.param(Probed(None, "00:00:01:10", Fraction(50), _tmcd(35, Fraction(25))),
                     r"counts 25 frames a second \(at 25/1\) but its video runs at 50/1",
                     id="25-tmcd-on-50-stream"),
        pytest.param(Probed(None, "01:00:00:00", Fraction(50), _tmcd(90000, Fraction(25))),
                     "counts 25 frames a second", id="25-tmcd-on-50-plain-hour"),
        pytest.param(Probed(None, None, Fraction(50), _tmcd(35, Fraction(25))),
                     "counts 25 frames", id="25-tmcd-on-50-no-tag"),
        pytest.param(Probed(None, None, Fraction(120), _tmcd(30, Fraction(30))),
                     r"counts 30 frames a second \(at 30/1\) but its video runs at 120/1",
                     id="120-fps-stream-30-tmcd"),
        pytest.param(Probed(None, tmcd=_tmcd(30, Fraction(25, 2), per_second=13)),
                     r"counts 13 frames a second \(at 25/2\) but its tmcd track runs at 25/2",
                     id="no-video-odd-tmcd"),
        pytest.param(Probed(None, "01:00:00;00", Fraction(30), _tmcd(107892, NTSC, drop=True)),
                     "drop-frame but its video runs at 30/1", id="drop-frame-tmcd-on-30"),
        pytest.param(Probed(None, "01:00:00:00", Fraction(50), None,
                            "the tmcd track counts 50 frames a second at 25 fps", None,
                            "unsupported"),
                     "tmcd track cannot be used", id="unsupported-tmcd"),
        pytest.param(Probed(None, None, None, None, "counts 50 at 25", None, "unsupported"),
                     "tmcd track cannot be used", id="unsupported-tmcd-no-tag"),
    ],
)  # fmt: skip
def test_an_ambiguous_rate_is_refused_unless_the_stub_gives_the_timecode(
    seen: Probed, why: str
) -> None:
    stub = _stub(_source(), fps="50/1")
    with pytest.raises(ProduceInputError, match=why) as caught:
        source_start_timecodes(stub, {"cam": seen})
    assert caught.value.code == "timecode-rate-ambiguous" and caught.value.exit_code == 2
    assert "start_timecode and start_timecode_rate" in caught.value.hint
    given = _stub(_source(start_timecode=70, start_timecode_rate={"num": 50}), fps="50/1")
    timecodes, warnings = source_start_timecodes(given, {"cam": seen})
    assert timecodes == {"cam": StartTimecode(70, Fraction(50))}
    assert "start-timecode-reader-mismatch" not in [w.code for w in warnings]


def test_a_negative_tmcd_start_blocks_the_tag_too() -> None:
    """Prism on PR #15: a tmcd sample of -1 (flags 0x6) is refused by the reader; ffprobe reads
    the same sample unsigned and wraps it into an unrelated positive tag, so the tag must not be
    used either. The stub's value settles it."""
    seen = Probed(None, "09:51:31:20", Fraction(25), None, "the tmcd sample is -1", None,
                  "negative_timecode")  # fmt: skip
    stub = _stub(_source(), fps="25/1")
    with pytest.raises(ProduceInputError, match="tmcd track cannot be used") as caught:
        source_start_timecodes(stub, {"cam": seen})
    assert caught.value.code == "timecode-rate-ambiguous"
    given = _stub(_source(start_timecode=0, start_timecode_rate={"num": 25}), fps="25/1")
    assert source_start_timecodes(given, {"cam": seen})[0] == {
        "cam": StartTimecode(0, Fraction(25))
    }


def test_the_r_frame_rate_snaps_when_the_average_does_not() -> None:
    stub = _stub(_source(), fps="30000/1001")
    seen = Probed(None, "01:00:00:00", Fraction(59, 2), None, None, NTSC)  # VFR: average 29.5
    assert source_start_timecodes(stub, {"cam": seen})[0] == {"cam": StartTimecode(108000, NTSC)}


def test_the_tag_counts_at_the_snapped_rate() -> None:
    """Round 1 counted a tag at the raw stream rate (or the tmcd rate without video); now it
    counts at the snapped rate: the video's, else the tmcd track's, else the plan's."""
    stub = _stub(_source(), fps="24/1")
    no_video = Probed(None, "01:00:00:00", None, _tmcd(108000, Fraction(2997, 100)))
    timecodes, _ = source_start_timecodes(stub, {"cam": no_video})
    assert timecodes == {"cam": StartTimecode(108000, NTSC)}
    assert (timecodes["cam"].origin, timecodes["cam"].rate_origin) == ("media probe", "tmcd reader")
    assert timecodes["cam"].to_dict(Fraction(24))["rate_from"] == "tmcd reader"
    odd_average = Probed(None, "00:01:00:00", Fraction(2401, 100))  # 24.01 snaps to 24
    assert source_start_timecodes(stub, {"cam": odd_average})[0] == {
        "cam": StartTimecode(1440, Fraction(24))
    }


def test_when_the_tag_and_the_tmcd_track_disagree_the_probe_wins_with_a_warning() -> None:
    stub = _stub(_source(), fps="30000/1001")
    seen = Probed(None, "01:00:00:00", NTSC, _tmcd(5))
    timecodes, warnings = source_start_timecodes(stub, {"cam": seen})
    assert timecodes == {"cam": StartTimecode(108000, NTSC)}
    assert timecodes["cam"].origin == "media probe"
    assert [w.code for w in warnings] == ["start-timecode-reader-mismatch"]
    assert "01:00:00:00 (frame 108000)" in warnings[0].message
    assert "00:00:00:05 (frame 5" in warnings[0].message


def test_a_tag_that_cannot_be_read_leaves_the_tmcd_track() -> None:
    stub = _stub(_source(), fps="30000/1001")
    timecodes, warnings = source_start_timecodes(
        stub, {"cam": Probed(None, "junk", NTSC, _tmcd(900))}
    )
    assert timecodes == {"cam": StartTimecode(900, NTSC)}
    assert timecodes["cam"].origin == "tmcd reader"
    assert [w.code for w in warnings] == ["start-timecode-unread"]


def test_a_drop_frame_tmcd_track_counts_real_frames() -> None:
    stub = _stub(_source(), fps="30000/1001")
    seen = Probed(None, "01:00:00;00", NTSC, _tmcd(107892, drop=True))
    timecodes, warnings = source_start_timecodes(stub, {"cam": seen})
    assert timecodes == {"cam": StartTimecode(107892, NTSC)} and warnings == []
    assert timecodes["cam"].drop_frame is True
    assert timecodes["cam"].to_dict(NTSC)["timecode"] == "01:00:00;00"


def test_the_stub_still_wins_over_the_tmcd_track() -> None:
    stub = _stub(_source(start_timecode=3, start_timecode_rate="30000/1001"), fps="30000/1001")
    timecodes, warnings = source_start_timecodes(stub, {"cam": Probed(None, tmcd=_tmcd())})
    assert timecodes == {"cam": StartTimecode(3, NTSC)}
    assert timecodes["cam"].origin == "stub"
    assert [w.code for w in warnings] == ["start-timecode-mismatch"]
    assert "tmcd track 04:49:40:20 (frame 521420)" in warnings[0].message
    agreeing = _stub(_source(start_timecode=521420), fps="30000/1001")
    assert source_start_timecodes(agreeing, {"cam": Probed(None, tmcd=_tmcd())})[1] == []


def test_the_rate_check_for_a_stub_value_uses_the_tmcd_rate() -> None:
    """A stub frame count without its rate is fine for BRAW in a 29.97 plan (the tmcd track runs
    at the plan's rate) even though the video stream averages another rate; refused when the
    tmcd track runs at another rate."""
    stub = _stub(_source(start_timecode=521420), fps="30000/1001")
    seen = Probed((4112, 2176), BRAW_TAG, BRAW_AVG, _tmcd())
    assert source_start_timecodes(stub, {"cam": seen}) == ({"cam": StartTimecode(521420)}, [])
    with pytest.raises(ProduceInputError) as caught:
        source_start_timecodes(stub, {"cam": Probed(None, tmcd=_tmcd(90000, Fraction(25)))})
    assert caught.value.code == "start-timecode-rate-required"


def test_a_tmcd_track_that_cannot_be_read_warns_only_when_nothing_else_gives_one() -> None:
    stub = _stub(_source(media="A001.braw"), fps="30000/1001")
    broken = Probed(None, tmcd_problem="the tmcd sample at 99 is past the end")
    timecodes, warnings = source_start_timecodes(stub, {"cam": broken})
    assert timecodes == {} and [w.code for w in warnings] == ["start-timecode-unread"]
    assert "past the end" in warnings[0].message and "aar media timecode" in warnings[0].hint
    tagged = Probed(None, "01:00:00:00", NTSC, None, "malformed")
    assert source_start_timecodes(stub, {"cam": tagged}) == (
        {"cam": StartTimecode(108000, NTSC)},
        [],
    )
    given = _stub(_source(media="A001.braw", start_timecode=9), fps="30000/1001")
    assert source_start_timecodes(given, {"cam": broken})[1] == []


def test_read_tmcd_reads_only_quicktime_files(tmp_path: Path) -> None:
    good = Movie().build()
    for name in ("A001.braw", "B.MOV", "c.mp4", "D.Braw"):
        (tmp_path / name).write_bytes(good)
        tmcd, problem, code = build_module.read_tmcd(tmp_path / name)
        assert tmcd is not None and tmcd.frames == 521420 and (problem, code) == (None, None)
    for name in ("cam.wav", "cam.mkv", "cam.mxf"):
        (tmp_path / name).write_bytes(good)
        assert build_module.read_tmcd(tmp_path / name) == (None, None, None)
    (tmp_path / "plain.mov").write_bytes(Movie(with_tmcd=False).build())
    no_track = build_module.read_tmcd(tmp_path / "plain.mov")
    assert no_track == (None, None, None)  # a file without a tmcd track: nothing to say
    (tmp_path / "cut.braw").write_bytes(good[:5000])
    tmcd, problem, code = build_module.read_tmcd(tmp_path / "cut.braw")
    assert tmcd is None and problem is not None and "past the end of the file" in problem
    assert code == "malformed"
    odd = Movie()
    odd.tmcd_stbl = [stsd(tmcd_entry(per_second=60)), co64(TMCD_AT)]
    (tmp_path / "odd.mov").write_bytes(odd.build())
    assert build_module.read_tmcd(tmp_path / "odd.mov")[2] == "unsupported"
    negative = Movie(frame=2**32 - 1)  # -1 as a signed 32-bit sample
    negative.tmcd_stbl = [stsd(tmcd_entry(0x6, 25, 1, 25)), co64(TMCD_AT)]
    (tmp_path / "negative.mov").write_bytes(negative.build())
    assert build_module.read_tmcd(tmp_path / "negative.mov")[2] == "negative_timecode"
    tmcd, problem, code = build_module.read_tmcd(tmp_path / "gone.braw")
    assert tmcd is None and problem is not None and "file not found" in problem
    assert code == "io"


def test_probe_sources_reads_braw_that_ffprobe_cannot_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ffprobe failing on a BRAW (an older ffprobe, or one without the mov demuxer): the
    picture size is still a warning, and the start timecode comes from the tmcd track."""
    braw = tmp_path / "A001.braw"
    braw.write_bytes(Movie().build())
    junk = tmp_path / "junk.mov"
    junk.write_bytes(b"not a movie")

    def no_ffprobe(path: object, **_: object) -> object:
        raise MediaError("ffprobe exited 1", tool="ffprobe")

    monkeypatch.setattr(build_module, "probe", no_ffprobe)
    stub = _stub(_source(media=str(braw)), {"id": "junk", "media": str(junk)}, fps="30000/1001")
    probed, warnings = probe_sources(stub, {}, None)
    assert [(w.code, w.source) for w in warnings] == [
        ("frame-size-unread", "cam"),
        ("frame-size-unread", "junk"),
    ]
    assert probed == {"cam": Probed(None, tmcd=read_timecode(braw))}  # junk: nothing kept
    timecodes, warnings = source_start_timecodes(stub, probed)
    assert timecodes == {"cam": StartTimecode(521420, NTSC)} and warnings == []


def test_probe_sources_reads_the_tmcd_track_of_the_cuts_sources(tmp_path: Path) -> None:
    braw = tmp_path / "A001.braw"
    braw.write_bytes(Movie().build())
    stub = _stub(_source(media=str(braw)), fps="30000/1001")
    audio = {"cam": SourceAudio(braw, 2, 52.5, (4112, 2176), BRAW_TAG, BRAW_AVG)}
    probed, warnings = probe_sources(stub, audio, None)
    assert warnings == []
    assert probed == {"cam": Probed((4112, 2176), BRAW_TAG, BRAW_AVG, read_timecode(braw))}
    timecodes, _ = source_start_timecodes(stub, probed)
    assert timecodes == {"cam": StartTimecode(521420, NTSC)}


def test_the_braw_timecode_exports_on_the_29_97_grid() -> None:
    """End to end: the tmcd track's 521420 at 30000/1001 puts the asset at exactly
    521420 x 1001/30000 s in a 29.97 plan; the video stream's average rate would put it between
    two frames, which the export refuses."""
    data = load_plan(GOLDEN_PLANS["tc_2997"]).model_dump(mode="json")
    for source in data["sources"]:
        source["start_timecode"] = 521420
        source["start_timecode_rate"] = {"num": 30000, "den": 1001}
    result = export(EditPlan.model_validate(data), ccn())
    starts = {parse_time(a.get("start", "")) for a in ET.fromstring(result.fcpxml).iter("asset")}
    assert starts == {Fraction(521420 * 1001, 30000)}
    for source in data["sources"]:
        source["start_timecode_rate"] = {"num": BRAW_AVG.numerator, "den": BRAW_AVG.denominator}
    with pytest.raises(ExportError) as caught:
        export(EditPlan.model_validate(data), ccn())
    assert caught.value.code == "start_timecode_off_grid"


def test_the_start_timecode_report_says_what_supplied_it() -> None:
    plan_fps = Fraction(24)
    rows = {
        "stub": StartTimecode(86400),
        "stub-rate": StartTimecode(90000, Fraction(25)),
        "probe": StartTimecode(86400, plan_fps, "media probe"),
        "both": StartTimecode(521420, NTSC, "media probe", "tmcd reader", False),
        "tmcd-df": StartTimecode(107892, NTSC, "tmcd reader", None, True),
    }
    got = {key: value.to_dict(plan_fps) for key, value in rows.items()}
    assert got["stub"] == {
        "frames": 86400,
        "rate": None,
        "timecode": "01:00:00:00",
        "drop_frame": None,
        "from": "stub",
        "rate_from": "plan fps",
    }
    assert (got["stub-rate"]["rate"], got["stub-rate"]["rate_from"]) == ("25/1", "stub")
    assert (got["probe"]["from"], got["probe"]["rate_from"]) == ("media probe", "media probe")
    assert got["both"]["rate"] == "30000/1001" and got["both"]["timecode"] == "04:49:40:20"
    assert (got["both"]["from"], got["both"]["rate_from"]) == ("media probe", "tmcd reader")
    assert got["tmcd-df"]["timecode"] == "01:00:00;00" and got["tmcd-df"]["drop_frame"] is True
    # A label that cannot be written (drop-frame at the plan's 24) is left out, not raised.
    assert StartTimecode(5, None, "stub", None, True).to_dict(plan_fps)["timecode"] is None


@needs_ffmpeg
@pytest.mark.parametrize(
    ("rate", "timescale", "duration", "frames"),
    [
        pytest.param("30000/1001", 2997, 100, 108000, id="29.97+tmcd-2997/100"),
        pytest.param("30000/1001", 30, 1, 108000, id="29.97+tmcd-30/1"),
        pytest.param("24000/1001", 24, 1, 86400, id="23.976+tmcd-24/1"),
    ],
)
def test_the_validators_retimed_tmcd_files_land_on_the_plan_grid(
    tmp_path: Path, rate: str, timescale: int, duration: int, frames: int
) -> None:
    out = tmp_path / "retimed.mov"
    ffmpeg(
        "-f", "lavfi", "-i", f"testsrc=size=64x48:rate={rate}:duration=1", "-f", "lavfi",
        "-i", "sine=d=1", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-timecode", "01:00:00:00", str(out),
    )  # fmt: skip
    retime_tmcd(out, timescale, duration)
    assert read_timecode(out).rate == Fraction(timescale, duration)
    fps = Fraction(rate)
    stub = _stub(_source(media=str(out)), fps=rate)
    for audio in (open_audio(stub, ["cam"], None), {}):  # a source of the cut, and not
        probed, warnings = probe_sources(stub, audio, None)
        assert warnings == []
        timecodes, warnings = source_start_timecodes(stub, probed)
        assert timecodes == {"cam": StartTimecode(frames, fps)} and warnings == []
        assert Fraction(frames) / fps * fps == frames  # the asset starts on a plan frame
    if rate == "30000/1001":  # and the export takes it
        data = load_plan(GOLDEN_PLANS["tc_2997"]).model_dump(mode="json")
        for source in data["sources"]:
            source["start_timecode"] = frames
            source["start_timecode_rate"] = {"num": 30000, "den": 1001}
        result = export(EditPlan.model_validate(data), ccn())
        starts = {
            parse_time(a.get("start", "")) for a in ET.fromstring(result.fcpxml).iter("asset")
        }
        assert starts == {Fraction(frames * 1001, 30000)}


# -- S18a round 3: exact rates without a tmcd track, unused sources, the rate order ------------


@pytest.mark.parametrize("fps", [15, 12, 72, 90, 100, 120, 240])
def test_a_non_standard_rate_without_a_tmcd_track_counts_at_the_exact_rate(fps: int) -> None:
    """Validator R2-1: round 2 refused these (no standard rate) and aborted the produce; with no
    tmcd track nothing disagrees, so the tag counts at the stream's exact rate, as dev did."""
    rate = Fraction(fps)
    stub = _stub(_source(), fps="24/1")
    seen = Probed(None, "01:00:00:00", rate, None, None, rate)
    timecodes, warnings = source_start_timecodes(stub, {"cam": seen})
    assert timecodes == {"cam": StartTimecode(3600 * fps, rate)} and warnings == []


@pytest.mark.parametrize(
    ("average", "base", "rate"),
    [
        pytest.param(Fraction(2997, 25), Fraction(120), Fraction(120), id="vfr-119.88-r-120"),
        pytest.param(Fraction(120), Fraction(90000), Fraction(120), id="r-out-of-range"),
        pytest.param(Fraction(120), Fraction(1199, 10), Fraction(120), id="r-not-clean"),
        pytest.param(Fraction(120), None, Fraction(120), id="no-r"),
        pytest.param(None, Fraction(120000, 1001), Fraction(120000, 1001), id="r-ntsc-only"),
    ],
)
def test_the_exact_rate_is_a_clean_r_frame_rate_else_the_average(
    average: Fraction | None, base: Fraction | None, rate: Fraction
) -> None:
    stub = _stub(_source(), fps="24/1")
    seen = Probed(None, "00:00:01:00", average, None, None, base)
    timecodes, _ = source_start_timecodes(stub, {"cam": seen})
    assert timecodes == {"cam": StartTimecode(round(rate), rate)}


def test_a_tmcd_track_that_agrees_with_an_exact_rate_is_fine() -> None:
    stub = _stub(_source(), fps="24/1")
    seen = Probed(None, "01:00:00:00", Fraction(120), _tmcd(432000, Fraction(120)))
    assert source_start_timecodes(stub, {"cam": seen}) == (
        {"cam": StartTimecode(432000, Fraction(120))},
        [],
    )


def test_the_average_is_tried_before_r_frame_rate() -> None:
    """Validator R2-5 (N12): both snap, to different rates; the average wins (a VFR phone clip
    averaging 29.97 with r_frame_rate 30 is 29.97, as BRAW's average is)."""
    stub = _stub(_source(), fps="30000/1001")
    seen = Probed(None, "01:00:00:00", NTSC, None, None, Fraction(30))
    timecodes, _ = source_start_timecodes(stub, {"cam": seen})
    assert timecodes == {"cam": StartTimecode(108000, NTSC)}
    swapped = Probed(None, "01:00:00:00", Fraction(30), None, None, NTSC)
    assert source_start_timecodes(stub, {"cam": swapped})[0] == {
        "cam": StartTimecode(108000, Fraction(30))
    }


def test_only_the_sources_the_plan_plays_are_worked_out() -> None:
    """Validator R2-1: an unused source can neither refuse nor warn; it keeps the stub's
    values. The same source used is refused (a 25 fps tmcd on a 50 fps stream)."""
    ambiguous = Probed(None, "00:00:01:10", Fraction(50), _tmcd(35, Fraction(25)))
    slomo = Probed(None, "01:00:00:00", Fraction(120), None, None, Fraction(120))
    stub = _stub(
        _source(id="a", media="a.mov"),
        _source(id="b", media="b.mov", start_timecode=7),  # a stub value, rate missing
        _source(id="c", media="c.mov"),
        fps="24/1",
    )
    probed = {"a": slomo, "b": Probed(None, None, Fraction(50)), "c": ambiguous}
    timecodes, warnings = source_start_timecodes(stub, probed, {"a"})
    assert timecodes == {"a": StartTimecode(432000, Fraction(120))} and warnings == []
    with pytest.raises(ProduceInputError) as caught:  # b used: its rate is required
        source_start_timecodes(stub, probed, {"a", "b"})
    assert caught.value.code == "start-timecode-rate-required"
    with pytest.raises(ProduceInputError) as caught:  # c used: ambiguous
        source_start_timecodes(stub, probed, {"a", "c"})
    assert caught.value.code == "timecode-rate-ambiguous"
    assert source_start_timecodes(stub, probed, set()) == ({}, [])


@needs_ffmpeg
def test_a_real_120_fps_clip_with_a_tag_counts_at_120(tmp_path: Path) -> None:
    out = tmp_path / "slomo.mov"
    ffmpeg(
        "-f", "lavfi", "-i", "testsrc=size=64x48:rate=120:duration=1", "-f", "lavfi",
        "-i", "sine=d=1", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-timecode", "01:00:00:00", str(out),
    )  # fmt: skip
    stub = _stub(_source(media=str(out)), fps="24/1")
    for audio in (open_audio(stub, ["cam"], None), {}):  # used by the cut, and not
        probed, warnings = probe_sources(stub, audio, None)
        assert warnings == []
        assert source_start_timecodes(stub, probed, {"cam"}) == (
            {"cam": StartTimecode(432000, Fraction(120))},
            [],
        )
