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

from aar.media.probe import parse_probe_json, timecode_frames
from aar.plan.model import EditPlan
from aar.plan.produce.audio import open_audio
from aar.plan.produce.build import Probed, StartTimecode, probe_sources, source_start_timecodes
from aar.plan.produce.errors import ProduceInputError

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
