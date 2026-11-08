"""Source.frame_size in the plan producer (S18): the stub's own size, else the media probe's display
size (pixel aspect and rotation applied), else the face track's; disagreements are warnings."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest
from produce_support import ffmpeg, needs_ffmpeg

from aar.media import probe
from aar.media.probe import parse_probe_json
from aar.plan.model import EditPlan
from aar.plan.produce.audio import open_audio
from aar.plan.produce.build import probe_frame_sizes, source_frame_sizes
from aar.plan.produce.inputs import FaceTrack


def _stub(*sources: dict[str, object]) -> EditPlan:
    return EditPlan.model_validate(
        {
            "project": "P",
            "timeline": "T",
            "kind": "short",
            "fps": "30000/1001",
            "resolution": {"width": 1080, "height": 1920},
            "profile": "ccn",
            "sources": list(sources),
            "segments": [],
        }
    )


def _faces(size: list[int]) -> FaceTrack:
    return FaceTrack.model_validate(
        {
            "schema": "aar.faces/1",
            "header": {"frame_size": size, "input": {"path": "cam.mov"}},
            "frames": [],
        }
    )


def _source(**extra: object) -> dict[str, object]:
    return {"id": "cam", "media": "cam.mov", **extra}


# -- precedence --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("stub_size", "probed", "face", "want", "warns"),
    [
        # the stub's explicit size wins over everything ...
        ([1920, 1080], None, None, (1920, 1080), []),
        ([1920, 1080], (3840, 2160), [640, 360], (1920, 1080), []),
        # ... and says so when the media or the face track show another aspect
        ([1920, 1080], (1080, 1920), None, (1920, 1080), ["media probe"]),
        ([1920, 1080], None, [360, 640], (1920, 1080), ["face track"]),
        # then the media probe
        (None, (3840, 2160), None, (3840, 2160), []),
        (None, (160, 90), [640, 360], (160, 90), []),  # same aspect, another scale
        (None, (4096, 2160), [640, 338], (4096, 2160), []),  # the face track rounds to even px
        (None, (1920, 1080), [360, 640], (1920, 1080), ["face track"]),  # disagree: the probe
        # then the face track
        (None, None, [640, 360], (640, 360), []),
    ],
)
def test_frame_size_precedence(
    stub_size: list[int] | None,
    probed: tuple[int, int] | None,
    face: list[int] | None,
    want: tuple[int, int],
    warns: list[str],
) -> None:
    stub = _stub(_source(**({"frame_size": stub_size} if stub_size else {})))
    faces = {"cam": _faces(face)} if face else {}
    sizes, warnings = source_frame_sizes(stub, {"cam": probed}, faces)
    assert sizes == {"cam": want}
    assert [w.code for w in warnings] == ["frame-size-mismatch"] * len(warns)
    for warning, other in zip(warnings, warns, strict=True):
        assert other in warning.message and warning.source == "cam"


def test_an_audio_only_source_gets_no_frame_size() -> None:
    stub = _stub(_source(media="cam.wav"))
    assert source_frame_sizes(stub, {"cam": None}, {}) == ({}, [])


# -- the probe: pixel aspect and rotation ------------------------------------------------------


def _video(**raw: object) -> dict[str, object]:
    return {"streams": [{"index": 0, "codec_type": "video", "width": 1920, "height": 1080, **raw}]}


@pytest.mark.parametrize(
    ("raw", "rotation", "aspect", "size"),
    [
        ({}, 0, None, (1920, 1080)),
        ({"sample_aspect_ratio": "1:1"}, 0, Fraction(1), (1920, 1080)),
        ({"sample_aspect_ratio": "0:1"}, 0, None, (1920, 1080)),  # unknown aspect: as coded
        (
            {"side_data_list": [{"side_data_type": "Display Matrix", "rotation": -90}]},
            270,
            None,
            (1080, 1920),
        ),
        ({"side_data_list": [{"rotation": 90}]}, 90, None, (1080, 1920)),
        ({"side_data_list": [{"rotation": 180}]}, 180, None, (1920, 1080)),
        ({"tags": {"rotate": "90"}}, 90, None, (1080, 1920)),  # the older tag
        ({"width": 1440, "sample_aspect_ratio": "4:3"}, 0, Fraction(4, 3), (1920, 1080)),
        (
            {"width": 1440, "sample_aspect_ratio": "4:3", "side_data_list": [{"rotation": 90}]},
            90,
            Fraction(4, 3),
            (1080, 1920),
        ),
    ],
)
def test_probe_reads_the_display_size(
    raw: dict[str, object], rotation: int, aspect: Fraction | None, size: tuple[int, int]
) -> None:
    info = parse_probe_json(_video(**raw), "x.mov", 1)
    video = info.video
    assert video is not None
    assert (video.rotation, video.sample_aspect, video.display_size) == (rotation, aspect, size)
    assert info.to_dict()["streams"][0]["display_size"] == list(size)


def test_an_audio_stream_has_no_display_size() -> None:
    info = parse_probe_json({"streams": [{"index": 0, "codec_type": "audio"}]}, "x.wav", 1)
    assert info.audio is not None and info.audio.display_size is None and info.video is None


# -- a real rotated file -----------------------------------------------------------------------


@pytest.fixture
def rotated(tmp_path: Path) -> Path:
    """A 160x90 phone clip with sound, stored with a 90 degree display rotation."""
    flat, out = tmp_path / "flat.mp4", tmp_path / "rotated.mp4"
    ffmpeg(
        "-f", "lavfi", "-i", "testsrc=size=160x90:rate=25", "-f", "lavfi", "-i", "sine=d=1",
        "-t", "1", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", str(flat),
    )  # fmt: skip
    ffmpeg("-display_rotation", "90", "-i", str(flat), "-c", "copy", str(out))
    return out


@needs_ffmpeg
def test_a_rotated_source_is_sized_as_shown(rotated: Path) -> None:
    video = probe(rotated).video
    assert video is not None and (video.width, video.height, video.rotation) == (160, 90, 90)
    assert video.display_size == (90, 160)
    stub = _stub(_source(media=str(rotated)))
    # A source of the cut: open_audio's probe carries the size ...
    assert open_audio(stub, ["cam"], None)["cam"].frame_size == (90, 160)
    # ... and a source the cut does not use is probed on its own.
    sizes, warnings = probe_frame_sizes(stub, {}, None)
    assert sizes == {"cam": (90, 160)} and warnings == []
    assert source_frame_sizes(stub, sizes, {})[0] == {"cam": (90, 160)}


@needs_ffmpeg
def test_unreadable_media_is_a_warning_and_missing_media_is_skipped(tmp_path: Path) -> None:
    junk = tmp_path / "junk.mov"
    junk.write_bytes(b"not a movie")
    stub = _stub(
        {"id": "junk", "media": str(junk)},
        {"id": "gone", "media": str(tmp_path / "gone.mov")},
        {"id": "pool", "media": "Ep1 Multicam"},
    )
    sizes, warnings = probe_frame_sizes(stub, {}, None)
    assert sizes == {}
    assert [(w.code, w.source) for w in warnings] == [("frame-size-unread", "junk")]
