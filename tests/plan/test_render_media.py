"""S18b hub media: every source resolves to a readable, probed hub file that can be cut exactly,
or is refused with a typed code before anything renders (round 2: exact rates, no VFR, frame
counts from the video stream, stream start times kept)."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest
from export_support import permissive
from render_support import Spec, make_media

from aar.plan import render
from aar.plan.export import media_map_from_mapping
from aar.plan.model import EditPlan, Fps, Source
from aar.plan.render.errors import RenderError
from aar.plan.render.media import resolve_source
from aar.plan.render.timeline import build_timeline

FPS = Fps(num=24000, den=1001)


@pytest.fixture(scope="module")
def hub_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("hub")
    (root / "show").mkdir()
    small = (64, 36)
    make_media(root / "show" / "A001.mov", Spec("red", 440, 24, size=small))
    make_media(root / "B 002 (wide).mov", Spec("green", 660, 24, size=small))
    make_media(root / "pal.mov", Spec("blue", 880, 25, size=small, rate=Fraction(25)))
    make_media(root / "film.mov", Spec("blue", 880, 24, size=small, rate=Fraction(24)))
    make_media(root / "ntsc30.mov", Spec("blue", 880, 30, size=small, rate=Fraction(30)))
    make_media(root / "silent.mov", Spec("yellow", 330, 24, size=small, audio=False))
    make_media(root / "vfr.mov", Spec("red", 440, 120, size=small, drop=(60, 90)))
    make_media(root / "early.mov", Spec("red", 440, 120, size=small, drop=(5,)))
    make_media(root / "late.mov", Spec("red", 440, 48, size=small, audio_offset_s=0.25))
    make_media(root / "anamorphic.mov", Spec("red", 440, 24, size=(48, 36), sar="4/3"))
    make_media(root / "rotated.mov", Spec("red", 440, 24, size=small, rotate=90))
    # Matroska keeps no stream duration in its header; its audio runs 3 s past the video.
    make_media(root / "long_audio.mkv", Spec("red", 440, 24, size=small, container="matroska"))
    (root / "junk.mov").write_bytes(b"not a media file" * 64)
    return root


def _resolve(source: Source, fps: Fps = FPS, **kwargs: object):
    kwargs.setdefault("media_map", None)
    kwargs.setdefault("media_root", None)
    return resolve_source(source, fps, **kwargs)  # type: ignore[arg-type]


def _refused(source: Source, code: str, fps: Fps = FPS, **kwargs: object) -> RenderError:
    with pytest.raises(RenderError) as info:
        _resolve(source, fps, **kwargs)
    assert info.value.code == code, info.value.message
    return info.value


def test_a_hub_path_that_exists_is_used_and_probed(hub_dir: Path) -> None:
    media = _resolve(Source(id="a", media=str(hub_dir / "show" / "A001.mov")))
    assert media.rule == "plan-path" and media.path == hub_dir / "show" / "A001.mov"
    assert media.display == (64, 36) and media.frames == 24 and media.channels == 1
    assert media.fps == Fraction(24000, 1001)
    url = _resolve(Source(id="a", media=f"file://{hub_dir}/B%20002%20(wide).mov"))
    assert url.path == hub_dir / "B 002 (wide).mov"
    rel = _resolve(Source(id="a", media="show/A001.mov"), media_root=hub_dir)
    assert rel.path == hub_dir / "show" / "A001.mov"


def test_an_edit_pc_path_goes_through_the_media_map(hub_dir: Path) -> None:
    media_map = media_map_from_mapping({str(hub_dir): "E:\\CCN"})
    for media in (
        "E:\\CCN\\show\\A001.mov",
        "E:/CCN/show/A001.mov",
        "file:///E:/CCN/show/A001.mov",
    ):
        got = _resolve(Source(id="a", media=media), media_map=media_map)
        assert got.rule == "media-map" and got.path == hub_dir / "show" / "A001.mov", media
    spaced = _resolve(Source(id="b", media="E:\\CCN\\B 002 (wide).mov"), media_map=media_map)
    assert spaced.path.name == "B 002 (wide).mov"


def test_media_not_on_the_hub_is_refused(hub_dir: Path) -> None:
    media_map = media_map_from_mapping({str(hub_dir): "E:\\CCN"})
    cases = [
        (Source(id="a", media="E:\\CCN\\show\\A001.mov"), {}, "no-media-map"),
        (Source(id="a", media="F:\\Other\\A001.mov"), {"media_map": media_map}, "not-in-media-map"),
        (Source(id="a", media=str(hub_dir / "missing.mov")), {}, "missing"),
        (Source(id="a", media="E:\\CCN\\missing.mov"), {"media_map": media_map}, "missing"),
        (Source(id="a", media="Multicam 1"), {}, "pool-name"),
        (Source(id="a", media="show/A001.mov"), {}, "relative"),
        (Source(id="a", media=str(hub_dir / "show")), {}, "unreadable"),
    ]
    for source, kwargs, reason in cases:
        error = _refused(source, "media_not_on_hub", **kwargs)
        assert error.details["reason"] == reason, (source.media, error.details)


def test_a_rate_that_is_not_the_plans_is_refused(hub_dir: Path) -> None:
    _refused(Source(id="a", media=str(hub_dir / "junk.mov")), "media_unreadable")
    error = _refused(Source(id="a", media=str(hub_dir / "pal.mov")), "source_rate_mismatch")
    assert error.details["fps"] == "25/1"
    # 24 in a 23.976 plan, and 30 in a 29.97 plan, are 0.1 % off: they drift 0.6 s in 10 minutes.
    _refused(Source(id="a", media=str(hub_dir / "film.mov")), "source_rate_mismatch")
    ntsc = Fps(num=30000, den=1001)
    error = _refused(
        Source(id="a", media=str(hub_dir / "ntsc30.mov")), "source_rate_mismatch", ntsc
    )
    assert "conform" in error.fix_hint
    assert _resolve(Source(id="a", media=str(hub_dir / "ntsc30.mov")), Fps(num=30)).frames == 30


def test_a_variable_frame_rate_is_refused(hub_dir: Path) -> None:
    # Two frames dropped with their timestamps kept: avg 23.976 * 118/120 is inside 0.1 % of the
    # base rate only at long lengths; here the frame count against the duration gives it away.
    error = _refused(Source(id="a", media=str(hub_dir / "vfr.mov")), "source_vfr")
    assert "conform" in error.fix_hint
    # A drop among the first packets: their timestamps are uneven.
    early = _refused(Source(id="a", media=str(hub_dir / "early.mov")), "source_vfr")
    assert "packet" in early.message or "average" in early.message or "frames" in early.message


def test_frame_size_display_size_and_its_aspect(hub_dir: Path) -> None:
    wrong = Source(id="a", media=str(hub_dir / "show" / "A001.mov"), frame_size=[1080, 1080])
    _refused(wrong, "frame_size_mismatch")
    # Any size with the file's display aspect is the same picture (a face track's size), to
    # within 0.5 %: 1000x564 is 0.3 % off 16:9, 1000x557 is 1 % off.
    near = Source(id="a", media=str(hub_dir / "show" / "A001.mov"), frame_size=[1000, 564])
    assert _resolve(near).display == (64, 36)
    off = Source(id="a", media=str(hub_dir / "show" / "A001.mov"), frame_size=[1000, 557])
    _refused(off, "frame_size_mismatch")
    # The display size applies the pixel aspect (48x36 at 4:3 shows as 64x36) and the rotation.
    wide = Source(id="a", media=str(hub_dir / "anamorphic.mov"), frame_size=[64, 36])
    got = _resolve(wide)
    assert got.display == (64, 36) and got.sample_aspect == Fraction(4, 3)
    turned = Source(id="a", media=str(hub_dir / "rotated.mov"), frame_size=[36, 64])
    assert _resolve(turned).display == (36, 64)


def test_a_file_without_sound_renders_silent(hub_dir: Path) -> None:
    media = _resolve(Source(id="a", media=str(hub_dir / "silent.mov")))
    assert media.channels is None


def test_stream_start_times_and_the_video_frame_count(hub_dir: Path) -> None:
    late = _resolve(Source(id="a", media=str(hub_dir / "late.mov")))
    assert late.audio_start_s - late.video_start_s == Fraction(1, 4)
    mkv = _resolve(Source(id="a", media=str(hub_dir / "long_audio.mkv")))
    assert mkv.frames == 24  # the video's own frames, not the container's (longer) duration


def _plan(media: str, src_out: int) -> EditPlan:
    return EditPlan.model_validate(
        {
            "project": "P",
            "timeline": "T",
            "kind": "longform",
            "fps": "24000/1001",
            "resolution": {"width": 64, "height": 36},
            "profile": "x",
            "sources": [{"id": "a", "media": media}],
            "segments": [{"source": "a", "src_in": 0, "src_out": src_out}],
        }
    )


@pytest.mark.parametrize("name", ["show/A001.mov", "long_audio.mkv"])
def test_a_segment_past_the_last_video_frame_is_refused(hub_dir: Path, name: str) -> None:
    plan = _plan(str(hub_dir / name), 25)
    hub = {"a": _resolve(plan.sources[0])}
    with pytest.raises(RenderError) as info:
        build_timeline(render.render_layout(plan, hub, permissive()), hub)
    assert info.value.code == "segment_past_media_end"
    assert info.value.details["location"] == "segments[0]"
    assert info.value.details["needed_frames"] == 25 and info.value.details["media_frames"] == 24
    fits = _plan(str(hub_dir / name), 24)  # the last frame exactly
    assert build_timeline(render.render_layout(fits, hub, permissive()), hub).total_frames == 24
