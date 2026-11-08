"""S18b hub media: every source resolves to a readable, probed hub file, or is refused with a
typed code before anything renders."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest
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
    make_media(root / "show" / "A001.mov", Spec("red", 440, 24, size=(64, 36)))
    make_media(root / "B 002 (wide).mov", Spec("green", 660, 24, size=(64, 36)))
    make_media(root / "pal.mov", Spec("blue", 880, 25, size=(64, 36), rate=Fraction(25)))
    make_media(root / "silent.mov", Spec("yellow", 330, 24, size=(64, 36), audio=False))
    (root / "junk.mov").write_bytes(b"not a media file" * 64)
    return root


def _resolve(source: Source, **kwargs: object):
    kwargs.setdefault("media_map", None)
    kwargs.setdefault("media_root", None)
    kwargs.setdefault("fps_tolerance", 0.001)
    return resolve_source(source, FPS, **kwargs)  # type: ignore[arg-type]


def _refused(source: Source, code: str, **kwargs: object) -> RenderError:
    with pytest.raises(RenderError) as info:
        _resolve(source, **kwargs)
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


def test_media_that_cannot_be_rendered_is_refused(hub_dir: Path) -> None:
    _refused(Source(id="a", media=str(hub_dir / "junk.mov")), "media_unreadable")
    error = _refused(Source(id="a", media=str(hub_dir / "pal.mov")), "fps_not_renderable")
    assert error.details["fps"] == "25"
    wrong = Source(id="a", media=str(hub_dir / "show" / "A001.mov"), frame_size=[1080, 1080])
    _refused(wrong, "frame_size_mismatch")
    # Any size with the file's display aspect is the same picture (a face track's size).
    ok = Source(id="a", media=str(hub_dir / "show" / "A001.mov"), frame_size=[3840, 2160])
    assert _resolve(ok).display == (64, 36)
    # 25 fps is 4 % off 23.976; a near-identical rate is accepted with a wider tolerance.
    assert (
        _resolve(Source(id="a", media=str(hub_dir / "pal.mov")), fps_tolerance=0.045).frames == 25
    )


def test_a_file_without_sound_renders_silent(hub_dir: Path) -> None:
    media = _resolve(Source(id="a", media=str(hub_dir / "silent.mov")))
    assert media.channels is None


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


def test_a_timeline_that_plays_past_the_end_of_a_file_is_refused(hub_dir: Path) -> None:
    from export_support import permissive

    plan = _plan(str(hub_dir / "show" / "A001.mov"), 30)
    hub = {"a": _resolve(plan.sources[0])}
    with pytest.raises(RenderError) as info:
        build_timeline(render.render_layout(plan, hub, permissive()), hub)
    assert info.value.code == "media_too_short"
    assert info.value.details == {"source": "a", "needed_frames": 30, "media_frames": 24}
    fits = _plan(str(hub_dir / "show" / "A001.mov"), 24)
    assert build_timeline(render.render_layout(fits, hub, permissive()), hub).total_frames == 24
