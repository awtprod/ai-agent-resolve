"""The known-trap registry, and proof that the fake reproduces each trap as observed.

The ops' tests rely on the fake behaving like Resolve where Resolve is awkward. Each check
below drives the fake's scripting API directly and asserts the trap's symptom, so a fake that
quietly stopped modelling a trap fails here, not only in some op test.
"""

from __future__ import annotations

import re
from collections.abc import Callable

import pytest

from aar.node.ops import list_ops
from aar.node.traps import TRAPS, get_trap, trap_ids

from .fake_resolve import FakeResolve, FakeTimeline

LISTED = {
    "append-endframe-exclusive",
    "append-fps-short",
    "append-still-ignores-range",
    "append-ungraded",
    "av-append-video-only",
    "volume-key",
    "setproperty-after-color",
    "rcm-input-colorspace",
    "custom-res-resets-colour-science",
    "multicam-angles",
    "deliver-clobber",
    "render-uses-original-media",
    "pm-folder-scoped-load",
    "source-frame-readback-rounding",
    "cloud-proxy-only-no-audio",
}


def test_every_known_trap_is_registered_with_all_fields() -> None:
    assert set(trap_ids()) == LISTED
    for trap in TRAPS.values():
        assert re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", trap.id)
        for text in (trap.title, trap.symptom, trap.cause, trap.fix, trap.confirmed):
            assert text.strip(), trap.id
        assert "{" not in trap.confirmed and "{" not in trap.presumed
        assert trap.fully_confirmed == (trap.presumed == "")
        assert set(trap.to_dict()) == {
            "id",
            "title",
            "symptom",
            "cause",
            "fix",
            "confirmed",
            "presumed",
        }


def test_presumed_behaviours_are_marked() -> None:
    # The jam on setters other than SetProperty is assumed, not seen: it must say so.
    assert "SetFades" in TRAPS["setproperty-after-color"].presumed
    assert TRAPS["volume-key"].fully_confirmed


def test_trap_texts_are_generic() -> None:
    windows_path = re.compile(r"\b[A-Za-z]:[\\/]")
    for trap in TRAPS.values():
        text = " ".join(trap.to_dict().values())
        assert not windows_path.search(text), trap.id
        assert "\\" not in text, trap.id


def test_unknown_trap_id_raises() -> None:
    with pytest.raises(KeyError):
        get_trap("no-such-trap")


def test_ops_cite_only_registered_traps_and_every_trap_is_cited_or_handled() -> None:
    cited = {t for spec in list_ops() for t in spec.traps}
    assert cited <= set(TRAPS)
    # deliver-clobber is handled by the broker; the rest are cited by the ops that handle them
    assert cited | {"deliver-clobber"} >= LISTED


# -- the fake reproduces each trap ------------------------------------------------------------


def _world() -> tuple[FakeResolve, FakeTimeline]:
    fake = FakeResolve({"Show": ["Main"]}, current="Show", folder=("Shows",))
    return fake, fake.project("Show").timeline("Main")


def _endframe_exclusive() -> None:
    fake, _ = _world()
    clip = fake.project("Show").media_pool.add_clip((), "cam", fps=29.97, frames=1000)
    (item, _) = fake.project("Show").media_pool.AppendToTimeline(
        [{"mediaPoolItem": clip, "startFrame": 100, "endFrame": 130}]
    )
    assert item.GetEnd() - item.GetStart() == 30  # [100, 130): endFrame not included


def _fps_short() -> None:
    fake, _ = _world()
    clip = fake.project("Show").media_pool.add_clip((), "24p", fps=24.0, frames=1000)
    (item, _) = fake.project("Show").media_pool.AppendToTimeline(
        [{"mediaPoolItem": clip, "startFrame": 0, "endFrame": 7}]
    )
    # 7 frames at 24 fps fill 8.74 frames at 29.97: lands 8, one short of the rounded 9.
    assert item.GetEnd() - item.GetStart() == 8


def _still_ignores_range() -> None:
    fake, _ = _world()
    still = fake.project("Show").media_pool.add_clip((), "card", kind="still", frames=1)
    (item,) = fake.project("Show").media_pool.AppendToTimeline(
        [{"mediaPoolItem": still, "startFrame": 0, "endFrame": 48, "recordFrame": 108000}]
    )
    assert (item.GetStart(), item.GetEnd() - item.GetStart()) == (108001, 150)


def _ungraded() -> None:
    fake = FakeResolve({"Show": ["Main", "Graded"]}, current="Show", folder=("Shows",))
    project = fake.project("Show")
    clip = project.media_pool.add_clip((), "cam", frames=1000)
    graded = project.timeline("Graded").place("video", 1, clip, 109000, 50)
    graded.graph.nodes[0].lut = "look.cube"
    (item, _) = project.media_pool.AppendToTimeline(
        [{"mediaPoolItem": clip, "startFrame": 0, "endFrame": 30}]
    )
    assert item.timeline is project.timeline("Main")
    assert item.GetNodeGraph().GetLUT(1) == ""  # appended ungraded
    item.graph.nodes[0].lut = "per-clip.cube"
    assert graded.CopyGrades([item])  # from another timeline
    assert item.GetNodeGraph().GetLUT(1) == "look.cube"  # the whole graph replaced


def _av_video_only() -> None:
    fake, tl = _world()
    braw = fake.project("Show").media_pool.add_clip((), "cam", frames=1000, braw=True)
    returned = fake.project("Show").media_pool.AppendToTimeline(
        [{"mediaPoolItem": braw, "startFrame": 0, "endFrame": 30}]
    )
    assert [i.GetType() for i in returned] == ["video"]
    assert len(tl.GetItemListInTrack("audio", 1)) == 1  # but the audio item is there


def _volume_key() -> None:
    fake, tl = _world()
    clip = fake.project("Show").media_pool.add_clip((), "cam", frames=1000)
    sound = tl.place("audio", 1, clip, 108000, 30)
    assert sound.SetProperty("Volume", -6.0) is False
    assert sound.GetProperty("AudioVolume") == 0.0
    assert sound.SetProperties({"AudioVolume": -6.0})
    assert sound.GetProperty("AudioVolume") == -6.0


def _after_color() -> None:
    fake, tl = _world()
    clip = fake.project("Show").media_pool.add_clip((), "cam", frames=1000)
    sound = tl.place("audio", 1, clip, 108000, 30)
    fake.OpenPage("color")
    fake.OpenPage("edit")
    assert sound.SetProperties({"AudioVolume": -1.0}) is False
    fake.OpenPage("media")
    fake.OpenPage("edit")
    assert sound.SetProperties({"AudioVolume": -1.0}) is True


def _rcm_input() -> None:
    fake, _ = _world()
    pool = fake.project("Show").media_pool
    graphic = pool.import_rec709((), "graphic")
    assert graphic.GetClipProperty()["Input Color Space"] == "Blackmagic Design Film Gen 5"


def _custom_res() -> None:
    fake, tl = _world()
    assert tl.SetSettings({"useCustomSettings": "1", "timelineResolutionWidth": "1080"})
    assert tl.GetSettings()["colorScienceMode"] == "davinciYRGB"
    assert fake.project("Show").GetSettings()["colorScienceMode"] == "davinciYRGBColorManagedv2"


def _multicam() -> None:
    fake, tl = _world()
    multi = fake.project("Show").media_pool.add_clip((), "mc", kind="multicam", frames=1000)
    placed = tl.place("video", 1, multi, 108000, 10)
    assert placed.angle == 1
    (item, *_) = fake.project("Show").media_pool.AppendToTimeline(
        [{"mediaPoolItem": multi, "startFrame": 0, "endFrame": 30, "recordFrame": 109000}]
    )
    assert item.angle is None


def _deliver_clobber() -> None:
    fake, _ = _world()
    project = fake.project("Show")
    project.SetRenderSettings({"TargetDir": "/renders/human"})
    assert project.LoadRenderPreset("YouTube 1080p")
    assert project.deliver.settings["TargetDir"] != "/renders/human"


def _original_media() -> None:
    fake, _ = _world()
    clip = fake.project("Show").media_pool.add_clip((), "cam", online="Offline", proxy="1920x1080")
    props = clip.GetClipProperty()
    assert (props["Online Status"], props["Proxy"]) == ("Offline", "1920x1080")


def _pm_scoped_load() -> None:
    fake, _ = _world()
    fake.add_project("Elsewhere only", ["x"], ("Archive",))
    pm = fake.GetProjectManager()
    assert pm.GetCurrentFolder() == "Shows"
    assert pm.LoadProject("Elsewhere only") is None  # not in the current folder
    pm.GotoRootFolder()
    assert pm.GetCurrentFolder() == ""


def _source_readback() -> None:
    fake, _ = _world()
    fake.source_readback_skew = -1
    clip = fake.project("Show").media_pool.add_clip((), "cam", frames=1000)
    (item, _) = fake.project("Show").media_pool.AppendToTimeline(
        [{"mediaPoolItem": clip, "startFrame": 100, "endFrame": 130}]
    )
    assert item.GetSourceStartFrame() == 99


def _cloud_proxy_only() -> None:
    fake, _ = _world()
    pool = fake.project("Show").media_pool
    # Proxy-only: a resolution proxy is present while the original reads offline, so a build
    # uses the (audio-less) proxy.
    proxy_only = pool.add_clip((), "cam", online="Offline", proxy="1920x1080")
    props = proxy_only.GetClipProperty()
    assert (props["Online Status"], props["Proxy"]) == ("Offline", "1920x1080")
    # The status constants are floats, but the clip property reads back as the string "-1".
    assert fake.CLOUD_SYNC_DEFAULT == -1.0 and props["Cloud Sync"] == "-1"
    # A proxy that is itself "Offline" is not a playable proxy (so not the proxy-only case).
    proxy_gone = pool.add_clip((), "cam2", online="Offline", proxy="Offline")
    assert proxy_gone.GetClipProperty()["Proxy"] == "Offline"
    # A fully-synced clip is neither.
    good = pool.add_clip((), "ok", online="Online")
    assert good.GetClipProperty()["Online Status"] == "Online"


BEHAVIOUR: dict[str, Callable[[], None]] = {
    "append-endframe-exclusive": _endframe_exclusive,
    "append-fps-short": _fps_short,
    "append-still-ignores-range": _still_ignores_range,
    "append-ungraded": _ungraded,
    "av-append-video-only": _av_video_only,
    "volume-key": _volume_key,
    "setproperty-after-color": _after_color,
    "rcm-input-colorspace": _rcm_input,
    "custom-res-resets-colour-science": _custom_res,
    "multicam-angles": _multicam,
    "deliver-clobber": _deliver_clobber,
    "render-uses-original-media": _original_media,
    "pm-folder-scoped-load": _pm_scoped_load,
    "source-frame-readback-rounding": _source_readback,
    "cloud-proxy-only-no-audio": _cloud_proxy_only,
}


def test_every_trap_has_a_behavioural_check() -> None:
    assert set(BEHAVIOUR) == set(TRAPS)


@pytest.mark.parametrize("trap_id", sorted(BEHAVIOUR))
def test_the_fake_reproduces_the_trap(trap_id: str) -> None:
    BEHAVIOUR[trap_id]()
