"""S18b filter graphs: golden graphs for the CCN long-form and reframed short fixtures, and the
rules every graph follows (frame-index trims, whole-frame dissolves, exact sample positions).

No ffmpeg: the hub files are described, not probed. Set ``AAR_UPDATE_RENDER_GOLDEN=1`` to rewrite
the goldens after a reviewed change.
"""

from __future__ import annotations

import os
import re
from fractions import Fraction
from pathlib import Path

import pytest
from export_support import EXPORT_FIXTURES, ccn, load_plan
from plan_support import FIXTURES

from aar.media.probe import MediaInfo
from aar.plan.export import export_fcpxml
from aar.plan.model import EditPlan
from aar.plan.render import render_layout
from aar.plan.render.graph import mix_graph, sample_at, video_graph
from aar.plan.render.media import HubSource
from aar.plan.render.timeline import build_timeline

RENDER_FIXTURES = FIXTURES / "render"
NAMES = ("ccn_longform_v2", "short_reframe")


def fake_hub(plan: EditPlan, size: tuple[int, int] = (320, 180)) -> dict[str, HubSource]:
    hub: dict[str, HubSource] = {}
    for source in plan.sources:
        path = Path(f"/srv/media/{source.id}.mov")
        info = MediaInfo(str(path), "mov", None, 1, None)
        hub[source.id] = HubSource(
            source.id,
            source.media,
            path,
            "media-map",
            info,
            size,
            None,
            Fraction(plan.fps.num, plan.fps.den),
            source.duration,
            1,
        )
    return hub


def graphs(name: str) -> tuple[str, str]:
    plan = load_plan(EXPORT_FIXTURES / f"{name}.plan.json")
    hub = fake_hub(plan)
    tl = build_timeline(render_layout(plan, hub, ccn()), hub)
    video = video_graph(tl, hub, font="font.ttf", luts={}, out_pix_fmt="yuv420p")
    return video.text, mix_graph(tl, hub).text


@pytest.mark.parametrize("name", NAMES)
def test_graphs_match_the_goldens(name: str) -> None:
    video, mix = graphs(name)
    for suffix, text in ((".video.ffgraph", video), (".mix.ffgraph", mix)):
        golden = RENDER_FIXTURES / f"{name}{suffix}"
        if os.environ.get("AAR_UPDATE_RENDER_GOLDEN") == "1":
            golden.write_text(text, encoding="utf-8")
        assert text == golden.read_text(encoding="utf-8"), f"{golden.name} changed"


@pytest.mark.parametrize("name", NAMES)
def test_trims_are_frame_indices_and_samples_never_seconds(name: str) -> None:
    video, mix = graphs(name)
    trims = re.findall(r"(?<![a-z])trim=([^,\[;]*)", video)
    assert trims and all(re.fullmatch(r"end_frame=\d+", t) for t in trims)  # after the seek
    atrims = re.findall(r"atrim=([^,\[;]*)", mix)
    assert atrims and all(
        re.fullmatch(r"start_pts=\d+:end_pts=\d+|end_sample=\d+", t) for t in atrims
    )
    assert "asplit" not in mix and "split=" not in video  # one input per use
    assert ("between(t" not in video and "enable='between(n," in video) or name == "short_reframe"


def test_longform_layout_is_the_exports() -> None:
    """The render's dissolves and the export report's transitions are the same placements."""
    plan = load_plan(EXPORT_FIXTURES / "ccn_longform_v2.plan.json")
    hub = fake_hub(plan)
    tl = build_timeline(render_layout(plan, hub, ccn()), hub)
    report = export_fcpxml(plan, ccn(), allow_unknown_timecode=True).report
    exported = {(t["cut"], t["frames"]) for t in report["transitions"] if t["form"] == "transition"}
    exported -= {(t["cut"], t["frames"]) for t in report["transitions"] if t["location"] == "head"}
    assert {(d.cut, d.frames) for d in tl.dissolves} == exported == {(306, 15)}
    outro = next(d for d in tl.dissolves if d.location == "outro.transition")
    # camA2 has 880 frames after its out point and the outro none before its in: the dissolve
    # starts on the cut (all on the outgoing tail), as the export's alignment rule says.
    assert outro.window_start == 306
    video, _ = graphs("ccn_longform_v2")
    assert ":duration=0.625625:offset=12.76275[" in video  # 15 and 306 frames at 23.976
    # The head's 15-frame dissolve from black is a fade on the first picture.
    assert "fade=t=in:start_frame=0:nb_frames=15" in video


def test_longform_split_crossfade_sounds_overlap_by_its_window() -> None:
    plan = load_plan(EXPORT_FIXTURES / "ccn_longform_v2.plan.json")
    hub = fake_hub(plan)
    tl = build_timeline(render_layout(plan, hub, ccn()), hub)
    by = {s.location: s for s in tl.sounds}
    # transitions[0]: 12 frames after segment 2 (cut at 216), centred: 6 before and 6 after.
    assert (by["segments[2]"].start, by["segments[2]"].end) == (144, 222)
    assert (by["segments[3]"].start, by["segments[3]"].end) == (210, 306)
    assert by["segments[2]"].fade_out == 12 and by["segments[3]"].fade_in == 12
    assert by["segments[3]"].src_start == 24  # its handle: 6 frames before src_in 30
    assert by["segments[0]"].fade_in == 24  # head.audio_fade_frames
    assert by["outro"].fade_out == 2 and by["outro"].gain_db == -1.0  # end fade; A5's gain
    assert by["segments[2]"].gain_db == -2.0
    # Muted A1-A4 feed the A5 bus: audible, as in S18.
    assert set(by) == {"segments[0]", "segments[1]", "segments[2]", "segments[3]", "outro"}
    _, mix = graphs("ccn_longform_v2")
    start = sample_at(plan.fps, 24)
    assert start == 24 * 2002  # 23.976 at 48 kHz: 2002 samples a frame, exactly
    assert f"atrim=start_pts={start}:end_pts={start + (306 - 210) * 2002}" in mix
    assert f"adelay=delays={210 * 2002}S:all=1" in mix  # placed at record 210
    assert "amix=inputs=5:normalize=0:duration=longest" in mix  # every sound, summed


def test_short_reframe_windows_move_per_frame() -> None:
    video, _ = graphs("short_reframe")
    perspectives = re.findall(
        r"perspective=(.*?):interpolation=cubic:sense=source:eval=(\w+)", video
    )
    assert [mode for _, mode in perspectives] == ["frame", "frame"]
    assert all("(in-1)" in corners for corners, _ in perspectives)


def test_every_use_is_its_own_input_seeked_before_its_range() -> None:
    from aar.plan.render.graph import mix_graph as mix_of
    from aar.plan.render.graph import picture_use

    plan = load_plan(EXPORT_FIXTURES / "ccn_longform_v2.plan.json")
    hub = fake_hub(plan)
    tl = build_timeline(render_layout(plan, hub, ccn()), hub)
    video = video_graph(tl, hub, font=None, luts={}, out_pix_fmt="yuv420p")
    assert len(video.uses) == len(tl.pictures) == 5  # camA twice: two inputs, no split
    frame = Fraction(1001, 24000)
    seeks = [u.seek_s for u in video.uses]
    assert seeks[0] is None  # segment 0 starts at frame 0: no seek
    assert seeks[1] == (72 - Fraction(1, 2)) * frame  # half a frame before frame 72
    assert seeks[2] == (144 - Fraction(1, 2)) * frame
    assert video.uses[1].args()[:4] == ["-threads", "1", "-ss", "2.982146"]
    assert picture_use(tl.pictures[0], hub["camA"], plan.fps).span_s > 72 * frame
    mix = mix_of(tl, hub)
    assert len(mix.uses) == len(tl.sounds) == 5
    # A sound's input seeks a quarter second before its first sample (then cuts by time).
    assert mix.uses[1].seek_s == 72 * frame - Fraction(1, 4)
