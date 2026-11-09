"""S18b framing: each frame's crop window is S18's, and ffmpeg gets its exact corners.

No ffmpeg here. The corners are checked against S18's own transform maths (the FCPXML values
``transform_spec`` writes, mapped forward), and the perspective expressions are evaluated in Python
at every frame against the corners computed directly.
"""

from __future__ import annotations

import math
import random
import re

import pytest

from aar.plan.export.layout import ClipSpec, transform_spec
from aar.plan.export.verify import window_from_plan
from aar.plan.model import ReframeKey, Resolution, Segment, Transform
from aar.plan.render.framing import (
    Geometry,
    framing_for,
    picture_filters,
    reframe_at,
)

PORTRAIT = Resolution(width=1080, height=1920)
LANDSCAPE = Resolution(width=1920, height=1080)


def _forward(seg: Segment, frame_size: tuple[int, int], res: Resolution, u: float, v: float):
    """Where source point (u, v) lands in the frame, from S18's FCPXML transform: the source filled
    to the frame, scaled, rotated, then moved by ``position`` (percent of the frame height, y up).
    Returns image coordinates of the frame (x right, y down, from the top-left)."""
    spec = transform_spec(seg, frame_size, res.width, res.height)
    fill = max(res.width / frame_size[0], res.height / frame_size[1])
    px, py = (u - frame_size[0] / 2) * fill, (frame_size[1] / 2 - v) * fill
    sx, sy = spec.scale
    if spec.keyframes:
        sx, sy = spec.keyframes[0].scale
    pos = spec.keyframes[0].position if spec.keyframes else spec.position
    a = math.radians(spec.rotation)
    qx = math.cos(a) * sx * px - math.sin(a) * sy * py + pos[0] * res.height / 100
    qy = math.sin(a) * sx * px + math.cos(a) * sy * py + pos[1] * res.height / 100
    return res.width / 2 + qx, res.height / 2 - qy


@pytest.mark.parametrize("seed", range(40))
def test_corners_land_on_the_frame_corners_under_s18s_transform(seed: int) -> None:
    rng = random.Random(seed)
    res = rng.choice([PORTRAIT, LANDSCAPE])
    frame_size = rng.choice([(1920, 1080), (3840, 2160), (1080, 1920), (1440, 1080)])
    tf = Transform(
        zoom=rng.uniform(0.8, 1.6),
        pan=rng.uniform(-200, 200),
        tilt=rng.uniform(-200, 200),
        rotation_deg=rng.choice([0.0, rng.uniform(-30, 30)]),
        flip_h=rng.random() < 0.5,
        flip_v=rng.random() < 0.3,
    )
    key = ReframeKey(
        time=0, x=rng.uniform(0.3, 0.7), y=rng.uniform(0.3, 0.7), zoom=rng.uniform(1, 2)
    )
    seg = Segment(source="a", src_in=0, src_out=10, reframe=[key], transform=tf)
    geo = Geometry(tf, frame_size, frame_size, res)
    want = [(0, 0), (res.width, 0), (0, res.height), (res.width, res.height)]
    for (u, v), (wx, wy) in zip(geo.corners(key.x, key.y, key.zoom), want, strict=True):
        x, y = _forward(seg, frame_size, res, u, v)
        assert x == pytest.approx(wx, abs=1e-6)
        assert y == pytest.approx(wy, abs=1e-6)


def test_the_window_is_s18s_window_scaled_to_the_hub_file() -> None:
    tf = Transform(zoom=1.2, pan=30, tilt=-12)
    geo = Geometry(tf, (3840, 2160), (640, 360), PORTRAIT)
    got = geo.window(0.55, 0.45, 1.1)
    plan = window_from_plan(0.55, 0.45, 1.1, tf, (3840, 2160), 1080, 1920)
    k = 640 / 3840
    assert got == pytest.approx(tuple(v * k for v in plan))


def test_reframe_holds_before_and_after_and_interpolates_between() -> None:
    keys = [ReframeKey(time=10, x=0.4, y=0.5, zoom=1.0), ReframeKey(time=20, x=0.6, y=0.3, zoom=2)]
    assert reframe_at(keys, -5) == (0.4, 0.5, 1.0)  # a head handle frame holds the first key
    assert reframe_at(keys, 10) == (0.4, 0.5, 1.0)
    assert reframe_at(keys, 15) == pytest.approx((0.5, 0.4, 1.5))
    assert reframe_at(keys, 20) == (0.6, 0.3, 2.0)
    assert reframe_at(keys, 99) == (0.6, 0.3, 2.0)
    assert reframe_at([], 3) == (0.5, 0.5, 1.0)


def _clip(frame_size: tuple[int, int] | None, conform: bool) -> ClipSpec:
    return ClipSpec(
        "segments[0]", 0, 0, 10, 0, 10, "x", "all", 1, frame_size=frame_size, conform=conform
    )


def _ffeval(expr: str, frame_in: int) -> float:
    """Evaluate a perspective corner expression (numbers, + - * /, if, lt, in) as ffmpeg would."""
    py = re.sub(r"\bin\b", "IN", expr.replace("if(", "_if(").replace("lt(", "_lt("))
    env = {"_if": lambda c, a, b: a if c else b, "_lt": lambda a, b: float(a < b), "IN": frame_in}
    return float(eval(py, {"__builtins__": {}}, env))


def _final_corners(framing, frame: int) -> list[tuple[float, float]]:
    """The corners perspective must get at rendered ``frame``: S18's window corners, moved into
    the padded, cropped box and by half an output pixel (index versus area coordinates)."""
    plan = framing.window
    geo = framing.geometry
    tl, tr, bl, br = geo.corners(*reframe_at(list(framing.keys), frame - framing.lead))
    left, top, bw, bh = plan.box
    ax, ay = (tr[0] - tl[0]) / bw, (tr[1] - tl[1]) / bw
    bx, by = (bl[0] - tl[0]) / bh, (bl[1] - tl[1]) / bh
    dx = plan.pad - left + 0.5 * (ax + bx) - 0.5
    dy = plan.pad - top + 0.5 * (ay + by) - 0.5
    return [(u + dx, v + dy) for u, v in (tl, tr, bl, br)]


@pytest.mark.parametrize("lead", [0, 6])
def test_perspective_expressions_give_the_exact_corners_at_every_frame(lead: int) -> None:
    keys = [
        ReframeKey(time=0, x=0.5, y=0.5, zoom=1.0),
        ReframeKey(time=40, x=0.55, y=0.5, zoom=1.05),
        ReframeKey(time=79, x=0.6, y=0.45, zoom=1.1),
    ]
    tf = Transform(zoom=1.2, rotation_deg=4.0, flip_h=True)
    framing = framing_for(
        _clip((1920, 1080), True),
        keys=keys,
        transform=tf,
        display=(640, 360),
        resolution=PORTRAIT,
        lead=lead,
        frames=80 + lead + 5,
    )
    assert framing.mode == "window" and framing.window is not None and framing.window.moving
    exprs = framing.window.expressions
    for frame in range(80 + lead + 5):
        want = _final_corners(framing, frame)
        got = [_ffeval(e, frame + 1) for e in exprs]  # perspective's `in` counts from 1
        for i, (u, v) in enumerate(want):
            assert got[2 * i] == pytest.approx(u, abs=1e-6)
            assert got[2 * i + 1] == pytest.approx(v, abs=1e-6)
    # Every corner lies inside the crop box the filter gets.
    _, _, bw, bh = framing.window.box
    for frame in range(80 + lead + 5):
        for u, v in _final_corners(framing, frame):
            assert -0.5 <= u <= bw and -0.5 <= v <= bh


def test_a_static_window_is_evaluated_once() -> None:
    framing = framing_for(
        _clip((3840, 2160), True),
        keys=[],
        transform=Transform(zoom=1.1),
        display=(320, 180),
        resolution=LANDSCAPE,
        lead=0,
        frames=72,
    )
    assert framing.window is not None and not framing.window.moving
    filters = picture_filters(framing, LANDSCAPE)
    assert filters[0] == "format=yuv444p"
    assert any(f.startswith("crop=") and f.endswith(":exact=1") for f in filters)
    perspective = next(f for f in filters if f.startswith("perspective="))
    assert "eval=init" in perspective and "sense=source" in perspective
    assert filters[-1] == "scale=1920:1080"
    # The static zoom of 1.1 shows the centre 1/1.1 of the picture.
    assert framing.window.first == pytest.approx((160, 90, 320 / 1.1, 180 / 1.1))


def test_a_window_that_leaves_the_picture_is_padded_with_black() -> None:
    framing = framing_for(
        _clip((320, 180), True),
        keys=[],
        transform=Transform(pan=400.0),  # moves the picture right: the window reaches past x=0
        display=(320, 180),
        resolution=Resolution(width=320, height=180),
        lead=0,
        frames=5,
    )
    plan = framing.window
    assert plan is not None and plan.pad > 0
    assert plan.first[0] - plan.first[2] / 2 < 0  # S18's window starts left of the picture
    filters = picture_filters(framing, Resolution(width=320, height=180))
    assert f"pad=iw+{2 * plan.pad}:ih+{2 * plan.pad}:{plan.pad}:{plan.pad}:black" in filters


def test_unconformed_clips_scale_or_fit() -> None:
    same = framing_for(
        _clip(None, False),
        keys=[],
        transform=None,
        display=(640, 360),
        resolution=LANDSCAPE,
        lead=0,
        frames=1,
    )
    assert same.mode == "scale" and picture_filters(same, LANDSCAPE) == ["scale=1920:1080"]
    fit = framing_for(
        _clip(None, False),
        keys=[],
        transform=None,
        display=(640, 360),
        resolution=PORTRAIT,
        lead=0,
        frames=1,
    )
    assert fit.mode == "fit" and fit.fitted == (1080, 608)
    assert picture_filters(fit, PORTRAIT) == [
        "format=yuv444p",
        "scale=1080:608",
        "pad=1080:1920:0:656:black",
    ]
    # A 4:3 picture in an 854x480 frame: 640 wide, 107 px each side (4:4:4, so the left edge is
    # not rounded down to an even pixel and the picture sits exactly in the middle).
    centred = framing_for(
        _clip(None, False),
        keys=[],
        transform=None,
        display=(640, 480),
        resolution=Resolution(width=854, height=480),
        lead=0,
        frames=1,
    )
    assert centred.fitted == (640, 480)
    assert picture_filters(centred, Resolution(width=854, height=480))[-1] == (
        "pad=854:480:107:0:black"
    )
    # An odd fitted width is moved to the frame's parity, so both margins are whole and equal.
    odd = framing_for(
        _clip(None, False),
        keys=[],
        transform=None,
        display=(1000, 1001),
        resolution=Resolution(width=640, height=360),
        lead=0,
        frames=1,
    )
    assert odd.fitted is not None and (640 - odd.fitted[0]) % 2 == 0


def test_a_long_reframe_nests_logarithmically_and_stays_exact() -> None:
    keys = [
        ReframeKey(time=12 * i, x=0.5 + 0.1 * math.sin(i / 7), zoom=1 + 0.2 * (i % 3))
        for i in range(300)
    ]
    framing = framing_for(
        _clip((1920, 1080), True),
        keys=keys,
        transform=Transform(zoom=1.1),
        display=(640, 360),
        resolution=PORTRAIT,
        lead=0,
        frames=12 * 300,
    )
    assert framing.window is not None
    expr = framing.window.expressions[0]
    depth = deepest = 0
    for ch in expr:
        depth += {"(": 1, ")": -1}.get(ch, 0)
        deepest = max(deepest, depth)
    assert deepest < 40  # a chain of 300 pieces would nest 300 deep
    for frame in (0, 5, 1501, 2567, 12 * 299, 12 * 300 - 1):
        want = _final_corners(framing, frame)
        got = [_ffeval(e, frame + 1) for e in framing.window.expressions]
        assert got == pytest.approx([c for corner in want for c in corner], abs=1e-6)


def test_straight_runs_of_keyframes_merge_exactly() -> None:
    from aar.plan.render.framing import reframe_values, simplify_keys

    # A steady pan over 7200 keys (one every 12 frames) is two keys: the same path.
    keys = [
        ReframeKey(time=12 * i, x=0.3 + 0.4 * i / 7199, y=0.5, zoom=1.0 + 0.5 * i / 7199)
        for i in range(7200)
    ]
    merged = simplify_keys(keys)
    assert [k.time for k in merged] == [0, 12 * 7199]
    frames = 12 * 7199 + 1
    every = reframe_values(keys, 0, frames)
    fewer = reframe_values(merged, 0, frames)
    assert (
        max(
            abs(a - b) for u, v in zip(every, fewer, strict=True) for a, b in zip(u, v, strict=True)
        )
        < 1e-9
    )
    # A bend is kept.
    bent = [ReframeKey(time=0, x=0.4), ReframeKey(time=10, x=0.5), ReframeKey(time=20, x=0.5)]
    assert simplify_keys(bent) == bent
    framing = framing_for(
        _clip((1920, 1080), True),
        keys=keys,
        transform=None,
        display=(640, 360),
        resolution=PORTRAIT,
        lead=0,
        frames=frames,
    )
    assert framing.window is not None and sum(map(len, framing.window.expressions)) < 4096


def test_a_reframe_too_long_for_per_frame_expressions_is_refused_fast() -> None:
    import time

    from aar.plan.render.errors import RenderError

    # The validator's case: 7200 keys that are nowhere straight, over 86400 frames.
    keys = [
        ReframeKey(time=12 * i, x=0.3 + 0.4 * ((i * 7919) % 101) / 100,
                   zoom=1.0 + 0.5 * ((i * 104729) % 97) / 96)
        for i in range(7200)
    ]  # fmt: skip
    started = time.monotonic()
    with pytest.raises(RenderError) as info:
        framing_for(
            _clip((1920, 1080), True),
            keys=keys,
            transform=None,
            display=(640, 360),
            resolution=PORTRAIT,
            lead=0,
            frames=86400,
        )
    assert info.value.code == "reframe_too_complex"
    assert info.value.details["location"] == "segments[0]"
    assert time.monotonic() - started < 2.0


def test_one_pass_reframe_values_are_reframe_at_for_every_frame() -> None:
    from aar.plan.render.framing import reframe_values

    keys = [
        ReframeKey(time=3, x=0.4, y=0.6, zoom=1.0),
        ReframeKey(time=10, x=0.55, y=0.5, zoom=1.4),
        ReframeKey(time=11, x=0.5, y=0.45, zoom=1.2),
        ReframeKey(time=40, x=0.6, y=0.55, zoom=1.9),
    ]
    for lead in (0, 5):
        values = reframe_values(keys, lead, 60)
        assert values == [reframe_at(keys, i - lead) for i in range(60)]
