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
    assert picture_filters(fit, PORTRAIT) == ["scale=1080:608", "pad=1080:1920:0:656:black"]
