"""How each clip's picture is framed: the S18 crop window per frame, fed to ffmpeg (S18b).

A clip the export conforms with ``fill`` (its aspect differs from the timeline's, or it has a
reframe or transform) shows a window of its source: S18's own
:func:`aar.plan.export.verify.window_from_plan` gives its centre and size for the reframe values
at that frame (``x``, ``y`` and ``zoom`` linear between keyframes, held before the first and after
the last, as the plan says) and the static transform. The window's corners follow from S18's
transform maths (a source point ``p`` lands at ``R(rot) * s * F * (p - c) + (pan, tilt)``), so a
rotation or flip is a rotated or mirrored window.

ffmpeg gets the window as the four corners of a ``perspective`` filter (``sense=source``: those
source points go to the output's corners), one expression per coordinate in the filter's input
frame count, piecewise over the keyframes. Every corner coordinate is ``A * x + B / zoom + C``
(``y`` for a vertical one), so the expression is exact at every frame, not only at keys. The
corners carry a half-pixel correction (``perspective`` maps pixel indices, the window is in pixel
areas). Before it, the source is padded with black when a window leaves the picture (Resolve shows
black there) and cropped to the box all windows of the clip fit in; after it, ``scale`` makes the
timeline size. A clip that is not conformed is scaled to the frame when its aspect is the
timeline's, and fitted (letterboxed, Resolve's default) when it is not.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Literal

from aar.plan.export.layout import ClipSpec
from aar.plan.export.verify import window_from_plan
from aar.plan.model import ReframeKey, Resolution, Transform
from aar.plan.render.errors import RenderError

FramingMode = Literal["scale", "fit", "window"]
Point = tuple[float, float]
#: A window in source pixels: centre x, centre y, width, height (S18's ``Window``).
Window = tuple[float, float, float, float]

#: Interpolation of the window's picture (``perspective``) and of the scale to the frame.
PERSPECTIVE_INTERPOLATION = "cubic"
#: The affine form of a corner is checked against a direct evaluation at every key to this.
AFFINE_CHECK_PX = 1e-6
#: Margin (source px) kept around the windows when cropping to their box, for the interpolation.
BOX_MARGIN = 2


def reframe_at(keys: list[ReframeKey], t: int) -> tuple[float, float, float]:
    """The reframe ``(x, y, zoom)`` at frame ``t`` of a segment: linear between keyframes, held
    before the first and after the last (frames played from the handles included)."""
    if not keys:
        return 0.5, 0.5, 1.0
    if t <= keys[0].time:
        return keys[0].x, keys[0].y, keys[0].zoom
    for a, b in itertools.pairwise(keys):
        if t < b.time:
            f = (t - a.time) / (b.time - a.time)
            return a.x + (b.x - a.x) * f, a.y + (b.y - a.y) * f, a.zoom + (b.zoom - a.zoom) * f
    last = keys[-1]
    return last.x, last.y, last.zoom


@dataclass(frozen=True)
class Geometry:
    """What the window maths needs for one clip: S18's inputs plus the hub file's size."""

    transform: Transform
    frame_size: tuple[int, int]  # the plan's Source.frame_size (S18's source pixels)
    media: tuple[int, int]  # the hub file's display size (its real pixels)
    resolution: Resolution

    @property
    def to_media(self) -> tuple[float, float]:
        return self.media[0] / self.frame_size[0], self.media[1] / self.frame_size[1]

    def window(self, x: float, y: float, zoom: float) -> Window:
        """S18's window for these reframe values, in the hub file's pixels."""
        res = self.resolution
        cx, cy, w, h = window_from_plan(
            x, y, zoom, self.transform, self.frame_size, res.width, res.height
        )
        mx, my = self.to_media
        return cx * mx, cy * my, w * mx, h * my

    def corners(self, x: float, y: float, zoom: float) -> tuple[Point, Point, Point, Point]:
        """The source points (hub pixels, pixel-area coordinates) shown at the output's top-left,
        top-right, bottom-left and bottom-right corners."""
        cu, cv, ww, wh = self.window(x, y, zoom)
        res = self.resolution
        tf = self.transform
        fx = -1.0 if tf.flip_h else 1.0
        fy = -1.0 if tf.flip_v else 1.0
        kx, ky = ww / res.width, wh / res.height  # source px per timeline px
        angle = math.radians(tf.rotation_deg)
        cos, sin = math.cos(angle), math.sin(angle)
        half_w, half_h = res.width / 2.0, res.height / 2.0
        out: list[Point] = []
        for qx, qy in ((-half_w, half_h), (half_w, half_h), (-half_w, -half_h), (half_w, -half_h)):
            # q = R s F (p - c) + t, so p - p0 = F^-1 s^-1 R(-rot) q (y up; the image's v is down).
            rx = cos * qx + sin * qy
            ry = -sin * qx + cos * qy
            out.append((cu + kx / fx * rx, cv - ky / fy * ry))
        return out[0], out[1], out[2], out[3]


@dataclass(frozen=True)
class WindowPlan:
    """The ffmpeg framing of one windowed clip: pad, box crop, perspective corners."""

    pad: int  # black margin added on every side (hub px), 0 for none
    box: tuple[int, int, int, int]  # x, y, width, height of the crop, in padded px
    expressions: tuple[str, ...]  # x0, y0, x1, y1, x2, y2, x3, y3 for perspective
    moving: bool  # per-frame evaluation; False: constant corners
    first: Window  # the window at the clip's first and last rendered frames (for the report)
    last: Window


@dataclass(frozen=True)
class Framing:
    """How one clip's picture becomes a timeline frame."""

    mode: FramingMode
    #: The source picture's display size and, for ``fit``, the fitted size inside the frame.
    display: tuple[int, int]
    fitted: tuple[int, int] | None = None
    window: WindowPlan | None = None
    geometry: Geometry | None = None
    keys: tuple[ReframeKey, ...] = ()
    lead: int = 0  # rendered frames before the segment's own first frame (a head handle)

    def at(self, frame: int) -> Window | None:
        """The S18 window at rendered frame ``frame`` of the clip (None when not windowed)."""
        if self.geometry is None:
            return None
        return self.geometry.window(*reframe_at(list(self.keys), frame - self.lead))


def _num(value: float) -> str:
    """A number as ffmpeg's expression parser reads it: fixed point, no exponent, no suffix."""
    text = f"{value:.9f}".rstrip("0").rstrip(".")
    return "0" if text in ("", "-0") else text


def _piecewise(points: list[tuple[int, float]], t: str) -> str:
    """Linear between ``(time, value)`` points, held outside them, as an ffmpeg expression in
    ``t``. The pieces are chosen by a balanced tree of ``if(lt(...))`` tests, so the nesting
    grows with the log of the key count (a long reframe has thousands of keys)."""
    if all(v == points[0][1] for _, v in points):
        return _num(points[0][1])

    def piece(k: int) -> str:
        (t0, v0), (t1, v1) = points[k], points[k + 1]
        return f"{_num(v0)}+({_num(v1 - v0)})*({t}-({t0}))/{t1 - t0}"

    def tree(lo: int, hi: int) -> str:  # pieces lo .. hi - 1
        if hi - lo == 1:
            return piece(lo)
        mid = (lo + hi) // 2
        return f"if(lt({t},{points[mid][0]}),{tree(lo, mid)},{tree(mid, hi)})"

    first, last = points[0], points[-1]
    inner = tree(0, len(points) - 1)
    return f"if(lt({t},{first[0]}),{_num(first[1])},if(lt({t},{last[0]}),{inner},{_num(last[1])}))"


#: Collinear keyframes are merged when the middle one sits on its neighbours' line to this.
COLLINEAR_TOLERANCE = 1e-12
#: The most bytes of corner expressions one moving window may take: perspective parses them again
#: at every frame, so a path past this is refused (``reframe_too_complex``).
MAX_EXPRESSION_BYTES = 256 * 1024


def simplify_keys(keys: list[ReframeKey]) -> list[ReframeKey]:
    """The keys without those that lie on the straight line between their neighbours in ``x``,
    ``y`` and ``zoom`` (within :data:`COLLINEAR_TOLERANCE`): the same path, interpolated
    linearly, with fewer pieces. The first and last keys always stay."""
    if len(keys) < 3:
        return list(keys)
    kept = [keys[0]]
    for middle, after in itertools.pairwise(keys[1:]):
        before = kept[-1]
        f = (middle.time - before.time) / (after.time - before.time)
        on_line = all(
            abs(getattr(before, name) + (getattr(after, name) - getattr(before, name)) * f
                - getattr(middle, name)) <= COLLINEAR_TOLERANCE
            for name in ("x", "y", "zoom")
        )  # fmt: skip
        if not on_line:
            kept.append(middle)
    kept.append(keys[-1])
    return kept


def reframe_values(
    keys: list[ReframeKey], lead: int, frames: int
) -> list[tuple[float, float, float]]:
    """:func:`reframe_at` for rendered frames ``0 .. frames - 1`` (segment frame ``i - lead``),
    in one pass over the keys."""
    if not keys:
        return [(0.5, 0.5, 1.0)] * frames
    out: list[tuple[float, float, float]] = []
    k = 0
    for i in range(frames):
        t = i - lead
        while k + 1 < len(keys) and keys[k + 1].time <= t:
            k += 1
        a = keys[k]
        if t <= keys[0].time or k + 1 >= len(keys):
            out.append((a.x, a.y, a.zoom))
            continue
        b = keys[k + 1]
        f = (t - a.time) / (b.time - a.time)
        out.append((a.x + (b.x - a.x) * f, a.y + (b.y - a.y) * f, a.zoom + (b.zoom - a.zoom) * f))
    return out


def _window_plan(
    geo: Geometry, keys: list[ReframeKey], lead: int, frames: int, location: str
) -> WindowPlan:
    keys = simplify_keys(keys) or [ReframeKey(time=0)]
    t = f"(in-1-{lead})" if lead else "(in-1)"
    xs = _piecewise([(k.time, k.x) for k in keys], t)
    ys = _piecewise([(k.time, k.y) for k in keys], t)
    zs = _piecewise([(k.time, k.zoom) for k in keys], t)
    size = 4 * (len(xs) + len(ys) + 2 * len(zs)) + 400
    if size > MAX_EXPRESSION_BYTES:
        raise RenderError(
            "reframe_too_complex",
            f"{location}: its reframe needs about {size // 1024} KiB of per-frame corner "
            f"expressions ({len(keys)} keyframes after merging straight runs), over the "
            f"{MAX_EXPRESSION_BYTES // 1024} KiB limit",
            "use fewer keyframes (aar plan produce's key_step_s), or split the segment",
            location=location,
            keyframes=len(keys),
            bytes=size,
        )
    values = reframe_values(keys, lead, frames)
    moving = len(set(values)) > 1
    sample = values if moving else values[:1]
    points = [p for v in sample for p in geo.corners(*v)]
    us = [p[0] for p in points]
    vs = [p[1] for p in points]
    mw, mh = geo.media
    overflow = max(0.0, -min(us), -min(vs), max(us) - mw, max(vs) - mh)
    pad = math.ceil(overflow) + BOX_MARGIN if overflow > 0 else 0
    left = max(0, math.floor(min(us) + pad) - BOX_MARGIN)
    top = max(0, math.floor(min(vs) + pad) - BOX_MARGIN)
    right = min(mw + 2 * pad, math.ceil(max(us) + pad) + BOX_MARGIN)
    bottom = min(mh + 2 * pad, math.ceil(max(vs) + pad) + BOX_MARGIN)
    bw, bh = max(2, right - left), max(2, bottom - top)

    def final(x: float, y: float, zoom: float) -> list[Point]:
        """The four corners as perspective reads them: in the box, and moved by half an output
        pixel (perspective maps pixel indices; the window is in pixel areas)."""
        tl, tr, bl, br = geo.corners(x, y, zoom)
        ax, ay = (tr[0] - tl[0]) / bw, (tr[1] - tl[1]) / bw
        bx, by = (bl[0] - tl[0]) / bh, (bl[1] - tl[1]) / bh
        dx = pad - left + 0.5 * (ax + bx) - 0.5
        dy = pad - top + 0.5 * (ay + by) - 0.5
        return [(u + dx, v + dy) for u, v in (tl, tr, bl, br)]

    # Each coordinate is A * x + B / zoom + C (y for the vertical ones): read A, B, C off three
    # evaluations, then check the form at every key.
    base, unit, half = final(0.0, 0.0, 1.0), final(1.0, 1.0, 1.0), final(0.0, 0.0, 0.5)
    coeffs: list[tuple[float, float, float]] = []
    for corner in range(4):
        for axis in range(2):
            b = half[corner][axis] - base[corner][axis]
            a = unit[corner][axis] - base[corner][axis]
            coeffs.append((a, b, base[corner][axis] - b))
    for key in keys:
        direct = final(key.x, key.y, key.zoom)
        for index, (a, b, c) in enumerate(coeffs):
            share = key.x if index % 2 == 0 else key.y
            got = a * share + b / key.zoom + c
            if abs(got - direct[index // 2][index % 2]) > AFFINE_CHECK_PX:
                raise AssertionError(f"corner {index} is not affine in x and 1/zoom")
    exprs: list[str] = []
    for index, (a, b, c) in enumerate(coeffs):
        share = xs if index % 2 == 0 else ys
        if not moving:
            x0, y0, z0 = values[0]
            exprs.append(_num(a * (x0 if index % 2 == 0 else y0) + b / z0 + c))
        else:
            exprs.append(f"{_num(a)}*({share})+{_num(b)}/({zs})+{_num(c)}")
    return WindowPlan(
        pad,
        (left, top, bw, bh),
        tuple(exprs),
        moving,
        geo.window(*values[0]),
        geo.window(*values[-1]),
    )


def framing_for(
    clip: ClipSpec,
    *,
    keys: list[ReframeKey],
    transform: Transform | None,
    display: tuple[int, int],
    resolution: Resolution,
    lead: int,
    frames: int,
) -> Framing:
    """The framing of one layout clip that renders ``frames`` frames, ``lead`` of them before
    its plan range (S18's conform decides: ``fill`` is a window)."""
    if clip.conform and clip.frame_size is not None:
        geo = Geometry(transform or Transform(), clip.frame_size, display, resolution)
        plan = _window_plan(geo, keys, lead, frames, clip.location)
        return Framing("window", display, None, plan, geo, tuple(keys), lead)
    width, height = resolution.width, resolution.height
    if display[0] * height == display[1] * width:
        return Framing("scale", display)
    fit = min(width / display[0], height / display[1])
    fitted = (
        _centred(display[0] * fit, width),
        _centred(display[1] * fit, height),
    )
    if abs(fitted[0] - width) <= 1 and abs(fitted[1] - height) <= 1:
        return Framing("scale", display)
    return Framing("fit", display, fitted)


def _centred(exact: float, frame: int) -> int:
    """The fitted size nearest ``exact`` that leaves an even margin, so the picture sits exactly
    in the middle of the frame (the padding is the same on both sides)."""
    options = [n for n in range(int(exact) - 1, int(exact) + 3) if 2 <= n <= frame]
    return min((n for n in options if (frame - n) % 2 == 0), key=lambda n: (abs(n - exact), n))


def picture_filters(framing: Framing, resolution: Resolution) -> list[str]:
    """The filters that turn the clip's decoded (display-size) picture into a timeline frame."""
    width, height = resolution.width, resolution.height
    if framing.mode == "scale":
        return [f"scale={width}:{height}"]
    if framing.mode == "fit":
        assert framing.fitted is not None
        fw, fh = framing.fitted
        # In 4:4:4, so the pad's left edge is not rounded to an even pixel for the chroma.
        return [
            "format=yuv444p",
            f"scale={fw}:{fh}",
            f"pad={width}:{height}:{(width - fw) // 2}:{(height - fh) // 2}:black",
        ]
    plan = framing.window
    assert plan is not None
    filters = ["format=yuv444p"]
    if plan.pad:
        filters.append(f"pad=iw+{2 * plan.pad}:ih+{2 * plan.pad}:{plan.pad}:{plan.pad}:black")
    x, y, w, h = plan.box
    filters.append(f"crop={w}:{h}:{x}:{y}:exact=1")
    names = ("x0", "y0", "x1", "y1", "x2", "y2", "x3", "y3")
    corners = ":".join(
        f"{name}='{expr}'" for name, expr in zip(names, plan.expressions, strict=True)
    )
    filters.append(
        f"perspective={corners}:interpolation={PERSPECTIVE_INTERPOLATION}:sense=source"
        f":eval={'frame' if plan.moving else 'init'}"
    )
    filters.append(f"scale={width}:{height}")
    return filters
