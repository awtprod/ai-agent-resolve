"""Transform 5, framing: punch-ins on jump cuts, and the steady camera path for shorts.

**Jump cuts.** A cut between two pieces of the same camera (``no-same-camera-jump``'s test)
steps to the next framing level, so the zoom changes across it: a long-form toggles its static
``Segment.transform`` zoom between 1.0 and ``punch_in_zoom`` (CICGTD ``edl_v7.py``); a short
cycles through the profile's ``framing`` presets (the supoclip planner's A/B). Any other join
returns to the first level.

**Steady path** (shorts, ``steady()`` in the supoclip planner): the face track's largest face,
interpolated to every source frame, is box-smoothed; the camera holds while the face sways
inside a dead zone of ``dead_zone`` crop widths and follows only when it would leave; the held
path glides through a Gaussian of ``glide_sigma_s``; then up to ``safety_passes`` times, where
the face would come within ``safety_margin`` of the crop edge, the path eases over early (the
shortfall's max within ``safety_window_s``, smoothed by a ``safety_sigma_s`` Gaussian). New
here: the path is clamped so the crop stays in the picture, then slew-limited to
``speed_headroom`` times ``no-whip-pan``'s ``max_pan_speed``, forward then backward, so no two
frames (and so no two keyframes) move faster. Keyframes every ``key_step_s`` plus the last
frame carry the path; zoom and height are the preset's, static per segment, so the zoom speed
is 0. Units: x is 0-1 of the source width, as ``ReframeKey.x``.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence

from aar.plan.model import ReframeKey, Source
from aar.plan.produce.timeline import Piece, is_cut, same_camera
from aar.plan.profile import FramingPreset, SteadySettings


def framing_levels(pieces: Sequence[Piece], sources: dict[str, Source], levels: int) -> list[int]:
    """The framing level of each piece: the next level across a same-camera cut, else 0."""
    out: list[int] = []
    for index, piece in enumerate(pieces):
        level = 0
        if index:
            prev = pieces[index - 1]
            if is_cut(prev, piece) and same_camera(sources[prev.source], sources[piece.source]):
                level = (out[-1] + 1) % levels
        out.append(level)
    return out


def _box(values: Sequence[float], width: int) -> list[float]:
    """Moving average over ``width`` samples with edge padding (``np.convolve`` 'valid')."""
    if width <= 1:
        return list(values)
    left, right = (width - 1) // 2, width // 2
    padded = [values[0]] * left + list(values) + [values[-1]] * right
    return [sum(padded[i : i + width]) / width for i in range(len(values))]


def gaussian(values: Sequence[float], sigma: float) -> list[float]:
    """Gaussian smoothing, kernel ±3 sigma, edge padded (as the ported ``np.convolve``)."""
    reach = int(3 * sigma)
    if reach < 1 or not values:
        return list(values)
    kernel = [math.exp(-0.5 * (i / sigma) ** 2) for i in range(-reach, reach + 1)]
    total = sum(kernel)
    kernel = [k / total for k in kernel]
    padded = [values[0]] * reach + list(values) + [values[-1]] * reach
    return [sum(k * padded[i + j] for j, k in enumerate(kernel)) for i in range(len(values))]


def steady(x: Sequence[float], width: float, fps: float, s: SteadySettings) -> list[float]:
    """The camera-operator path for a face centre ``x`` (one value per frame).

    ``width`` is the crop width in the same units as ``x``.
    """
    if not x:
        return []
    face = _box(x, max(1, round(s.pre_smooth_s * fps)))
    dead = s.dead_zone * width
    start = face[: max(1, round(s.start_median_s * fps))]
    cur = float(statistics.median(start))
    held: list[float] = []
    for value in face:
        if value - cur > dead:
            cur = value - dead
        elif cur - value > dead:
            cur = value + dead
        held.append(cur)
    path = gaussian(held, s.glide_sigma_s * fps)
    limit = max(0.0, width / 2 - s.safety_margin)
    half = int(s.safety_window_s * fps)
    for _ in range(s.safety_passes):
        need = []
        for f, c in zip(face, path, strict=True):
            d = f - c
            need.append(d - limit if d > limit else (d + limit if d < -limit else 0.0))
        if not any(need):
            break
        corr = []
        for i in range(len(need)):
            window = need[max(0, i - half) : i + half + 1]
            corr.append(max(max(window), 0.0) + min(min(window), 0.0))
        path = [c + e for c, e in zip(path, gaussian(corr, s.safety_sigma_s * fps), strict=True)]
    return path


def clamp_inside(path: Sequence[float], width: float) -> list[float]:
    """Keep a crop of ``width`` inside [0, 1]; a crop wider than the picture centres."""
    if width >= 1:
        return [0.5] * len(path)
    lo, hi = width / 2, 1 - width / 2
    return [min(hi, max(lo, value)) for value in path]


def slew_limit(path: Sequence[float], step: float) -> list[float]:
    """Limit the change between neighbouring frames to ``step``, forward then backward."""
    out = list(path)
    for i in range(1, len(out)):
        out[i] = min(out[i - 1] + step, max(out[i - 1] - step, out[i]))
    for i in range(len(out) - 2, -1, -1):
        out[i] = min(out[i + 1] + step, max(out[i + 1] - step, out[i]))
    return out


def crop_width(output_aspect: float, source_aspect: float, zoom: float) -> float:
    """The crop width as a share of the source width, at full height divided by ``zoom``."""
    return output_aspect / source_aspect / zoom


def face_x(samples: Sequence[tuple[float, float]], times: Sequence[float]) -> list[float] | None:
    """The face centre at each time, linearly interpolated, held at the ends; ``None`` when
    the track has no face at all."""
    if not samples:
        return None
    out: list[float] = []
    j = 0
    for t in times:
        while j + 1 < len(samples) and samples[j + 1][0] <= t:
            j += 1
        t0, x0 = samples[j]
        if t <= t0 or j + 1 >= len(samples):
            out.append(x0)
            continue
        t1, x1 = samples[j + 1]
        out.append(x0 + (x1 - x0) * (t - t0) / (t1 - t0))
    return out


def reframe_keys(
    face: Sequence[float] | None,
    preset: FramingPreset,
    width: float,
    fps: float,
    s: SteadySettings,
    *,
    max_pan_speed: float,
) -> tuple[list[ReframeKey], float]:
    """Keyframes for one segment of ``len(face)`` frames, and the face's least distance from
    the crop edge (as a share of the source width; ``inf`` without a face).

    Without a face the frame holds the centre: one key.
    """
    if not face:
        return [ReframeKey(time=0, x=0.5, y=preset.y, zoom=preset.zoom)], math.inf
    path = steady(face, width, fps, s)
    path = slew_limit(clamp_inside(path, width), max_pan_speed * s.speed_headroom / fps)
    margin = min(width / 2 - abs(f - c) for f, c in zip(face, path, strict=True))
    step = max(1, round(s.key_step_s * fps))
    times = list(range(0, len(path), step))
    if times[-1] != len(path) - 1:
        times.append(len(path) - 1)
    keys = [ReframeKey(time=t, x=round(path[t], 4), y=preset.y, zoom=preset.zoom) for t in times]
    return keys, margin
