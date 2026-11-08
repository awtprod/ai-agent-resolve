"""Machine-readable synthetic media for the hub render tests (S18b), and readers for its output.

Each source frame is a solid per-source colour (the colour band that names the source) with one
pattern block: a 6x6-cell square whose outer ring is black and whose inner 4x4 cells are the frame
index in binary (bit ``b`` at row ``b // 4``, column ``b % 4``: white for 1, grey for 0). Its sound
is a sine at a per-source frequency and level, optionally with silent spans (so a cut can sit in
quiet, as QA's join check wants).

The readers decode a render's frames one at a time from an ffmpeg pipe (constant memory): the
source a frame shows (its background colour), the frame index (the bits), and where the block's
ring edges land with sub-pixel precision (the 50% crossing between the background and the ring),
which locates the crop window. Audio is read as 48 kHz float PCM and a tone's amplitude measured by
projection onto that frequency.
"""

from __future__ import annotations

import math
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np

from aar.plan.profile import Profile

FFMPEG = "ffmpeg"
RATE = Fraction(24000, 1001)
SAMPLE_RATE = 48000

#: Background colour per test source (RGB), far enough apart to tell even blended.
COLOURS = {
    "red": (200, 60, 60),
    "green": (60, 180, 60),
    "blue": (60, 90, 200),
    "yellow": (200, 180, 60),
}


@dataclass(frozen=True)
class Spec:
    """One synthetic source."""

    colour: str
    tone_hz: float
    frames: int
    size: tuple[int, int] = (320, 180)
    #: Centre of the pattern block, in source pixels (whole numbers keep its edges on pixels).
    centre: tuple[int, int] | None = None
    cell: int = 6
    tone_dbfs: float = -20.0
    channels: int = 1
    #: Source frame spans [a, b) of silence (a cut's in point sits in one for QA's join check).
    quiet: tuple[tuple[int, int], ...] = ()
    #: Source frame spans [a, b) where the tone is louder by the given dB (a "laugh").
    bursts: tuple[tuple[int, int, float], ...] = ()
    rate: Fraction = RATE
    audio: bool = True

    @property
    def block(self) -> tuple[int, int, int, int]:
        """The pattern block's ring in source pixel-area coordinates: left, top, right, bottom."""
        cx, cy = self.centre or (self.size[0] // 2, self.size[1] // 2)
        half = 3 * self.cell
        return cx - half, cy - half, cx + half, cy + half


def pattern_frame(spec: Spec, index: int) -> np.ndarray:
    w, h = spec.size
    img = np.empty((h, w, 3), np.uint8)
    img[:] = COLOURS[spec.colour]
    left, top, right, bottom = spec.block
    img[top:bottom, left:right] = 0
    c = spec.cell
    for bit in range(16):
        row, col = divmod(bit, 4)
        y, x = top + (1 + row) * c, left + (1 + col) * c
        img[y : y + c, x : x + c] = 255 if (index >> bit) & 1 else 96
    return img


def _audio_expr(spec: Spec) -> str:
    amp = 10 ** (spec.tone_dbfs / 20)
    gate = "1"
    for a, b in spec.quiet:
        t0 = float(Fraction(a) / spec.rate)
        t1 = float(Fraction(b) / spec.rate)
        gate += f"*(1-between(t,{t0:.6f},{t1:.6f}))"
    for a, b, db in spec.bursts:
        t0 = float(Fraction(a) / spec.rate)
        t1 = float(Fraction(b) / spec.rate)
        gate += f"*(1+{10 ** (db / 20) - 1:.6f}*between(t,{t0:.6f},{t1:.6f}))"
    return f"{amp:.6f}*sin(2*PI*{spec.tone_hz}*t)*{gate}"


def make_media(path: Path, spec: Spec) -> Path:
    """Write ``spec`` as a small H.264 (lossless luma) + PCM file at ``path``."""
    w, h = spec.size
    seconds = float(Fraction(spec.frames) / spec.rate)
    cmd = [FFMPEG, "-hide_banner", "-nostdin", "-v", "error", "-y"]
    cmd += ["-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}"]
    cmd += ["-r", f"{spec.rate.numerator}/{spec.rate.denominator}", "-i", "pipe:"]
    if spec.audio:
        layout = "mono" if spec.channels == 1 else "stereo"
        expr = _audio_expr(spec)
        if spec.channels == 2:
            expr = f"{expr}|{expr}"
        cmd += [
            "-f",
            "lavfi",
            "-i",
            f"aevalsrc='{expr}':s={SAMPLE_RATE}:c={layout}:d={seconds:.6f}",
        ]
        cmd += ["-map", "0:v", "-map", "1:a", "-c:a", "pcm_s16le"]
    cmd += ["-c:v", "libx264", "-preset", "ultrafast", "-qp", "0", "-pix_fmt", "yuv420p"]
    cmd += ["-f", "mov", str(path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    try:
        for index in range(spec.frames):
            proc.stdin.write(pattern_frame(spec, index).tobytes())
    finally:
        proc.stdin.close()
    assert proc.wait(timeout=120) == 0, f"ffmpeg failed writing {path}"
    return path


# -- reading a render ----------------------------------------------------------------------------


def iter_frames(
    path: Path, size: tuple[int, int], picks: list[int]
) -> Iterator[tuple[int, np.ndarray]]:
    """The frames numbered ``picks`` (0-based), one at a time, as planar YUV 4:4:4 (shape
    ``(3, height, width)``): the Y plane is the encoded luma, untouched by chroma subsampling."""
    w, h = size
    wanted = sorted(set(picks))
    expr = "+".join(f"eq(n\\,{n})" for n in wanted)
    cmd = [FFMPEG, "-hide_banner", "-nostdin", "-v", "error", "-i", str(path)]
    cmd += ["-vf", f"select='{expr}'", "-fps_mode", "passthrough"]
    cmd += ["-f", "rawvideo", "-pix_fmt", "yuv444p", "pipe:"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    assert proc.stdout is not None
    frame_bytes = w * h * 3
    try:
        for n in wanted:
            data = proc.stdout.read(frame_bytes)
            assert len(data) == frame_bytes, f"frame {n} missing from {path}"
            yield n, np.frombuffer(data, np.uint8).reshape(3, h, w)
    finally:
        proc.stdout.close()
        proc.kill()
        proc.wait()


def frame_count(path: Path) -> int:
    out = subprocess.run(
        [
            *("ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0"),
            *("-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(path)),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return int(out.strip())


def to_yuv(rgb: tuple[float, float, float]) -> np.ndarray:
    """BT.601 limited-range Y, U, V of an RGB colour (ffmpeg's default for these files)."""
    r, g, b = rgb
    y = 16 + (65.481 * r + 128.553 * g + 24.966 * b) / 255
    u = 128 + (-37.797 * r - 74.203 * g + 112.0 * b) / 255
    v = 128 + (112.0 * r - 93.786 * g - 18.214 * b) / 255
    return np.array([y, u, v])


_PALETTE_YUV = {name: to_yuv(rgb) for name, rgb in COLOURS.items()}


def nearest_colour(yuv: np.ndarray) -> tuple[str, float]:
    """The palette colour nearest a YUV sample, and its distance."""
    best = min(_PALETTE_YUV, key=lambda name: float(np.linalg.norm(yuv - _PALETTE_YUV[name])))
    return best, float(np.linalg.norm(yuv - _PALETTE_YUV[best]))


def blend_of(yuv: np.ndarray, a: str, b: str) -> float | None:
    """How far along the line from colour ``a`` to ``b`` a sample lies (0 = a, 1 = b), or None
    when it is not near that line (more than 12 levels off it)."""
    pa, pb = _PALETTE_YUV[a], _PALETTE_YUV[b]
    d = pb - pa
    t = float((yuv - pa) @ d / (d @ d))
    off = float(np.linalg.norm(pa + t * d - yuv))
    return t if off <= 12 else None


@dataclass(frozen=True)
class Reading:
    """What one output frame shows."""

    colour: str
    colour_distance: float
    background: np.ndarray  # mean Y, U, V near the top-left corner
    index: int | None
    #: The ring's outer edges in output pixel-area coordinates: left, top, right, bottom.
    ring: tuple[float, float, float, float] | None


def _crossing(profile: np.ndarray, level: float, rising: bool) -> float | None:
    """Sub-pixel position (pixel-area coordinates) where ``profile`` crosses ``level``: the first
    fall below it (``rising`` False) or the last rise above it after a dark run."""
    below = profile < level
    idx = np.nonzero(below)[0]
    if not len(idx):
        return None
    if not rising:
        i = int(idx[0])
        if i == 0:
            return None
        a, b = profile[i - 1], profile[i]
        return (i - 1) + (a - level) / (a - b) + 0.5
    j = int(idx[-1])
    if j + 1 >= len(profile):
        return None
    a, b = profile[j], profile[j + 1]
    return j + (level - a) / (b - a) + 0.5


def read_frame(frame: np.ndarray, *, flip_h: bool = False, flip_v: bool = False) -> Reading:
    """Decode one YUV frame: background colour (sampled near the top-left corner), the bits and
    the ring edges (on the Y plane)."""
    lum = frame[0].astype(float)
    bg = frame[:, 2:6, 2:6].reshape(3, -1).mean(axis=1)
    colour, distance = nearest_colour(bg)
    black = 16.0
    level = (float(bg[0]) + black) / 2  # the 50% crossing between the background and the ring
    # The ring is black; a title box (black at 55%) over the background is not this dark.
    dark = lum < black + (float(bg[0]) - black) / 4
    rows = np.nonzero(dark.sum(axis=1) > 3)[0]
    cols = np.nonzero(dark.sum(axis=0) > 3)[0]
    if not len(rows) or not len(cols):
        return Reading(colour, distance, bg, None, None)
    top, bottom = int(rows[0]), int(rows[-1]) + 1
    left, right = int(cols[0]), int(cols[-1]) + 1
    cell_h, cell_w = (bottom - top) / 6, (right - left) / 6
    # Edges: along the middle of the ring's first row (left, right) and first column (top, bottom).
    row = lum[int(top + cell_h / 2), :]
    col = lum[:, int(left + cell_w / 2)]
    edges = (
        _crossing(row, level, rising=False),
        _crossing(col, level, rising=False),
        _crossing(row, level, rising=True),
        _crossing(col, level, rising=True),
    )
    ring = None if any(e is None for e in edges) else tuple(float(e) for e in edges)  # type: ignore[arg-type]
    white = 16 + 219 * 175 / 255  # between the grey (96) and white (255) cells, in Y
    index = 0
    for bit in range(16):
        r, c = divmod(bit, 4)
        if flip_h:
            c = 3 - c
        if flip_v:
            r = 3 - r
        y = int(top + (1.5 + r) * cell_h)
        x = int(left + (1.5 + c) * cell_w)
        if lum[y, x] > white:
            index |= 1 << bit
    return Reading(colour, distance, bg, index, ring)  # type: ignore[arg-type]


def window_from_ring(
    ring: tuple[float, float, float, float], spec: Spec, out_size: tuple[int, int]
) -> tuple[float, float, float, float]:
    """The source window (centre x, centre y, width, height in source px) that puts the block's
    ring at the measured output edges."""
    left, top, right, bottom = ring
    s_left, s_top, s_right, s_bottom = spec.block
    sx = (right - left) / (s_right - s_left)
    sy = (bottom - top) / (s_bottom - s_top)
    w, h = out_size[0] / sx, out_size[1] / sy
    x0, y0 = s_left - left / sx, s_top - top / sy
    return x0 + w / 2, y0 + h / 2, w, h


# -- audio ----------------------------------------------------------------------------------------


def read_pcm(path: Path) -> np.ndarray:
    """The first audio stream as 48 kHz float samples, shape (samples, channels)."""
    data = subprocess.run(
        [
            *(FFMPEG, "-hide_banner", "-nostdin", "-v", "error", "-i", str(path), "-map", "0:a:0"),
            *("-ar", str(SAMPLE_RATE), "-f", "f32le", "-acodec", "pcm_f32le", "pipe:"),
        ],
        capture_output=True,
        check=True,
    ).stdout
    channels = int(
        subprocess.run(
            [
                *("ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries"),
                *("stream=channels", "-of", "csv=p=0", str(path)),
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )
    return np.frombuffer(data, np.float32).reshape(-1, channels)


def tone_dbfs(pcm: np.ndarray, start: int, end: int, freq: float) -> float:
    """The amplitude (dBFS, peak) of a sine at ``freq`` in samples [start, end) of channel 0."""
    x = pcm[start:end, 0].astype(float)
    t = np.arange(start, end) / SAMPLE_RATE
    c = np.cos(2 * math.pi * freq * t)
    s = np.sin(2 * math.pi * freq * t)
    amp = 2 * math.hypot(float(x @ c), float(x @ s)) / max(1, len(x))
    return 20 * math.log10(max(amp, 1e-12))


def sample_at(frame: int, rate: Fraction = RATE) -> int:
    return round(Fraction(frame) / rate * SAMPLE_RATE)


def fast(profile: Profile) -> Profile:
    """The profile with a quick x264 preset (the encoder's speed setting is not under test)."""
    settings = profile.render.model_copy(
        update={"review_x264_preset": "veryfast", "master_x264_preset": "veryfast"}
    )
    return profile.model_copy(update={"render": settings})
