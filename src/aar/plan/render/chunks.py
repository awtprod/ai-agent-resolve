"""How a render fits in memory (S18b round 3): the budget, the cost model and the chunk plan.

Every picture use is its own ffmpeg input, and an open input holds decoded frames whether or not
the graph is reading it yet (the validator's round-2 measurement: 1,177 MB plus 44.7 MB per
1080p picture use, r = 0.99999). So the picture is encoded in **chunks**, one ffmpeg after
another, each opening only its own pictures, and the chunks are joined by stream copy: an
encode's memory is its base plus its chunk's uses, whatever the timeline's length.

* The **budget** is ``min(6 GiB, half of MemAvailable, the free headroom of every cgroup above the
  process that sets memory.max)``, read when the render starts; ``--memory-budget`` (or ``[render]
  memory_budget_mb``) replaces it.
* The **cost model** is the measurement, rounded up: a base of 300 MB plus 1,000 MB at a 1080p
  output (scaled by the output's pixels), and per picture use 50 MB, or 80 MB when it is windowed
  (a crop, punch-in or moving reframe), at a 1080p 8-bit 4:2:0 source (scaled by the source's
  decoded bytes per frame), plus 2 MB. The mix pass: 200 MB plus 5 MB per sound.
* A chunk is cut only at a **hard cut**: a record frame no picture crosses on any track. A
  dissolve's window, a head fade and a moving reframe all lie inside a picture (the outgoing
  picture plays through the window), so none is ever split. Titles are static text and may be
  split. When the pictures between two such frames alone are over the budget, the render is
  refused (``render_too_large``), naming that window.
* The same plan keeps each chunk's filter script under the script limit (moving reframes).
"""

from __future__ import annotations

import contextlib
import itertools
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aar.plan.model import Resolution
from aar.plan.render.errors import RenderError
from aar.plan.render.media import HubSource
from aar.plan.render.timeline import Picture, Timeline

BUDGET_CAP_MB = 6 * 1024
MEM_AVAILABLE_SHARE = 0.5
MB = 1024 * 1024

REFERENCE_PIXELS = 1920 * 1080
REFERENCE_BYTES_PER_PIXEL = 1.5  # 8-bit 4:2:0
ENCODE_FIXED_MB = 300.0
ENCODE_SCALED_MB = 1000.0  # at a 1920x1080 output
PLAIN_USE_MB = 50.0
WINDOW_USE_MB = 80.0
USE_FIXED_MB = 2.0
MIX_FIXED_MB = 200.0
SOUND_USE_MB = 5.0
#: A picture's share of its chunk's filter script beyond its framing expressions.
PICTURE_SCRIPT_BYTES = 1024


@dataclass(frozen=True)
class Budget:
    """The memory a render may use at once, and where the number came from."""

    mb: float
    source: str  # "override" or "default"
    limits: dict[str, float]  # each bound the default took the smallest of

    def as_dict(self) -> dict[str, Any]:
        return {
            "mb": round(self.mb),
            "source": self.source,
            "limits_mb": {k: round(v) for k, v in self.limits.items()},
        }


def _read(path: Path) -> str | None:
    with contextlib.suppress(OSError, UnicodeDecodeError):
        return path.read_text(encoding="ascii")
    return None


def mem_available_mb(meminfo: Path = Path("/proc/meminfo")) -> float | None:
    text = _read(meminfo)
    for line in (text or "").splitlines():
        if line.startswith("MemAvailable:"):
            fields = line.split()
            if len(fields) >= 2 and fields[1].isdigit():
                return int(fields[1]) / 1024
    return None


def cgroup_headroom_mb(
    proc_cgroup: Path = Path("/proc/self/cgroup"), root: Path = Path("/sys/fs/cgroup")
) -> float | None:
    """The least ``memory.max - memory.current`` over the process's cgroup (v2) and every cgroup
    above it that sets ``memory.max``; None when none does (or there is no cgroup v2)."""
    text = _read(proc_cgroup)
    if text is None:
        return None
    group = next((line[3:] for line in text.splitlines() if line.startswith("0::")), None)
    if group is None:
        return None
    parts = [p for p in group.strip().split("/") if p and p not in (".", "..")]
    headroom: float | None = None
    for depth in range(len(parts), -1, -1):
        folder = root.joinpath(*parts[:depth])
        limit, current = _read(folder / "memory.max"), _read(folder / "memory.current")
        if limit is None or current is None:
            continue
        limit, current = limit.strip(), current.strip()
        if not limit.isdigit() or not current.isdigit():
            continue  # "max": no limit here
        free = max(0, int(limit) - int(current)) / MB
        headroom = free if headroom is None else min(headroom, free)
    return headroom


def memory_budget(
    override_mb: float | None = None,
    *,
    meminfo: Path = Path("/proc/meminfo"),
    proc_cgroup: Path = Path("/proc/self/cgroup"),
    cgroup_root: Path = Path("/sys/fs/cgroup"),
) -> Budget:
    """The override when one is given, else the module docstring's default."""
    if override_mb is not None:
        return Budget(float(override_mb), "override", {})
    limits: dict[str, float] = {"cap": float(BUDGET_CAP_MB)}
    available = mem_available_mb(meminfo)
    if available is not None:
        limits["mem_available_share"] = available * MEM_AVAILABLE_SHARE
    headroom = cgroup_headroom_mb(proc_cgroup, cgroup_root)
    if headroom is not None:
        limits["cgroup_headroom"] = headroom
    return Budget(min(limits.values()), "default", limits)


def _bytes_per_pixel(pix_fmt: str | None) -> float:
    fmt = pix_fmt or "yuv420p"
    bpp = 3.0 if ("444" in fmt or "rgb" in fmt or "gbr" in fmt) else 2.0 if "422" in fmt else 1.5
    return bpp * (2 if any(depth in fmt for depth in ("p10", "p12", "p16")) else 1)


def encode_base_mb(res: Resolution) -> float:
    return ENCODE_FIXED_MB + ENCODE_SCALED_MB * res.width * res.height / REFERENCE_PIXELS


def picture_use_mb(pic: Picture, media: HubSource) -> float:
    """One picture input's memory: its decoded frames, in the source's coded size and format."""
    video = media.info.video
    width = video.width if video is not None and video.width else media.display[0]
    height = video.height if video is not None and video.height else media.display[1]
    scale = (width * height * _bytes_per_pixel(video and video.pix_fmt)) / (
        REFERENCE_PIXELS * REFERENCE_BYTES_PER_PIXEL
    )
    per = WINDOW_USE_MB if pic.framing.mode == "window" else PLAIN_USE_MB
    return USE_FIXED_MB + per * scale


def picture_script_bytes(pic: Picture) -> int:
    window = pic.framing.window
    exprs = sum(len(e) for e in window.expressions) if window is not None else 0
    return PICTURE_SCRIPT_BYTES + exprs


@dataclass(frozen=True)
class Chunk:
    """Record frames ``[start, end)`` of the picture, encoded by one ffmpeg."""

    start: int
    end: int
    pictures: int
    planned_mb: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "pictures": self.pictures,
            "planned_mb": round(self.planned_mb),
        }


def cut_points(tl: Timeline) -> list[int]:
    """The record frames a chunk may start or end at: the timeline's ends, and every picture's
    start or end that no picture (on any track) plays across."""
    candidates = sorted(
        {0, tl.total_frames} | {p.start for p in tl.pictures} | {p.end for p in tl.pictures}
    )
    spans = sorted((p.start, p.end) for p in tl.pictures)
    allowed: list[int] = []
    reach = -1  # the furthest end of the pictures that start before the candidate
    i = 0
    for frame in candidates:
        while i < len(spans) and spans[i][0] < frame:
            reach = max(reach, spans[i][1])
            i += 1
        if reach <= frame and 0 <= frame <= tl.total_frames:
            allowed.append(frame)
    return allowed


def _timecode(frame: int, fps_num: int, fps_den: int) -> str:
    seconds = frame * fps_den / fps_num
    return f"{int(seconds // 3600):d}:{int(seconds // 60 % 60):02d}:{seconds % 60:06.3f}"


def plan_chunks(
    tl: Timeline,
    hub: dict[str, HubSource],
    budget: Budget,
    *,
    max_script_bytes: int,
) -> list[Chunk]:
    """The chunks, each as long as the budget allows, cut only at :func:`cut_points`; or
    ``render_too_large`` when the pictures between two cut points alone are over the budget."""
    fps = tl.plan.fps
    base = encode_base_mb(tl.plan.resolution)
    points = cut_points(tl)
    by_start = sorted(tl.pictures, key=lambda p: p.start)
    blocks: list[tuple[int, int, int, float, int]] = []  # start, end, pictures, MB, script bytes
    i = 0
    for start, end in itertools.pairwise(points):
        count, cost, script = 0, 0.0, 0
        while i < len(by_start) and by_start[i].start < end:
            pic = by_start[i]
            count += 1
            cost += picture_use_mb(pic, hub[pic.source_id])
            script += picture_script_bytes(pic)
            i += 1
        blocks.append((start, end, count, cost, script))
    chunks: list[Chunk] = []
    cur_start, cur_count, cur_cost, cur_script = 0, 0, 0.0, 0
    for start, end, count, cost, script in blocks:
        if base + cost > budget.mb:
            window = (
                f"record frames {start}..{end} ({_timecode(start, fps.num, fps.den)} to "
                f"{_timecode(end, fps.num, fps.den)})"
            )
            raise RenderError(
                "render_too_large",
                f"{window} hold {count} overlapping picture(s) with no hard cut between them; "
                f"encoding them at once needs about {base + cost:.0f} MB, over the memory "
                f"budget of {budget.mb:.0f} MB",
                "render smaller hub copies (proxies), put a hard cut in that stretch, or give "
                "--memory-budget on a machine with the memory",
                start=start,
                end=end,
                pictures=count,
                estimated_mb=round(base + cost),
                budget_mb=round(budget.mb),
            )
        if script > max_script_bytes:
            raise RenderError(
                "reframe_too_complex",
                f"record frames {start}..{end} need a {script // 1024} KiB filter script, over "
                f"the {max_script_bytes // 1024} KiB limit (moving reframes take most of it)",
                "use fewer reframe keyframes, or split the segment",
                start=start,
                end=end,
                bytes=script,
            )
        over = base + cur_cost + cost > budget.mb or cur_script + script > max_script_bytes
        if over and start > cur_start:
            chunks.append(Chunk(cur_start, start, cur_count, base + cur_cost))
            cur_start, cur_count, cur_cost, cur_script = start, 0, 0.0, 0
        cur_count += count
        cur_cost += cost
        cur_script += script
    chunks.append(Chunk(cur_start, tl.total_frames, cur_count, base + cur_cost))
    return chunks


def mix_groups(count: int, budget: Budget) -> int:
    """The most sounds one mix pass may open; ``render_too_large`` when not even one fits."""
    room = math.floor((budget.mb - MIX_FIXED_MB) / SOUND_USE_MB)
    if room < 1 and count:
        raise RenderError(
            "render_too_large",
            f"the memory budget of {budget.mb:.0f} MB does not hold one sound of the mix "
            f"(about {MIX_FIXED_MB + SOUND_USE_MB:.0f} MB)",
            "give a larger --memory-budget",
            budget_mb=round(budget.mb),
        )
    return max(room, 1)


def mix_memory_mb(sounds: int) -> float:
    return MIX_FIXED_MB + SOUND_USE_MB * sounds
