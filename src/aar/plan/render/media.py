"""Hub media for the render: every used source must be a readable file on the hub (S18b).

A source resolves to a hub file in one of two ways, and anything else is refused as
``media_not_on_hub`` (with a ``reason``):

* ``plan-path``: the plan's media is a hub path (absolute, or a ``file:`` URL of one; a relative
  name resolves against ``media_root``, the plan file's folder in the CLI) and the file exists;
* ``media-map``: the plan's media is an edit-PC path (``E:\\...``, ``\\\\host\\share\\...`` or a
  ``file:`` URL of one) and the media map (the same ``linux_root -> windows_root`` JSON the FCPXML
  export takes, read the other way) maps it to a hub file that exists.

A media-pool name (a multicam clip) has no file and is refused. Each file is probed with
:func:`aar.media.probe` (streams, display size) and a timing probe of its own (stream start
times, frame counts, the base and average rates, the first packets' timestamps), and refused when
it cannot be cut exactly (round 2):

* ``media_unreadable`` (ffprobe cannot read it), ``no_video_stream``;
* ``source_rate_mismatch``: the video's base rate is not the plan's within 0.01 % (24 fps in a
  23.976 plan drifts 0.6 s in 10 minutes, so it is refused, not accepted);
* ``source_vfr``: a variable-frame-rate video: the base and average rates differ by more than
  0.1 %, any step between its packets' timestamps (one scan over all of them, sorted, demuxing
  only) is more than 1/8 frame off one frame (a dropped, doubled or shifted frame, named by its
  time), or its frame count is not the stream's duration times its rate. The picture is cut by
  timestamp and the sound by time, so they must agree;
* ``size_not_renderable`` (no picture size), ``frame_size_mismatch`` (a ``Source.frame_size``
  whose aspect is not the file's display aspect).

The frame count comes from the **video stream** (its ``nb_frames``, else its packet count),
never from the container's duration, which is the longest stream's. Stream start times are
kept, so a sound is cut where its picture is even when the audio starts after the video.
Nothing here writes.
"""

from __future__ import annotations

import itertools
import json
import os
import re
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal
from urllib.parse import unquote, urlsplit

from aar.media import MediaError, MediaInfo, input_url, probe, run_tool, timeout_for
from aar.plan.apply.inputs import classify_media
from aar.plan.export.errors import ExportError
from aar.plan.export.media import MediaMap, media_ref
from aar.plan.model import EditPlan, Fps, Source
from aar.plan.render.errors import RenderError

HubRule = Literal["plan-path", "media-map"]

#: How far a plan frame size's aspect may sit from the file's display aspect (a face track's
#: sampled size rounds, e.g. 853x480 for 16:9).
ASPECT_TOLERANCE = 0.005
_DRIVE_URL_PATH = re.compile(r"^/[A-Za-z]:/")


#: How far a source's base frame rate may sit from the plan's: 0.01 %.
RATE_TOLERANCE = Fraction(1, 10000)
#: How far the average rate may sit from the base rate before the video counts as variable.
VFR_TOLERANCE = Fraction(1, 1000)
#: A packet interval further than this share of a frame from one frame is a variable rate.
VFR_STEP_TOLERANCE = Fraction(1, 8)
#: The packet scans already done, by file identity (path, device, inode, size, mtime): a source
#: used by several renders in one process is scanned once.
_SCANS: dict[tuple[str, int, int, int, int, int], tuple[int, tuple[Fraction, Fraction] | None]] = {}
_SCAN_CACHE_SIZE = 256
SCAN_STDOUT_LIMIT = 64 * 1024 * 1024
TIMING_TIMEOUT_S = 30.0


@dataclass(frozen=True)
class HubSource:
    """One source's hub file and what ffprobe says about it."""

    source_id: str
    plan_media: str
    path: Path
    rule: HubRule
    info: MediaInfo
    #: The picture as decoded and shown (pixel aspect applied, display rotation applied).
    display: tuple[int, int]
    #: The pixel aspect when it is not square: the render scales to ``display`` first.
    sample_aspect: Fraction | None
    #: The video stream's base frame rate (r_frame_rate), the plan's within 0.01 %.
    fps: Fraction
    #: Whole frames in the video stream (never the container's length).
    frames: int
    #: The first audio stream's channel count; None when the file has no sound.
    channels: int | None
    video_index: int = 0
    audio_index: int | None = None
    #: Start times (seconds, absolute container time): the container, the video, the audio.
    start_s: Fraction = Fraction(0)
    video_start_s: Fraction = Fraction(0)
    audio_start_s: Fraction = Fraction(0)
    sample_rate: int | None = None
    #: The video stream's time base: a picture is cut by timestamps in these ticks.
    video_time_base: Fraction | None = None

    @property
    def video_offset_s(self) -> Fraction:
        """The video's start after the container's (what an input ``-ss`` counts from)."""
        return self.video_start_s - self.start_s

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source_id,
            "plan_media": self.plan_media,
            "hub_path": str(self.path),
            "rule": self.rule,
            "fps": f"{self.fps.numerator}/{self.fps.denominator}",
            "display_size": list(self.display),
            "frames": self.frames,
            "audio_channels": self.channels,
            "video_start_s": float(self.video_start_s),
            "audio_start_s": float(self.audio_start_s) if self.channels else None,
        }


def _refuse(source: Source, why: str, hint: str, reason: str, **extra: Any) -> RenderError:
    return RenderError(
        "media_not_on_hub",
        f"source {source.id!r} media {source.media!r} {why}",
        hint,
        source=source.id,
        media=source.media,
        reason=reason,
        **extra,
    )


_MAP_HINT = (
    "give the hub copy's path in the plan (aar media ingest prints it), or pass --media-map with "
    "a JSON object of linux_root -> windows_root that covers this path"
)


def _edit_path(source: Source) -> str:
    """The edit-PC path of a Windows path or a ``file:`` URL of one (S18's own rules)."""
    try:
        return media_ref(source).windows_path
    except ExportError as exc:
        raise _refuse(
            source,
            f"is not a hub path or an edit-PC path the media map can read ({exc.message})",
            _MAP_HINT,
            str(exc.details.get("reason", exc.code)),
        ) from None


def _mapped(source: Source, edit: str, media_map: MediaMap | None) -> Path:
    if media_map is None:
        raise _refuse(
            source, "is an edit-PC path and no media map was given", _MAP_HINT, "no-media-map"
        )
    result = media_map.pathmap.edit_to_hub(edit)
    if result.new is None:
        raise _refuse(
            source,
            f"is an edit-PC path the media map does not cover ({result.reason})",
            _MAP_HINT,
            "not-in-media-map",
        )
    return Path(result.new)


def hub_path(
    source: Source, media_map: MediaMap | None, media_root: Path | None
) -> tuple[Path, HubRule]:
    """Where the source's media is on the hub, by the rules in the module docstring (the file's
    existence is checked by :func:`resolve_source`)."""
    media = source.media
    if media[:5].lower() == "file:":
        try:
            parts = urlsplit(media)
        except ValueError:
            raise _refuse(source, "is a malformed file: URL", _MAP_HINT, "malformed-url") from None
        path = unquote(parts.path)
        local = parts.netloc in ("", "localhost") and not parts.query and not parts.fragment
        hub_url = path.startswith("/") and not path.startswith("//")
        if local and hub_url and not _DRIVE_URL_PATH.match(path):
            return Path(path), "plan-path"
        return _mapped(source, _edit_path(source), media_map), "media-map"
    if media.startswith("/") and not media.startswith("//"):
        return Path(media), "plan-path"
    kind = classify_media(media)
    if kind == "import":
        return _mapped(source, _edit_path(source), media_map), "media-map"
    if kind == "pool":
        raise _refuse(
            source,
            "is a media-pool name (e.g. a multicam clip), not a file",
            "give the underlying camera file's hub path, or its edit-PC path with --media-map",
            "pool-name",
        )
    if media_root is None:
        raise _refuse(
            source,
            "is a relative name and there is no media root to resolve it against",
            "give an absolute hub path, or render from the CLI (relative to the plan's folder)",
            "relative",
        )
    return media_root / media, "plan-path"


def _probe(source: Source, path: Path) -> MediaInfo:
    try:
        return probe(path)
    except MediaError as exc:
        raise RenderError(
            "media_unreadable",
            f"source {source.id!r}: ffprobe cannot read {str(path)!r}: {exc.summary()}",
            "check the hub copy (aar media availability), or re-ingest it",
            source=source.id,
            path=str(path),
        ) from None


def _fraction(value: Any) -> Fraction | None:
    """A decimal or rational ffprobe value (``"0.250000"``, ``"24000/1001"``) exactly."""
    if not isinstance(value, str) or not value.strip() or value.strip() in ("N/A", "0/0"):
        return None
    try:
        return Fraction(value.strip())
    except (ValueError, ZeroDivisionError):
        return None


def _tag_duration(raw: dict[str, Any]) -> Fraction | None:
    """A Matroska stream's ``DURATION`` tag (``"00:00:29.196000000"``) in seconds."""
    tags = raw.get("tags") if isinstance(raw.get("tags"), dict) else {}
    for key, value in tags.items():
        if isinstance(key, str) and key.upper() == "DURATION" and isinstance(value, str):
            hms = value.strip().split(":")
            try:
                if len(hms) == 3:
                    return int(hms[0]) * 3600 + int(hms[1]) * 60 + Fraction(hms[2])
            except (ValueError, ZeroDivisionError):
                return None
    return None


def _elapsed(raw: dict[str, Any]) -> Fraction | None:
    """How long a stream lasts by its Matroska ``DURATION`` tag. ffmpeg's Matroska muxer writes
    the tag as the stream's **end** timestamp, not its length (round 4, measured on ffmpeg 6.1: a
    250-frame 25 fps video starting at 5 s says ``00:00:15``), so the stream's own start time
    comes off; a missing start counts as 0. None when there is no tag, or when the result is not
    a positive length (then the frame-count test does not run)."""
    end = _tag_duration(raw)
    if end is None:
        return None
    start = _fraction(raw.get("start_time"))
    elapsed = end - (start if start is not None else Fraction(0))
    return elapsed if elapsed > 0 else None


def _ffprobe_json(path: Path, args: list[str], timeout_s: float) -> dict[str, Any]:
    result = run_tool(
        "ffprobe",
        ["-hide_banner", "-v", "error", *args, "-of", "json", input_url(path)],
        timeout_s=timeout_s,
        stdout_limit=4 * 1024 * 1024,
    )
    data = json.loads(result.stdout.decode("utf-8", errors="replace") or "{}")
    if not isinstance(data, dict):
        raise MediaError("ffprobe printed JSON that is not an object", tool="ffprobe")
    return data


@dataclass(frozen=True)
class Timing:
    """The timing probe of one file (see the module docstring)."""

    start_s: Fraction
    video_start_s: Fraction
    audio_start_s: Fraction
    base_rate: Fraction | None
    avg_rate: Fraction | None
    frames: int | None
    frames_from: str
    duration_frames: Fraction | None  # stream duration x base rate, when known
    #: The first interval between packets (sorted by time) that is not one frame: (its start, s
    #: after the video's start; its length in frames). None when every step is one frame.
    uneven: tuple[Fraction, Fraction] | None
    sample_rate: int | None
    time_base: Fraction | None = None  # the video stream's time base (its pts tick)
    packets: int | None = None  # the video packets the scan could read


def _stream(data: dict[str, Any], index: int | None) -> dict[str, Any]:
    for raw in data.get("streams") or []:
        if isinstance(raw, dict) and raw.get("index") == index:
            return raw
    return {}


def scan_packets(
    path: Path, video_index: int, time_base: Fraction | None, rate: Fraction | None
) -> tuple[int, tuple[Fraction, Fraction] | None]:
    """Every video packet's timestamp, in one ffprobe pass (demuxing only, no decode): the packet
    count, and the first step between them, sorted (B-frames come out of order), that is not one
    frame within :data:`VFR_STEP_TOLERANCE`. Cached by the file's identity."""
    st = path.stat()
    key = (str(path), st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, video_index)
    if key in _SCANS:
        return _SCANS[key]
    out = run_tool(
        "ffprobe",
        [
            *("-hide_banner", "-v", "error", "-select_streams", str(video_index)),
            *("-show_entries", "packet=pts", "-of", "csv=p=0", input_url(path)),
        ],
        timeout_s=timeout_for(st.st_size / 20e6, base_s=TIMING_TIMEOUT_S, per_media_s=1.0),
        stdout_limit=SCAN_STDOUT_LIMIT,
    ).stdout
    # ffprobe can append packet side data (including an empty field) after the PTS.
    pts = sorted(
        int(t)
        for row in out.splitlines()
        if (t := row.partition(b",")[0].strip()).removeprefix(b"-").isdigit()
    )
    uneven = None
    if time_base and rate and len(pts) > 1:
        frame = 1 / (time_base * rate)  # ticks per frame
        slack = frame * VFR_STEP_TOLERANCE
        for a, b in itertools.pairwise(pts):
            if abs((b - a) - frame) > slack:
                uneven = ((a - pts[0]) * time_base, (b - a) / frame)
                break
    if len(_SCANS) >= _SCAN_CACHE_SIZE:
        _SCANS.pop(next(iter(_SCANS)))
    _SCANS[key] = (len(pts), uneven)
    return _SCANS[key]


def timing_probe(path: Path, video_index: int, audio_index: int | None) -> Timing:
    """Stream start times, rates, the video's frame count and its packets' spacing."""
    data = _ffprobe_json(
        path,
        [
            "-show_entries",
            "format=start_time:stream=index,codec_type,start_time,duration,nb_frames,"
            "r_frame_rate,avg_frame_rate,time_base,sample_rate:stream_tags=DURATION",
        ],
        TIMING_TIMEOUT_S,
    )
    video = _stream(data, video_index)
    audio = _stream(data, audio_index)
    fmt = data.get("format") if isinstance(data.get("format"), dict) else {}
    start = _fraction(fmt.get("start_time")) or Fraction(0)
    v_start = _fraction(video.get("start_time"))
    v_start = start if v_start is None else v_start
    a_start = _fraction(audio.get("start_time")) if audio else None
    base = _fraction(video.get("r_frame_rate")) or _fraction(video.get("avg_frame_rate"))
    avg = _fraction(video.get("avg_frame_rate"))
    duration = _fraction(video.get("duration")) or _elapsed(video)
    duration_frames = None if duration is None or base is None else duration * base
    frames: int | None = None
    frames_from = "nb_frames"
    nb = video.get("nb_frames")
    if isinstance(nb, str) and nb.isdigit() and int(nb) > 0:
        frames = int(nb)
    tb = _fraction(video.get("time_base"))
    count, uneven = scan_packets(path, video_index, tb, base)
    if frames is None and count:
        frames, frames_from = count, "packet count"
    rate = _fraction(audio.get("sample_rate")) if audio else None
    return Timing(
        start,
        v_start,
        start if a_start is None else a_start,
        base,
        avg,
        frames,
        frames_from,
        duration_frames,
        uneven,
        None if rate is None else int(rate),
        tb,
        count,
    )


def _rate_text(rate: Fraction) -> str:
    return f"{rate.numerator}/{rate.denominator} ({float(rate):.4f})"


def _check_rate(source: Source, path: Path, timing: Timing, want: Fraction) -> None:
    hint = "conform the hub copy to the plan's frame rate (constant) first, then render it"
    if not timing.time_base:
        raise RenderError(
            "media_unreadable",
            f"source {source.id!r}: the video of {str(path)!r} states no time base, so its "
            "frames cannot be picked by timestamp",
            "re-wrap the hub copy (ffmpeg -c copy into .mov or .mp4) and render again",
            source=source.id,
            path=str(path),
        )
    base = timing.base_rate
    if base is None or abs(base / want - 1) > RATE_TOLERANCE:
        got = "an unknown rate" if base is None else _rate_text(base)
        raise RenderError(
            "source_rate_mismatch",
            f"source {source.id!r}: {str(path)!r} runs at {got} fps, not the plan's "
            f"{_rate_text(want)}; frames are cut by index and sound by time, so even 0.1 % "
            "drifts apart",
            hint,
            source=source.id,
            path=str(path),
            fps=None if base is None else f"{base.numerator}/{base.denominator}",
            plan_fps=f"{want.numerator}/{want.denominator}",
        )
    if (
        timing.frames_from == "nb_frames"
        and timing.packets is not None
        and timing.frames is not None
        and timing.packets < timing.frames
    ):  # first: a truncated file's last packets leave a hole that looks like a variable rate
        raise RenderError(
            "media_unreadable",
            f"source {source.id!r}: the index of {str(path)!r} lists {timing.frames} video "
            f"frames but only {timing.packets} can be read (a truncated or damaged file)",
            "copy the file to the hub again (or re-wrap it with ffmpeg -c copy), then render",
            source=source.id,
            path=str(path),
            frames=timing.frames,
            readable=timing.packets,
        )
    why = None
    if timing.uneven is not None:  # first: it names the time
        at, step = timing.uneven
        why = (
            f"{float(at):.3f} s into its video a frame lasts {float(step):.3f} frames, not one "
            "(a dropped or shifted frame)"
        )
    elif timing.avg_rate is not None and abs(timing.avg_rate / base - 1) > VFR_TOLERANCE:
        why = f"its average rate {_rate_text(timing.avg_rate)} is not its base rate"
    elif (
        timing.frames is not None
        and timing.duration_frames is not None
        and abs(timing.frames - timing.duration_frames) > 1
    ):
        why = (
            f"it holds {timing.frames} frames but lasts {float(timing.duration_frames):.1f} "
            "frames at its rate (frames dropped or doubled)"
        )
    if why is not None:
        raise RenderError(
            "source_vfr",
            f"source {source.id!r}: {str(path)!r} has a variable frame rate: {why}",
            hint,
            source=source.id,
            path=str(path),
        )


def resolve_source(
    source: Source,
    plan_fps: Fps,
    *,
    media_map: MediaMap | None,
    media_root: Path | None,
) -> HubSource:
    """The source's hub file, probed and checked (module docstring)."""
    path, rule = hub_path(source, media_map, media_root)
    path = Path(os.path.abspath(path))
    if not path.is_file() or not os.access(path, os.R_OK):
        what = "is missing" if not path.exists() else "is not a readable regular file"
        raise _refuse(
            source,
            f"resolves to {str(path)!r} on the hub, which {what}",
            "ingest the media on the hub (aar media ingest) and give that path, or fix the map",
            "missing" if not path.exists() else "unreadable",
            hub_path=str(path),
        )
    info = _probe(source, path)
    video = info.video
    if video is None:
        raise RenderError(
            "no_video_stream",
            f"source {source.id!r}: {str(path)!r} has no video stream",
            "give a file with picture",
            source=source.id,
            path=str(path),
        )
    audio = info.audio
    try:
        timing = timing_probe(path, video.index, None if audio is None else audio.index)
    except (MediaError, json.JSONDecodeError) as exc:
        detail = exc.summary() if isinstance(exc, MediaError) else str(exc)
        raise RenderError(
            "media_unreadable",
            f"source {source.id!r}: ffprobe cannot read the timing of {str(path)!r}: {detail}",
            "check the hub copy (aar media availability), or re-ingest it",
            source=source.id,
            path=str(path),
        ) from None
    want = Fraction(plan_fps.num, plan_fps.den)
    _check_rate(source, path, timing, want)
    display = video.display_size
    if display is None or min(display) < 2:
        raise RenderError(
            "size_not_renderable",
            f"source {source.id!r}: {str(path)!r} has no usable picture size ({display})",
            "give a file whose video stream has a known width and height",
            source=source.id,
            path=str(path),
        )
    if source.frame_size is not None:
        fw, fh = source.frame_size[0], source.frame_size[1]
        off = abs(fw * display[1] - fh * display[0]) / (fw * display[1])
        if off > ASPECT_TOLERANCE:
            raise RenderError(
                "frame_size_mismatch",
                f"source {source.id!r}: the plan's frame_size {fw}x{fh} does not have the hub "
                f"file's display aspect ({display[0]}x{display[1]})",
                "set frame_size to the media's display size (aar plan produce reads it)",
                source=source.id,
                frame_size=[fw, fh],
                display_size=list(display),
            )
    if timing.frames is None:
        raise RenderError(
            "media_unreadable",
            f"source {source.id!r}: the video stream of {str(path)!r} has no frame count, "
            "duration or countable packets",
            "re-mux or re-ingest the hub copy",
            source=source.id,
            path=str(path),
        )
    aspect = video.sample_aspect if video.sample_aspect not in (None, Fraction(1)) else None
    return HubSource(
        source.id,
        source.media,
        path,
        rule,
        info,
        display,
        aspect,
        timing.base_rate or want,
        timing.frames,
        None if audio is None else (audio.channels or 1),
        video_index=video.index,
        audio_index=None if audio is None else audio.index,
        start_s=timing.start_s,
        video_start_s=timing.video_start_s,
        audio_start_s=timing.audio_start_s,
        sample_rate=timing.sample_rate or (None if audio is None else audio.sample_rate),
        video_time_base=timing.time_base,
    )


def used_source_ids(plan: EditPlan) -> list[str]:
    """Ids of the sources the timeline plays, in plan order (as the export resolves them)."""
    used = {seg.source for seg in plan.segments}
    if plan.outro is not None:
        used.add(plan.outro.source)
    seen: list[str] = []
    for source in plan.sources:
        if source.id in used and source.id not in seen:
            seen.append(source.id)
    return seen
