"""``aar.plan.render``: an EditPlan v2 rendered to MP4 with ffmpeg on the hub, no Resolve (S18b).

:func:`render_plan` lints the plan (and refuses one with lint errors, as the FCPXML export does),
resolves every used source to a probed hub file (:mod:`~aar.plan.render.media`), builds the FCPXML
export's own :class:`~aar.plan.export.layout.Layout` with those hub paths (so every placement,
handle, alignment and fade is S18's), plans the render's memory (:mod:`~aar.plan.render.chunks`)
and renders it in a work dir beside the output:

1. the mix (:func:`~aar.plan.render.graph.mix_graph`) to a 32-bit float WAV of the exact length,
   in as many passes as the memory budget needs (then summed, :func:`~.graph.sum_graph`);
2. its loudness to the profile's target (:mod:`~aar.plan.render.loudness`);
3. the picture in **chunks** (round 3), one ffmpeg after another, each with its own graph
   (:func:`~aar.plan.render.graph.video_graph` over the chunk's record frames) and only its own
   pictures' inputs, cut at hard cuts only; each chunk must come out exactly its frames, or the
   render stops with ``render_incomplete`` naming the picture that decodes short;
4. the chunks joined by stream copy (the concat demuxer; every chunk starts on an IDR frame of an
   identical encoder setup, its stream parameters checked equal) with the sound muxed once:
   ``review`` is H.264 High yuv420p + AAC with faststart, ``master`` a higher-quality H.264, or
   ProRes 422 HQ in a .mov when ``[render] master_codec`` says so. The joined stream's timestamps
   are put back on the exact frame grid (``setts``), so the join is one continuous stream; the file
   is checked for its exact frame count, strictly increasing DTS and the sound;
5. :func:`aar.qa.check` on the result with the plan's kind;
6. the file and its report (``<stem>.render.json``) placed as one pair by S18's pair writer.

The work dir is removed afterwards, on success and failure. Every ffmpeg call has a timeout
scaled to the source it decodes, passes argument lists only, and keeps a bounded stderr tail.
Nothing touches ``/tmp`` or the network.
"""

from __future__ import annotations

import contextlib
import itertools
import json
import math
import os
import secrets
import shutil
import signal
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal

from aar.host.transport import sha256_file
from aar.media import (
    MediaError,
    MediaToolMissing,
    input_url,
    probe,
    resolve_tool,
    run_tool,
    timeout_for,
)
from aar.plan.apply.inputs import plan_hash
from aar.plan.export.errors import ExportError
from aar.plan.export.files import _refuse_existing, write_pair
from aar.plan.export.layout import Layout, build_layout
from aar.plan.export.media import MediaMap, MediaRef
from aar.plan.inputs import LaughMap, Transcript
from aar.plan.lint import Finding, lint
from aar.plan.model import EditPlan, Source
from aar.plan.profile import Profile
from aar.plan.render.chunks import (
    Budget,
    Chunk,
    memory_budget,
    mix_groups,
    mix_memory_mb,
    plan_chunks,
)
from aar.plan.render.errors import RENDER_CODES, RUNTIME_CODES, RenderError, RenderTerminated
from aar.plan.render.fonts import FontChoice, find_font
from aar.plan.render.graph import (
    SAMPLE_RATE,
    mix_graph,
    picture_pts,
    picture_use,
    sample_at,
    sum_graph,
    video_graph,
)
from aar.plan.render.loudness import normalize
from aar.plan.render.media import HubSource, resolve_source, used_source_ids
from aar.plan.render.report import not_rendered, notes, rendered
from aar.plan.render.runner import BASE_ARGS, run_ffmpeg
from aar.plan.render.timeline import Timeline, build_timeline

__all__ = [
    "RENDER_CODES",
    "RUNTIME_CODES",
    "Preset",
    "RenderError",
    "RenderResult",
    "RenderTerminated",
    "render_layout",
    "render_plan",
    "render_report_path",
]

Preset = Literal["review", "master"]
LUT_SUFFIXES = (".cube", ".3dl", ".dat", ".m3d", ".csp")
LUT_MAX_BYTES = 64 * 1024 * 1024


@dataclass
class RenderResult:
    """The placed file, its report, and the lint warnings that did not block it."""

    out: Path
    report_path: Path
    report: dict[str, Any]
    warnings: list[Finding] = field(default_factory=list)

    @property
    def qa_ok(self) -> bool:
        return bool(self.report["qa"]["ok"])


def render_report_path(out: Path) -> Path:
    """Where the render report goes: beside the file, ``<stem>.render.json`` (a name the FCPXML
    export's ``<stem>.report.json`` never takes)."""
    return out.with_name(f"{out.stem}.render.json")


def _container(out: Path, preset: Preset, profile: Profile) -> str:
    suffix = out.suffix.lower()
    prores = preset == "master" and profile.render.master_codec == "prores"
    allowed = (".mov",) if prores else (".mp4", ".mov")
    if suffix not in allowed:
        raise RenderError(
            "bad_output_path",
            f"--out {out.name!r}: a {'ProRes ' if prores else ''}{preset} render is written as "
            f"{' or '.join(allowed)}",
            f"give an output name ending in {allowed[0]}",
            path=str(out),
        )
    if not out.parent.is_dir():
        raise RenderError(
            "bad_output_path",
            f"the output's folder {str(out.parent)!r} does not exist",
            "create the folder first, or give another --out",
            path=str(out),
        )
    return "mov" if suffix == ".mov" else "mp4"


@dataclass(frozen=True)
class Codec:
    """The encoder setup: every chunk's video arguments (identical, so the chunks join by stream
    copy), the sound's, the picture format the graph ends in and the chunks' container."""

    video: list[str]
    audio: list[str]
    pix_fmt: str
    part_format: str

    def args(self, container: str) -> list[str]:
        return [*self.video, *self.audio, "-movflags", "+faststart", "-f", container]


def _codec_args(preset: Preset, profile: Profile, container: str) -> Codec:
    settings = profile.render
    if preset == "master" and settings.master_codec == "prores":
        video = ["-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le"]
        return Codec(video, ["-c:a", "pcm_s24le"], "yuv422p10le", "mov")
    crf = settings.review_crf if preset == "review" else settings.master_crf
    speed = settings.review_x264_preset if preset == "review" else settings.master_x264_preset
    kbps = settings.review_audio_kbps if preset == "review" else settings.master_audio_kbps
    video = ["-c:v", "libx264", "-profile:v", "high", "-pix_fmt", "yuv420p"]
    # Closed GOPs (x264's default, stated): each chunk is decodable from its first frame.
    video += ["-preset", speed, "-crf", str(crf), "-x264-params", "open-gop=0"]
    return Codec(video, ["-c:a", "aac", "-b:a", f"{kbps}k"], "yuv420p", "mp4")


def _hub_ref(hub: dict[str, HubSource]) -> Any:
    def resolve(source: Source) -> MediaRef:
        media = hub[source.id]
        path = str(media.path)
        return MediaRef(source.id, source.media, path, "hub-path", path, media.path.name)

    return resolve


def render_layout(plan: EditPlan, hub: dict[str, HubSource], profile: Profile) -> Layout:
    """S18's layout with hub paths. Start timecodes do not matter to a render (no clip links to
    a file), so they are left out: none is refused, unknown or off the frame grid."""
    untimed = plan.model_copy(
        update={
            "sources": [
                s.model_copy(update={"start_timecode": None, "start_timecode_rate": None})
                for s in plan.sources
            ]
        }
    )
    try:
        return build_layout(
            untimed,
            timeline_start=profile.export.timeline_start,
            allow_unknown_timecode=True,
            resolve_media=_hub_ref(hub),
        )
    except ExportError as exc:
        raise RenderError.from_export(exc) from None


def _luts(plan: EditPlan, used: list[str], media_root: Path | None) -> dict[str, Path]:
    """The LUT file of each used source whose ``color.lut`` names a LUT file on the hub."""
    found: dict[str, Path] = {}
    for source in plan.sources:
        lut = source.color.lut if source.color is not None else None
        if source.id not in used or lut is None or not lut.lower().endswith(LUT_SUFFIXES):
            continue
        path = Path(lut)
        if not path.is_absolute():
            if media_root is None:
                continue
            path = media_root / path
        with contextlib.suppress(OSError):
            if path.is_file() and os.access(path, os.R_OK) and path.stat().st_size < LUT_MAX_BYTES:
                found[source.id] = path
    return found


def _qa_plan(plan: EditPlan, hub: dict[str, HubSource]) -> EditPlan:
    """The plan with each used source's media replaced by its hub path, for QA's join checks."""
    return plan.model_copy(
        update={
            "sources": [
                s.model_copy(update={"media": str(hub[s.id].path)}) if s.id in hub else s
                for s in plan.sources
            ]
        }
    )


def _work_dir(out: Path) -> Path:
    """A fresh work dir beside the output (so the result is placed by a rename on its own file
    system, never through /tmp)."""
    while True:
        work = out.parent / f".{out.name}.aar-render-{secrets.token_hex(4)}"
        try:
            work.mkdir(mode=0o700)
        except FileExistsError:
            continue
        return work


#: The largest picture filter script a render writes (moving reframes take most of it).
MAX_FILTER_SCRIPT_BYTES = 512 * 1024


def estimated_memory_mb(
    tl: Timeline, hub: dict[str, HubSource], budget: Budget | None = None
) -> float:
    """The planned peak of the render's largest ffmpeg under ``budget`` (the default budget
    when None): its largest chunk, or the mix pass."""
    budget = budget or memory_budget()
    chunks = plan_chunks(tl, hub, budget, max_script_bytes=MAX_FILTER_SCRIPT_BYTES)
    per_pass = mix_groups(len(tl.sounds), budget)
    mix = mix_memory_mb(min(per_pass, len(tl.sounds)))
    return max([mix, *(chunk.planned_mb for chunk in chunks)])


def _timeout(settings: Any, spans: list[Fraction]) -> float:
    """A call's timeout from the source seconds it decodes (round 2: not the timeline's length;
    each seek may decode back to the keyframe before its range, counted in the spans)."""
    return timeout_for(
        float(sum(spans, Fraction(0))),
        base_s=settings.timeout_base_s,
        per_media_s=settings.timeout_per_media_s,
    )


@contextlib.contextmanager
def _sigterm_interrupts() -> Iterator[None]:
    """SIGTERM raises :class:`RenderTerminated` (an interrupt) while rendering, so ffmpeg's
    process group is killed and the work dir removed, as for Ctrl-C. Only in the main thread (a
    signal handler cannot be set from any other); the previous handler comes back afterwards."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def stop(signum: int, _frame: Any) -> None:
        raise RenderTerminated(signum)

    previous = signal.signal(signal.SIGTERM, stop)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def _incomplete(message: str, stage: str, **details: Any) -> RenderError:
    return RenderError(
        "render_incomplete",
        message,
        "this is a render bug or a broken source; report the plan",
        stage=stage,
        **details,
    )


@dataclass(frozen=True)
class Packets:
    """A file's video packets: (pts, dts) in decode order, and whether the first is a key."""

    times: list[tuple[int, int]]
    first_key: bool
    time_base: Fraction


def _packets(path: Path, stage: str) -> Packets:
    """Every video packet's timestamps, from one ffprobe pass."""
    try:
        info = probe(path)
        video = info.video
        if video is None:
            raise _incomplete(f"{path.name} has no video stream", stage)
        data = run_tool(
            "ffprobe",
            [
                *("-hide_banner", "-v", "error", "-select_streams", str(video.index)),
                *("-show_entries", "packet=pts,dts,flags:stream=time_base"),
                *("-of", "csv=p=0", input_url(path)),
            ],
            timeout_s=timeout_for(video.duration_s, base_s=60.0, per_media_s=0.5),
            stdout_limit=PACKET_STDOUT_LIMIT,
        )
    except MediaError as exc:
        raise RenderError(
            "render_incomplete",
            f"{path.name} cannot be read back: {exc.summary()}",
            "check the disk and ffmpeg, then render again",
            stage=stage,
        ) from None
    times: list[tuple[int, int]] = []
    first_key = False
    time_base = None
    for line in data.stdout.decode("utf-8", errors="replace").splitlines():
        fields = line.strip().split(",")
        if len(fields) == 1 and "/" in fields[0]:
            num, _, den = fields[0].partition("/")
            if num.isdigit() and den.isdigit() and int(den):
                time_base = Fraction(int(num), int(den))
            continue
        if len(fields) < 3 or not all(f.lstrip("-").isdigit() for f in fields[:2]):
            raise _incomplete(f"{path.name} has a video packet without timestamps", stage)
        if not times:
            first_key = fields[2].startswith("K")
        times.append((int(fields[0]), int(fields[1])))
    if time_base is None:
        raise _incomplete(f"{path.name}: its video stream states no time base", stage)
    return Packets(times, first_key, time_base)


#: Bytes of ffprobe packet listing read back (about 20 bytes a frame: hours of video).
PACKET_STDOUT_LIMIT = 64 * 1024 * 1024


def _check_output(path: Path, frames: int, frame_s: Fraction) -> None:
    """The joined file holds exactly the timeline's frames as one continuous stream: one packet
    per frame, the presentation times exactly the frame grid with no hole, the decode times
    strictly increasing and never after a packet's presentation; and a sound stream. Else
    ``render_incomplete`` (never left to QA)."""
    packets = _packets(path, "verify")
    problems = []
    times = packets.times
    if len(times) != frames:
        problems.append(f"its video holds {len(times)} frames, not the timeline's {frames}")
    tick = frame_s / packets.time_base
    if tick.denominator != 1:
        problems.append(f"its video time base {packets.time_base} does not hold a frame exactly")
    else:
        step = int(tick)
        pts = sorted(t[0] for t in times)
        if pts and pts != [pts[0] + k * step for k in range(len(pts))]:
            problems.append("its frames' presentation times are not one continuous frame grid")
        dts = [t[1] for t in times]
        if any(b <= a for a, b in itertools.pairwise(dts)):
            problems.append("its decode times are not strictly increasing")
        if any(d > p for p, d in times):
            problems.append("a frame is decoded after it is shown")
    with contextlib.suppress(MediaError):
        if probe(path).audio is None:
            problems.append("it has no audio stream")
    if problems:
        raise _incomplete(f"the encode is incomplete: {'; '.join(problems)}", "verify")


def _stream_setup(path: Path) -> dict[str, Any]:
    """What must be equal across chunks for a stream-copy join: the codec, its parameters and
    its extradata (H.264's SPS/PPS)."""
    try:
        data = run_tool(
            "ffprobe",
            [
                *("-hide_banner", "-v", "error", "-select_streams", "v:0", "-show_streams"),
                *("-show_data_hash", "sha256", "-of", "json", input_url(path)),
            ],
            timeout_s=60.0,
        )
        stream = json.loads(data.stdout.decode("utf-8", errors="replace"))["streams"][0]
    except (MediaError, ValueError, KeyError, IndexError):
        raise _incomplete(f"{path.name} cannot be read back", "join") from None
    keys = (
        "codec_name", "profile", "level", "width", "height", "pix_fmt", "sample_aspect_ratio",
        "has_b_frames", "time_base", "r_frame_rate", "field_order", "color_range",
        "extradata_hash",
    )  # fmt: skip
    return {key: stream.get(key) for key in keys}


def _wav_args(name: str) -> list[str]:
    return [
        *("-c:a", "pcm_f32le", "-ar", str(SAMPLE_RATE), "-ac", "2"),
        *("-rf64", "auto", "-f", "wav", name),
    ]


def _mix(tl: Timeline, hub: dict[str, HubSource], per_pass: int, settings: Any, work: Path) -> int:
    """``mix.wav``: every sound, in passes of at most ``per_pass`` sounds (in record order, each
    pass over its own sounds' samples only), summed when there is more than one. The passes."""
    fps = tl.plan.fps
    total = sample_at(fps, tl.total_frames)
    order = sorted(range(len(tl.sounds)), key=lambda i: (tl.sounds[i].start, i))
    groups = [order[i : i + per_pass] for i in range(0, len(order), per_pass)] or [[]]
    if len(groups) == 1:
        mix = mix_graph(tl, hub)
        (work / "mix.ffgraph").write_text(mix.text, encoding="utf-8")
        run_ffmpeg(
            [
                *("-v", "error", "-y", "-copyts", *(a for use in mix.uses for a in use.args())),
                *("-filter_complex_script", "mix.ffgraph", "-map", "[mix]", *_wav_args("mix.wav")),
            ],
            "mix",
            _timeout(settings, [use.span_s for use in mix.uses]),
            work,
        )
        return 1
    spans: list[tuple[int, int]] = []
    for n, group in enumerate(groups):
        first = min(sample_at(fps, tl.sounds[i].start) for i in group)
        end = max(sample_at(fps, tl.sounds[i].end) for i in group)
        mix = mix_graph(tl, hub, group, (first, end))
        script = f"group{n:03d}.mix.ffgraph"
        (work / script).write_text(mix.text, encoding="utf-8")
        run_ffmpeg(
            [
                *("-v", "error", "-y", "-copyts", *(a for use in mix.uses for a in use.args())),
                *("-filter_complex_script", script, "-map", "[mix]"),
                *_wav_args(f"group{n:03d}.wav"),
            ],
            f"mix {n + 1}/{len(groups)}",
            _timeout(settings, [use.span_s for use in mix.uses]),
            work,
        )
        spans.append((first, end))
    (work / "sum.ffgraph").write_text(sum_graph(spans, total), encoding="utf-8")
    inputs = [a for n in range(len(groups)) for a in ("-i", f"group{n:03d}.wav")]
    timeline_s = Fraction(total, SAMPLE_RATE)
    run_ffmpeg(
        [
            *("-v", "error", "-y", *inputs, "-filter_complex_script", "sum.ffgraph"),
            *("-map", "[mix]", *_wav_args("mix.wav")),
        ],
        "mix sum",
        _timeout(settings, [timeline_s]),
        work,
    )
    for n in range(len(groups)):
        (work / f"group{n:03d}.wav").unlink(missing_ok=True)
    return len(groups)


def _track_timescale(fps: Any) -> int:
    """A video track timescale that holds one frame exactly (a whole number of ticks), at least
    10,000 ticks a second, the same for every chunk and the joined file."""
    return fps.num * max(1, math.ceil(10000 / fps.num))


def _encode_chunks(
    tl: Timeline,
    hub: dict[str, HubSource],
    chunks: list[Chunk],
    codec: Codec,
    font_name: str | None,
    lut_names: dict[str, str],
    settings: Any,
    work: Path,
) -> list[Path]:
    """Each chunk encoded alone (picture only), checked to hold exactly its frames, starting on
    a key frame, with the same stream setup as the first chunk."""
    fps = tl.plan.fps
    rate = f"{fps.num}/{fps.den}"
    parts: list[Path] = []
    setup: dict[str, Any] | None = None
    for k, chunk in enumerate(chunks):
        graph = video_graph(
            tl,
            hub,
            font=font_name,
            luts=lut_names,
            out_pix_fmt=codec.pix_fmt,
            span=(chunk.start, chunk.end),
        )
        script = len(graph.text.encode("utf-8"))
        if script > MAX_FILTER_SCRIPT_BYTES:
            raise RenderError(
                "reframe_too_complex",
                f"the picture's filter script for record frames {chunk.start}..{chunk.end} is "
                f"{script // 1024} KiB, over the {MAX_FILTER_SCRIPT_BYTES // 1024} KiB limit "
                "(moving reframes take most of it)",
                "use fewer reframe keyframes, or split the segment",
                bytes=script,
            )
        for text in graph.titles:
            (work / text.name).write_text(text.text, encoding="utf-8")
        name = f"chunk{k:03d}.video.ffgraph"
        (work / name).write_text(graph.text, encoding="utf-8")
        part = work / f"part{k:03d}.{codec.part_format}"
        chunk_s = Fraction((chunk.end - chunk.start) * fps.den, fps.num)
        run_ffmpeg(
            [
                *("-v", "error", "-y", "-copyts", *(a for use in graph.uses for a in use.args())),
                *("-filter_complex_script", name, "-map", "[vout]", "-an"),
                *("-r", rate, "-fps_mode", "cfr", *codec.video),
                *("-video_track_timescale", str(_track_timescale(fps))),
                *("-f", codec.part_format, part.name),
            ],
            f"encode {k + 1}/{len(chunks)}" if len(chunks) > 1 else "encode",
            _timeout(settings, [use.span_s for use in graph.uses] + [chunk_s]),
            work,
        )
        (work / name).unlink(missing_ok=True)
        packets = _packets(part, "encode")
        want = chunk.end - chunk.start
        if len(packets.times) != want:
            raise _short_chunk(tl, hub, chunk, len(packets.times), settings, work)
        if not packets.first_key:
            raise _incomplete(f"chunk {k + 1} does not start on a key frame", "encode")
        mine = _stream_setup(part)
        if setup is None:
            setup = mine
        elif mine != setup:
            differ = sorted(key for key in setup if setup[key] != mine.get(key))
            raise _incomplete(
                f"chunk {k + 1}'s stream setup differs from the first chunk's ({', '.join(differ)})"
                ", so they cannot be joined by stream copy",
                "join",
                differ=differ,
            )
        parts.append(part)
    return parts


def _short_chunk(
    tl: Timeline,
    hub: dict[str, HubSource],
    chunk: Chunk,
    got: int,
    settings: Any,
    work: Path,
) -> RenderError:
    """``render_incomplete`` for a chunk that came out short, naming the first of its pictures
    whose source range decodes short on its own (a missing or damaged frame)."""
    fps = tl.plan.fps
    want = chunk.end - chunk.start
    inside = sorted(
        (p for p in tl.pictures if chunk.start <= p.start and p.end <= chunk.end),
        key=lambda p: (p.start, p.lane),
    )
    for pic in inside:
        media = hub[pic.source_id]
        use = picture_use(pic, media, fps)
        start, end = picture_pts(pic, media, fps)
        try:
            result = run_tool(
                "ffmpeg",
                [
                    *BASE_ARGS,
                    *("-v", "error", "-copyts", *use.args()),
                    "-filter_complex",
                    f"[0:{media.video_index}]trim=start_pts={start}:end_pts={end},scale=16:16[v]",
                    *("-map", "[v]", "-f", "framecrc", "-"),
                ],
                timeout_s=_timeout(settings, [use.span_s]),
                cwd=work,
                check=False,
            )
        except MediaError:
            continue
        decoded = sum(
            1 for line in result.stdout.decode("ascii", errors="replace").splitlines()
            if line and not line.startswith("#")
        )  # fmt: skip
        if decoded < pic.frames:
            last = pic.src_start + pic.frames - 1
            return RenderError(
                "render_incomplete",
                f"{pic.location}: source {pic.source_id!r} ({media.path.name}) frames "
                f"{pic.src_start}..{last} decode to {decoded} of {pic.frames} frames (a missing "
                f"or damaged frame), so record frames {chunk.start}..{chunk.end} came out "
                f"{got} frames, not {want}; nothing was shifted to cover it",
                "re-copy or re-encode that file to the hub (or cut around the damage), then "
                "render again",
                stage="encode",
                location=pic.location,
                source=pic.source_id,
                path=str(media.path),
                src_start=pic.src_start,
                src_end=last,
                decoded=decoded,
                expected=pic.frames,
            )
    return _incomplete(
        f"record frames {chunk.start}..{chunk.end} came out {got} frames, not {want}, and no "
        "picture there decodes short on its own",
        "encode",
        start=chunk.start,
        end=chunk.end,
        frames=got,
        expected=want,
    )


def _concat_list(parts: list[Path], chunks: list[Chunk], fps: Any) -> str:
    lines = ["ffconcat version 1.0"]
    for part, chunk in zip(parts, chunks, strict=True):
        seconds = Fraction((chunk.end - chunk.start) * fps.den, fps.num)
        lines += [f"file {part.name}", f"duration {float(seconds):.6f}"]
    return "\n".join(lines) + "\n"


def _join(
    parts: list[Path],
    chunks: list[Chunk],
    codec: Codec,
    container: str,
    fps: Any,
    settings: Any,
    timeline_s: Fraction,
    out: Path,
    work: Path,
) -> None:
    """The chunks joined by stream copy, with the sound muxed once. The concat demuxer places
    each chunk at the sum of the durations before it (in microseconds, so a long join drifts by a
    few); ``setts`` puts every presentation time back on the frame grid and numbers the decode
    times one frame apart, the codec's reorder delay before them (the delay the chunks share), so
    the result is one continuous stream with exact timestamps."""
    tick = _track_timescale(fps) * fps.den // fps.num
    delay = 0
    for part in parts:
        packets = _packets(part, "join")
        first_pts = min(t[0] for t in packets.times)
        delay = max(delay, -(-(first_pts - packets.times[0][1]) // tick))
    (work / "parts.ffconcat").write_text(_concat_list(parts, chunks, fps), encoding="utf-8")
    setts = f"setts=pts=round(PTS/{tick})*{tick}:dts=(N-{delay})*{tick}"
    run_ffmpeg(
        [
            *("-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", "parts.ffconcat"),
            *("-i", "final.wav", "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy"),
            *("-bsf:v", setts, *codec.audio, "-movflags", "+faststart"),
            *("-video_track_timescale", str(_track_timescale(fps))),
            # A movie timescale of 48 kHz states the sound's exact length (its edit list).
            *("-movie_timescale", str(SAMPLE_RATE), "-f", container, out.name),
        ],
        "join",
        _timeout(settings, [timeline_s]),
        work,
    )


def render_plan(
    plan: EditPlan,
    profile: Profile,
    *,
    out: Path,
    preset: Preset = "review",
    media_map: MediaMap | None = None,
    media_root: Path | None = None,
    transcript: Transcript | None = None,
    laughs: LaughMap | None = None,
    overwrite: bool = False,
    memory_budget_mb: float | None = None,
) -> RenderResult:
    """Render ``plan`` to ``out`` (module docstring); a typed :class:`RenderError` on refusal or
    failure. Relative source media resolve against ``media_root``. ``memory_budget_mb``
    replaces the default memory budget (and the profile's ``[render] memory_budget_mb``)."""
    if preset not in ("review", "master"):
        raise ValueError("preset must be 'review' or 'master'")
    out = Path(os.path.abspath(out))
    container = _container(out, preset, profile)
    res = plan.resolution
    if res.width % 2 or res.height % 2:
        raise RenderError(
            "odd_resolution",
            f"the plan's resolution {res.width}x{res.height} has an odd side; H.264 and ProRes "
            "pictures (4:2:0 and 4:2:2) need even dimensions",
            "give the plan an even width and height (e.g. 854x480, not 853x479)",
            width=res.width,
            height=res.height,
        )
    report_file = render_report_path(out)
    try:
        _refuse_existing([out, report_file], overwrite)
    except ExportError as exc:
        raise RenderError.from_export(exc) from None
    try:
        resolve_tool("ffmpeg")
        resolve_tool("ffprobe")
    except MediaToolMissing as exc:
        raise RenderError("tool_missing", str(exc), "install ffmpeg (with ffprobe)") from None
    findings = lint(plan, profile, transcript=transcript, laughs=laughs)
    if not findings.ok:
        errors = [f for f in findings.findings if f.severity == "error"]
        raise RenderError(
            "lint_errors",
            f"the plan has {len(errors)} lint error(s)",
            "fix the lint errors (aar plan lint) before rendering",
            errors=[{"rule": f.rule, "message": f.message} for f in errors],
        )
    warnings = [f for f in findings.findings if f.severity == "warn"]
    used = used_source_ids(plan)
    sources = plan.source_map()
    hub = {
        sid: resolve_source(
            sources[sid],
            plan.fps,
            media_map=media_map,
            media_root=media_root,
        )
        for sid in used
    }
    layout = render_layout(plan, hub, profile)
    tl = build_timeline(layout, hub)
    font = find_font(profile.render.font)
    luts = _luts(plan, used, media_root)
    codec = _codec_args(preset, profile, container)
    if memory_budget_mb is None:
        memory_budget_mb = profile.render.memory_budget_mb
    if memory_budget_mb is not None and memory_budget_mb <= 0:
        raise ValueError("memory_budget_mb must be above 0")
    budget = memory_budget(memory_budget_mb)
    chunks = plan_chunks(tl, hub, budget, max_script_bytes=MAX_FILTER_SCRIPT_BYTES)
    per_pass = mix_groups(len(tl.sounds), budget)
    work = _work_dir(out)
    try:
        with _sigterm_interrupts():
            return _render(
                plan,
                profile,
                tl,
                hub,
                font,
                luts,
                work=work,
                out=out,
                report_file=report_file,
                preset=preset,
                container=container,
                codec=codec,
                budget=budget,
                chunks=chunks,
                per_pass=per_pass,
                transcript=transcript,
                warnings=warnings,
                overwrite=overwrite,
            )
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _render(
    plan: EditPlan,
    profile: Profile,
    tl: Timeline,
    hub: dict[str, HubSource],
    font: FontChoice,
    luts: dict[str, Path],
    *,
    work: Path,
    out: Path,
    report_file: Path,
    preset: Preset,
    container: str,
    codec: Codec,
    budget: Budget,
    chunks: list[Chunk],
    per_pass: int,
    transcript: Transcript | None,
    warnings: list[Finding],
    overwrite: bool,
) -> RenderResult:
    from aar.qa import check as qa_check  # the QA library imports the plan package

    font_name = None
    if font.path is not None:
        font_name = f"font{font.path.suffix.lower()}"
        shutil.copyfile(font.path, work / font_name)
    lut_names: dict[str, str] = {}
    for n, (sid, path) in enumerate(sorted(luts.items())):
        lut_names[sid] = f"lut{n}{path.suffix.lower()}"
        shutil.copyfile(path, work / lut_names[sid])
    settings = profile.render
    samples = sample_at(plan.fps, tl.total_frames)
    mix_passes = _mix(tl, hub, per_pass, settings, work)
    timeline_s = Fraction(tl.total_frames * plan.fps.den, plan.fps.num)
    loudness = normalize(
        work / "mix.wav",
        work / "final.wav",
        kind=plan.kind,
        qa=profile.qa,
        settings=settings,
        samples=samples,
        timeout_s=_timeout(settings, [timeline_s]),
        cwd=work,
    )
    (work / "mix.wav").unlink(missing_ok=True)  # the scratch holds one WAV, not two
    if loudness.output_samples != samples:
        raise RenderError(
            "render_incomplete",
            f"the processed sound holds {loudness.output_samples} samples, not the timeline's "
            f"{samples}",
            "this is a render bug; report the plan",
            stage="loudness",
        )
    parts = _encode_chunks(tl, hub, chunks, codec, font_name, lut_names, settings, work)
    rendered_file = work / f"render.{container}"
    rate = f"{plan.fps.num}/{plan.fps.den}"
    _join(parts, chunks, codec, container, plan.fps, settings, timeline_s, rendered_file, work)
    for part in parts:
        part.unlink(missing_ok=True)
    _check_output(rendered_file, tl.total_frames, Fraction(plan.fps.den, plan.fps.num))
    rendered_file.chmod(0o644)  # as the pair writer's report (and S18's files) get
    try:
        qa = qa_check(
            rendered_file,
            kind=plan.kind,
            profile=profile,
            plan=_qa_plan(plan, hub),
            transcript=transcript,
        )
    except MediaToolMissing as exc:
        raise RenderError("tool_missing", str(exc), "install ffmpeg (with ffprobe)") from None
    qa_dict = qa.model_dump(mode="json")
    qa_dict["file"] = out.name
    if isinstance(qa_dict.get("media"), dict) and "path" in qa_dict["media"]:
        qa_dict["media"]["path"] = out.name
    report: dict[str, Any] = {
        "format": container,
        "preset": preset,
        "file": out.name,
        "sha256": sha256_file(str(rendered_file)).lower(),  # streamed
        "size_bytes": rendered_file.stat().st_size,
        "plan": {
            "project": plan.project,
            "timeline": plan.timeline,
            "kind": plan.kind,
            "sha256": plan_hash(plan),
        },
        "timeline": {
            "fps": rate,
            "total_frames": tl.total_frames,
            "duration_s": round(float(Fraction(tl.total_frames * plan.fps.den, plan.fps.num)), 6),
            "resolution": f"{plan.resolution.width}x{plan.resolution.height}",
            "audio_samples": samples,
            "sample_rate": SAMPLE_RATE,
        },
        "encode": {
            "args": codec.args(container),
            "pix_fmt": codec.pix_fmt,
            "frame_rate": rate,
            "chunks": len(chunks),
        },
        "memory": {
            "budget": budget.as_dict(),
            "chunks": [chunk.as_dict() for chunk in chunks],
            "mix_passes": mix_passes,
            "mix_planned_mb": round(mix_memory_mb(min(per_pass, len(tl.sounds)))),
        },
        "media": [media.as_dict() for media in hub.values()],
        "rendered": rendered(tl, font, {sid: str(path) for sid, path in luts.items()}),
        "loudness": loudness.as_dict(),
        "not_rendered": [item.as_dict() for item in not_rendered(plan, tl.layout, font, luts)],
        "notes": [item.as_dict() for item in notes(tl, font, hub)]
        + (
            [
                {
                    "feature": "limiter",
                    "location": "loudness",
                    "detail": loudness.warning,
                    "action": "check the peaks in the source, or limit them by hand",
                }
            ]
            if loudness.warning
            else []
        ),
        "lint_warnings": [{"rule": f.rule, "message": f.message} for f in warnings],
        "qa": qa_dict,
    }
    body = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    try:
        write_pair([(out, rendered_file), (report_file, body.encode("utf-8"))], overwrite=overwrite)
    except ExportError as exc:
        raise RenderError.from_export(exc) from None
    return RenderResult(out, report_file, report, warnings)
