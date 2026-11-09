"""The ffmpeg filter graphs of a render, as text (S18b): deterministic, so a golden can pin them.

Every picture and every sound is its own ffmpeg input (round 2): the file opened again with an
accurate input seek just before the range it plays (``-ss``, half a frame before the first frame
for a picture, a quarter second before the first sample for a sound), so nothing is decoded
from the file's start, no decoded frames queue between two uses of one file, and each use is
cut exactly where it begins. Each input decodes on one thread (``-threads 1``): an open decoder
holds a few frames, so the open inputs' memory is bounded and estimated before a render.

The **video graph** cuts each picture by frame count after its seek (``trim=end_frame``, never
seconds), restamps it as frames at the plan's rate (``settb`` + ``setpts=N``), frames it
(:mod:`~aar.plan.render.framing`), and assembles the spine: runs of pictures and black gaps joined
by ``concat``, the runs joined by ``xfade`` at each dissolve (offset = the window's first frame,
duration = its length, both whole frames). Connected clips (V2 and up) go on with ``overlay`` at
their record frames, the higher track on top, titles with ``drawtext`` on their record frames
(``enable`` on the frame number). Side files (title text, the font, a LUT) are named relative to
the work dir, which is ffmpeg's working directory, so no path needs filtergraph escaping.

The **mix graph** runs with ``-copyts``, so each sound is cut by timestamp (``atrim`` on
``start_pts``/``end_pts`` at 48 kHz) at the absolute time its first frame is shown: a source whose
audio starts after its video stays in sync, and an audio start after the sound's own start is
silence. Each sound gets its gain and fades (``afade``, linear: Resolve's "Cross Fade 0 dB"), is
delayed to its record position (``adelay``) and padded to the timeline's exact length (``apad``
+ ``atrim``), and all are summed (``amix`` with ``normalize=0``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from aar.media import input_url
from aar.plan.export.layout import TitleSpec, stack_lanes
from aar.plan.model import Fps, Resolution
from aar.plan.render.fixed import decimal
from aar.plan.render.framing import picture_filters
from aar.plan.render.media import HubSource
from aar.plan.render.timeline import Picture, Sound, Timeline

SAMPLE_RATE = 48000
#: The working picture format between filters, and a connected lane's (with alpha).
WORK_PIX_FMT = "yuv420p"
LANE_PIX_FMT = "yuva420p"


def sample_at(fps: Fps, frame: int, rate: int = SAMPLE_RATE) -> int:
    """The audio sample at the start of ``frame``: exact, rounded to the nearest sample."""
    return round(Fraction(frame * fps.den * rate, fps.num))


def seconds(fps: Fps, frames: int) -> str:
    """Frames as seconds for an ffmpeg duration option (microseconds; ffmpeg rounds the value
    back onto the frame)."""
    return decimal(Fraction(frames * fps.den, fps.num), 6).rstrip("0").rstrip(".") or "0"


def _rate(fps: Fps) -> str:
    return f"{fps.num}/{fps.den}"


def _tb(fps: Fps) -> str:
    return f"{fps.den}/{fps.num}"


@dataclass(frozen=True)
class TitleText:
    """One title line: the text file drawtext reads (in the work dir) and the text."""

    name: str
    text: str


@dataclass(frozen=True)
class VideoGraph:
    text: str
    titles: list[TitleText]
    uses: list[Use]


@dataclass(frozen=True)
class Use:
    """One ffmpeg input: a hub file opened for one picture or one sound."""

    path: Path
    stream: int  # the absolute stream index the graph reads
    seek_s: Fraction | None  # input -ss, from the file's start; None: from its start
    span_s: Fraction  # the source seconds it decodes (the timeout scales with these)

    def args(self) -> list[str]:
        seek = [] if self.seek_s is None else ["-ss", _clock(self.seek_s)]
        return ["-threads", "1", *seek, "-i", input_url(self.path)]


#: Seconds of source a seek may decode before the range (to the keyframe before it).
SEEK_SLACK_S = Fraction(5)
#: How far before a sound's first sample its input seeks (codec pre-roll, then cut by time).
AUDIO_PREROLL_S = Fraction(1, 4)


def _clock(value: Fraction) -> str:
    return decimal(max(value, Fraction(0)), 6)  # never a format spec: 3.11's Fraction has none


def picture_use(pic: Picture, hub: HubSource, fps: Fps) -> Use:
    """The input for one picture: seeked half a frame before its first frame, so the seek keeps
    exactly that frame first whatever the timestamps' rounding."""
    frame = Fraction(fps.den, fps.num)
    seek = None
    if pic.src_start > 0:
        seek = hub.video_offset_s + (pic.src_start - Fraction(1, 2)) * frame
    return Use(hub.path, hub.video_index, seek, pic.frames * frame + SEEK_SLACK_S)


def picture_pts(pic: Picture, hub: HubSource, fps: Fps) -> tuple[int, int]:
    """The presentation timestamps (the video stream's ticks, as ``-copyts`` keeps them) that
    bound the picture's source frames: from half a frame before its first frame to half a frame
    before the one after its last. A frame missing from the file leaves a hole, not a shift."""
    frame = Fraction(fps.den, fps.num)
    tb = hub.video_time_base
    assert tb, "a probed video stream has a time base"

    def tick(k: Fraction) -> int:
        return math.ceil((hub.video_start_s + (k - Fraction(1, 2)) * frame) / tb)

    return tick(Fraction(pic.src_start)), tick(Fraction(pic.src_start + pic.frames))


def _picture_chain(
    pic: Picture,
    label: str,
    out: str,
    fps: Fps,
    res: Resolution,
    hub: HubSource,
    lut: str | None,
) -> str:
    start, end = picture_pts(pic, hub, fps)
    filters = [
        f"trim=start_pts={start}:end_pts={end}",  # by timestamp: a missing frame stays missing
        f"settb={_tb(fps)}",
        "setpts=N",  # then counted: everything after this joins by frame count
    ]
    if hub.sample_aspect is not None:
        filters.append(f"scale={hub.display[0]}:{hub.display[1]},setsar=1")
    if lut is not None:
        filters.append(f"lut3d=file={lut}")
    filters += picture_filters(pic.framing, res)
    filters += ["setsar=1", f"format={WORK_PIX_FMT}"]
    if pic.fade_in:
        filters.append(f"fade=t=in:start_frame=0:nb_frames={pic.fade_in}")
    if pic.lane:
        filters.append(f"format={LANE_PIX_FMT}")  # opaque, over the lane's transparent gaps
    return f"[{label}]{','.join(filters)}[{out}]"


def _gap(out: str, frames: int, fps: Fps, res: Resolution, *, clear: bool = False) -> str:
    """Black frames (a gap in the spine), or transparent ones (a gap in a connected lane)."""
    colour, fmt = ("black@0", LANE_PIX_FMT) if clear else ("black", WORK_PIX_FMT)
    return (
        f"color=c={colour}:s={res.width}x{res.height}:r={_rate(fps)},format={fmt},"
        f"trim=end_frame={frames},settb={_tb(fps)},setpts=N[{out}]"
    )


def _join(members: list[str], out: str, fps: Fps) -> str:
    ins = "".join(f"[{m}]" for m in members)
    return f"{ins}concat=n={len(members)}:v=1:a=0,settb={_tb(fps)},setpts=N[{out}]"


def _title_filters(
    title: TitleSpec, number: int, res: Resolution, font: str, span: tuple[int, int]
) -> tuple[list[str], list[TitleText]]:
    """drawtext per line: lower thirds bottom left, titles and the sting centred; on the frames
    of ``span`` (a chunk of the timeline) that the title covers, counted from the span's start."""
    first = max(title.record_in, span[0]) - span[0]
    last = min(title.record_out, span[1]) - 1 - span[0]
    base = min(res.width, res.height)
    sizes = [round(base * (0.075 if j == 0 else 0.055)) for j in range(len(title.lines))]
    steps = [round(size * 1.4) for size in sizes]
    block = sum(steps)
    if title.kind == "lower-third":
        top = res.height - round(res.height * 0.12) - block
        x = str(round(res.width * 0.06))
    else:
        top = (res.height - block) // 2
        x = "(w-text_w)/2"
    filters: list[str] = []
    texts: list[TitleText] = []
    y = top
    for j, (line, size) in enumerate(zip(title.lines, sizes, strict=True)):
        name = f"title{number}_{j}.txt"
        texts.append(TitleText(name, line))
        filters.append(
            f"drawtext=fontfile={font}:textfile={name}:expansion=none:fontcolor=white"
            f":fontsize={size}:box=1:boxcolor=black@0.55:boxborderw={max(2, size // 5)}"
            f":x='{x}':y={y}:enable='between(n,{first},{last})'"
        )
        y += steps[j]
    return filters, texts


def video_graph(
    tl: Timeline,
    hub: dict[str, HubSource],
    *,
    font: str | None,
    luts: dict[str, str],
    out_pix_fmt: str,
    span: tuple[int, int] | None = None,
) -> VideoGraph:
    """The picture graph of ``span`` (record frames ``[start, end)``, a chunk; the whole
    timeline by default), ending in ``[vout]`` with exactly ``end - start`` frames. Its pictures
    are those that lie inside the span (a chunk is cut only where no picture, dissolve, fade or
    connected clip crosses). Run it with ``-copyts``: each picture is cut by its timestamps.
    ``luts`` maps a source id to its LUT file name in the work dir; ``font`` is the font file
    name there (None: no titles)."""
    plan = tl.plan
    fps, res = plan.fps, plan.resolution
    b0, b1 = span or (0, tl.total_frames)
    pictures = [p for p in tl.pictures if b0 <= p.start and p.end <= b1]
    lines: list[str] = []
    uses: list[Use] = []
    chains: dict[str, str] = {}
    for n, pic in enumerate(pictures):
        media = hub[pic.source_id]
        uses.append(picture_use(pic, media, fps))
        out = f"p{n}"
        label = f"{n}:{media.video_index}"
        lut = luts.get(pic.source_id)
        lines.append(_picture_chain(pic, label, out, fps, res, media, lut))
        chains[pic.location] = out
    # The spine: runs of pictures and black gaps (concat), each dissolve a blend of the
    # outgoing run's tail and the incoming run's head, all joined by one concat.
    incoming = {d.window_start: d for d in tl.dissolves if b0 <= d.window_start < b1}
    spine = sorted((p for p in pictures if p.lane == 0), key=lambda p: p.start)
    runs: list[list[str]] = [[]]
    bounds: list[list[int]] = [[b0, b0]]
    gaps = 0
    for pic in spine:
        cursor = bounds[-1][1]
        if pic.start in incoming and cursor > pic.start:
            runs.append([])
            bounds.append([pic.start, pic.start])
        elif pic.start > cursor:
            lines.append(_gap(f"g{gaps}", pic.start - cursor, fps, res))
            runs[-1].append(f"g{gaps}")
            gaps += 1
        runs[-1].append(chains[pic.location])
        bounds[-1][1] = pic.end
    if bounds[-1][1] < b1:
        lines.append(_gap(f"g{gaps}", b1 - bounds[-1][1], fps, res))
        runs[-1].append(f"g{gaps}")
        gaps += 1
        bounds[-1][1] = b1
    windows = [0] + [incoming[start].frames for start, _ in bounds[1:]]
    for r in range(1, len(runs)):
        if bounds[r - 1][1] - bounds[r][0] != windows[r]:  # S18's handles cover the window
            raise ValueError(f"dissolve at record frame {bounds[r][0]}: handles do not match")
    pieces: list[str] = []
    tails: list[str] = []
    for r, members in enumerate(runs):
        run = members[0] if len(members) == 1 else f"r{r}"
        if len(members) > 1:
            lines.append(_join(members, run, fps))
        length = bounds[r][1] - bounds[r][0]
        head = windows[r]
        tail = windows[r + 1] if r + 1 < len(runs) else 0
        parts = [
            (name, a, b)
            for name, a, b in (
                (f"h{r}", 0, head),
                (f"b{r}", head, length - tail),
                (f"t{r}", length - tail, length),
            )
            if b > a
        ]
        if len(parts) > 1:
            outs = "".join(f"[{run}_{k}]" for k in range(len(parts)))
            lines.append(f"[{run}]split={len(parts)}{outs}")
        for k, (name, a, b) in enumerate(parts):
            source = run if len(parts) == 1 else f"{run}_{k}"
            if len(parts) == 1:
                lines.append(f"[{source}]null[{name}]")
            else:
                lines.append(
                    f"[{source}]trim=start_frame={a}:end_frame={b},settb={_tb(fps)},"
                    f"setpts=N[{name}]"
                )
        if head:
            d = windows[r]
            lines.append(
                f"[{tails[-1]}][h{r}]blend=all_expr='A+(B-A)*T*{fps.num}/({fps.den}*{d})+0.5'"
                f":shortest=1,settb={_tb(fps)},setpts=N[d{r}]"
            )
            pieces.append(f"d{r}")
        if any(name == f"b{r}" for name, _, _ in parts):
            pieces.append(f"b{r}")
        if tail:
            tails.append(f"t{r}")
    if len(pieces) == 1:
        acc = pieces[0]
    else:
        lines.append(_join(pieces, "spine", fps))
        acc = "spine"
    # Connected clips: each lane a full-length stream (transparent gaps), laid over with
    # shortest=1, so a lane short of a frame leaves the chunk short.
    lanes = sorted({p.lane for p in pictures if p.lane})
    for lane in lanes:
        members = []
        cursor = b0
        for pic in sorted((p for p in pictures if p.lane == lane), key=lambda p: p.start):
            if pic.start > cursor:
                lines.append(_gap(f"g{gaps}", pic.start - cursor, fps, res, clear=True))
                members.append(f"g{gaps}")
                gaps += 1
            members.append(chains[pic.location])
            cursor = pic.end
        if cursor < b1:
            lines.append(_gap(f"g{gaps}", b1 - cursor, fps, res, clear=True))
            members.append(f"g{gaps}")
            gaps += 1
        label = f"lane{lane}"
        lines.append(
            _join(members, label, fps) if len(members) > 1 else f"[{members[0]}]null[{label}]"
        )
        lines.append(f"[{acc}][{label}]overlay=0:0:shortest=1[o{lane}]")
        acc = f"o{lane}"
    titles: list[TitleText] = []
    drawn: list[str] = []
    if font is not None:
        for number, title in enumerate(tl.titles):
            if title.record_out <= b0 or title.record_in >= b1:
                continue
            filters, texts = _title_filters(title, number, res, font, (b0, b1))
            drawn += filters
            titles += texts
    tail = [*drawn, f"settb={_tb(fps)}", "setpts=N", f"format={out_pix_fmt}"]
    lines.append(f"[{acc}]{','.join(tail)}[vout]")
    return VideoGraph(";\n".join(lines) + "\n", titles, uses)


@dataclass(frozen=True)
class MixGraph:
    text: str
    uses: list[Use]


def sound_use(sound: Sound, hub: HubSource, fps: Fps) -> tuple[Use, int, int]:
    """The input for one sound, and its first sample and the silence before the audio begins.

    The sound starts at the absolute time its first source frame is shown (the video stream's
    start plus the frames), which is a sample index in the copied timestamps at 48 kHz. When the
    file's audio starts later than that, the gap is silence (``lead`` samples)."""
    shown = hub.video_start_s + Fraction(sound.src_start * fps.den, fps.num)
    first = round(shown * SAMPLE_RATE)
    lead = max(0, round(hub.audio_start_s * SAMPLE_RATE) - first)
    length = Fraction(sound.end - sound.start) * fps.den / fps.num
    seek = shown - hub.start_s - AUDIO_PREROLL_S
    use = Use(
        hub.path,
        hub.audio_index if hub.audio_index is not None else 0,
        seek if seek > 0 else None,
        length + AUDIO_PREROLL_S,
    )
    return use, first, lead


def _sound_chain(sound: Sound, label: str, out: str, fps: Fps, channels: int, first: int,
                 lead: int) -> str:  # fmt: skip
    """One sound, exactly its own length: cut by timestamp, silence before a late audio start,
    its gain and fades. It is placed on the timeline by its layer (:func:`mix_graph`)."""
    length = sample_at(fps, sound.end) - sample_at(fps, sound.start)
    filters = [
        f"aresample={SAMPLE_RATE}",
        _channels(channels),
        f"atrim=start_pts={first}:end_pts={first + length}",
        "asetpts=PTS-STARTPTS",
    ]
    if lead:
        filters.append(f"adelay=delays={lead}S:all=1")
    filters += [f"apad=whole_len={length}", f"atrim=end_sample={length}"]
    if sound.gain_db:
        filters.append(f"volume=volume={sound.gain_db:.6f}dB")
    if sound.fade_in:
        n = sample_at(fps, sound.start + sound.fade_in) - sample_at(fps, sound.start)
        filters.append(f"afade=t=in:start_sample=0:nb_samples={n}:curve=tri")
    if sound.fade_out:
        at = sample_at(fps, sound.end - sound.fade_out) - sample_at(fps, sound.start)
        filters.append(f"afade=t=out:start_sample={at}:nb_samples={length - at}:curve=tri")
    return f"[{label}]{','.join(filters)}[{out}]"


def _silence(out: str, samples: int) -> str:
    return f"anullsrc=r={SAMPLE_RATE}:cl=stereo,atrim=end_sample={samples}[{out}]"


def _channels(channels: int) -> str:
    """The source's sound as stereo: a mono file on both sides, else its first two channels."""
    return "pan=stereo|c0=c0|c1=c0" if channels == 1 else "pan=stereo|c0=c0|c1=c1"


def _layered(lines: list[str], items: list[tuple[str, int, int]], total: int, tag: str) -> None:
    """Place ``(label, first sample, end sample)`` streams, each exactly its own length, on a
    timeline of ``total`` samples, ending in ``[mix]``.

    The items go on as few layers as there are overlaps (S18's ``stack_lanes``: two or three for
    an edit with crossfades), each layer a ``concat`` of silences and items to the full length,
    and the layers are summed (``amix`` with ``normalize=0``; every layer is the same length, so
    none ends early and no gain changes). The work is the layers times the timeline plus the
    items: linear in the timeline, not the items times it (round 3)."""
    layers = stack_lanes([(a, b) for _, a, b in items], 0)
    labels: list[str] = []
    gaps = 0
    for layer in range(max(layers, default=0) + 1):
        members = sorted((i for i, n in enumerate(layers) if n == layer), key=lambda i: items[i][1])
        pieces: list[str] = []
        cursor = 0
        for i in members:
            label, first, end = items[i]
            if first > cursor:
                lines.append(_silence(f"{tag}z{gaps}", first - cursor))
                pieces.append(f"{tag}z{gaps}")
                gaps += 1
            pieces.append(label)
            cursor = end
        if cursor < total or not pieces:
            lines.append(_silence(f"{tag}z{gaps}", total - cursor))
            pieces.append(f"{tag}z{gaps}")
            gaps += 1
        if len(pieces) == 1:
            labels.append(pieces[0])
            continue
        ins = "".join(f"[{piece}]" for piece in pieces)
        lines.append(f"{ins}concat=n={len(pieces)}:v=0:a=1[{tag}l{layer}]")
        labels.append(f"{tag}l{layer}")
    if len(labels) == 1:
        lines.append(f"[{labels[0]}]anull[mix]")
    else:
        ins = "".join(f"[{label}]" for label in labels)
        lines.append(f"{ins}amix=inputs={len(labels)}:normalize=0:duration=longest[mix]")


def mix_graph(
    tl: Timeline,
    hub: dict[str, HubSource],
    sounds: list[int] | None = None,
    span: tuple[int, int] | None = None,
) -> MixGraph:
    """The mix of ``sounds`` (indices into the timeline's sounds; all by default) over the
    samples ``span`` (the whole timeline by default), ending in ``[mix]`` with exactly that many
    samples. Run it with ``-copyts`` (each sound is cut by its timestamps). Each sound has its own
    input, so no branch of a split input is queued ahead of the sum (the round-1 loss)."""
    fps = tl.plan.fps
    first, last = span or (0, sample_at(fps, tl.total_frames))
    picked = [tl.sounds[i] for i in (range(len(tl.sounds)) if sounds is None else sounds)]
    lines: list[str] = []
    uses: list[Use] = []
    items: list[tuple[str, int, int]] = []
    for n, sound in enumerate(picked):
        media = hub[sound.source_id]
        use, at, lead = sound_use(sound, media, fps)
        uses.append(use)
        chain = _sound_chain(
            sound, f"{n}:{use.stream}", f"s{n}", fps, media.channels or 1, at, lead
        )
        lines.append(chain)
        items.append(
            (f"s{n}", sample_at(fps, sound.start) - first, sample_at(fps, sound.end) - first)
        )
    _layered(lines, items, last - first, "")
    return MixGraph(";\n".join(lines) + "\n", uses)


def sum_graph(parts: list[tuple[int, int]], total: int) -> str:
    """The mix of part mixes (inputs ``0..``, WAVs of samples ``[first, end)`` each) to
    ``total`` samples, ending in ``[mix]``: the same layering, so the sum is linear too."""
    lines = [f"[{n}:a]asetpts=PTS-STARTPTS[w{n}]" for n in range(len(parts))]
    _layered(lines, [(f"w{n}", a, b) for n, (a, b) in enumerate(parts)], total, "")
    return ";\n".join(lines) + "\n"
