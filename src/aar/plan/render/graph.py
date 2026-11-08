"""The ffmpeg filter graphs of a render, as text (S18b): deterministic, so a golden can pin them.

The **video graph** trims each picture by frame index (``trim=start_frame:end_frame``, never
seconds), restamps it as frames at the plan's rate (``settb`` + ``setpts=N``), frames it
(:mod:`~aar.plan.render.framing`), and assembles the spine: runs of pictures and black gaps joined
by ``concat``, the runs joined by ``xfade`` at each dissolve (offset = the window's first frame,
duration = its length, both whole frames). Connected clips (V2 and up) go on with ``overlay`` at
their record frames, titles with ``drawtext`` on their record frames (``enable`` on the frame
number). Side files (title text, the font, a LUT) are named relative to the work dir, which is
ffmpeg's working directory, so no path needs filtergraph escaping.

The **mix graph** trims each sound by sample (48 kHz, positions converted from frames exactly
with :func:`sample_at`), applies its gain and fades (``afade``, linear: Resolve's "Cross Fade 0
dB"), and lays the sounds on as few non-overlapping layers as there are overlaps (S18's
``stack_lanes``), each a ``concat`` of silences and sounds to the timeline's exact length, mixed
by summation (``amix`` with ``normalize=0``).
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from aar.plan.export.layout import TitleSpec, stack_lanes
from aar.plan.model import Fps, Resolution
from aar.plan.render.framing import picture_filters
from aar.plan.render.media import HubSource
from aar.plan.render.timeline import Picture, Sound, Timeline

SAMPLE_RATE = 48000
#: The working picture format between filters.
WORK_PIX_FMT = "yuv420p"


def sample_at(fps: Fps, frame: int, rate: int = SAMPLE_RATE) -> int:
    """The audio sample at the start of ``frame``: exact, rounded to the nearest sample."""
    return round(Fraction(frame * fps.den * rate, fps.num))


def seconds(fps: Fps, frames: int) -> str:
    """Frames as seconds for an ffmpeg duration option (microseconds; ffmpeg rounds the value
    back onto the frame)."""
    return f"{Fraction(frames * fps.den, fps.num):.6f}".rstrip("0").rstrip(".") or "0"


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


def _video_inputs(uses: list[int]) -> tuple[list[str], dict[int, list[str]]]:
    """One label per use of each input's video stream (``split`` when it is used more than once:
    each use trims its own frames)."""
    lines: list[str] = []
    labels: dict[int, list[str]] = {}
    for k in sorted(set(uses)):
        count = uses.count(k)
        names = [f"in{k}_{j}" for j in range(count)]
        outs = "".join(f"[{name}]" for name in names)
        lines.append(f"[{k}:v:0]{'null' if count == 1 else f'split={count}'}{outs}")
        labels[k] = names
    return lines, labels


def _picture_chain(
    pic: Picture, label: str, out: str, fps: Fps, res: Resolution, hub: HubSource, lut: str | None
) -> str:
    filters = [
        f"trim=start_frame={pic.src_start}:end_frame={pic.src_start + pic.frames}",
        f"settb={_tb(fps)}",
        f"setpts=N+{pic.start}" if pic.lane else "setpts=N",
    ]
    if hub.sample_aspect is not None:
        filters.append(f"scale={hub.display[0]}:{hub.display[1]},setsar=1")
    if lut is not None:
        filters.append(f"lut3d=file={lut}")
    filters += picture_filters(pic.framing, res)
    filters += ["setsar=1", f"format={WORK_PIX_FMT}"]
    if not pic.lane:
        filters.append(f"fps={_rate(fps)}")
    if pic.fade_in:
        filters.append(f"fade=t=in:start_frame=0:nb_frames={pic.fade_in}")
    return f"[{label}]{','.join(filters)}[{out}]"


def _gap(out: str, frames: int, fps: Fps, res: Resolution) -> str:
    return (
        f"color=c=black:s={res.width}x{res.height}:r={_rate(fps)},trim=end_frame={frames},"
        f"settb={_tb(fps)},setpts=N,format={WORK_PIX_FMT}[{out}]"
    )


def _title_filters(
    title: TitleSpec, number: int, res: Resolution, font: str
) -> tuple[list[str], list[TitleText]]:
    """drawtext per line: lower thirds bottom left, titles and the sting centred."""
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
            f":x='{x}':y={y}:enable='between(n,{title.record_in},{title.record_out - 1})'"
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
) -> VideoGraph:
    """The final encode's filter graph, ending in ``[vout]``. ``luts`` maps a source id to its
    LUT file name in the work dir; ``font`` is the font file name there (None: no titles)."""
    plan = tl.plan
    fps, res = plan.fps, plan.resolution
    pictures = tl.pictures
    lines, labels = _video_inputs([p.input for p in pictures])
    taken: dict[int, int] = {}
    chains: dict[str, str] = {}
    for n, pic in enumerate(pictures):
        j = taken.get(pic.input, 0)
        taken[pic.input] = j + 1
        out = f"p{n}"
        lines.append(
            _picture_chain(
                pic,
                labels[pic.input][j],
                out,
                fps,
                res,
                hub[pic.source_id],
                luts.get(pic.source_id),
            )
        )
        chains[pic.location] = out
    # The spine: runs joined by concat, the runs joined by xfade at each dissolve.
    incoming = {d.window_start: d for d in tl.dissolves}
    spine = sorted((p for p in pictures if p.lane == 0), key=lambda p: p.start)
    runs: list[list[str]] = [[]]
    run_starts: list[int] = [0]
    cursor = 0
    gaps = 0
    for pic in spine:
        if pic.start in incoming and cursor > pic.start:
            runs.append([])
            run_starts.append(pic.start)
        elif pic.start > cursor:
            lines.append(_gap(f"g{gaps}", pic.start - cursor, fps, res))
            runs[-1].append(f"g{gaps}")
            gaps += 1
        runs[-1].append(chains[pic.location])
        cursor = pic.end
    if cursor < tl.total_frames:
        lines.append(_gap(f"g{gaps}", tl.total_frames - cursor, fps, res))
        runs[-1].append(f"g{gaps}")
    run_labels: list[str] = []
    for r, members in enumerate(runs):
        if len(members) == 1:
            run_labels.append(members[0])
            continue
        ins = "".join(f"[{m}]" for m in members)
        lines.append(
            f"{ins}concat=n={len(members)}:v=1:a=0,settb={_tb(fps)},setpts=N,fps={_rate(fps)}[r{r}]"
        )
        run_labels.append(f"r{r}")
    acc = run_labels[0]
    for r in range(1, len(runs)):
        dissolve = incoming[run_starts[r]]
        duration, offset = seconds(fps, dissolve.frames), seconds(fps, dissolve.window_start)
        lines.append(
            f"[{acc}][{run_labels[r]}]xfade=transition=fade:duration={duration}:offset={offset}"
            f"[x{r}]"
        )
        acc = f"x{r}"
    for n, pic in enumerate(
        sorted((p for p in pictures if p.lane), key=lambda p: (p.lane, p.start))
    ):
        lines.append(f"[{acc}][{chains[pic.location]}]overlay=0:0:eof_action=pass[o{n}]")
        acc = f"o{n}"
    titles: list[TitleText] = []
    drawn: list[str] = []
    if font is not None:
        for number, title in enumerate(tl.titles):
            filters, texts = _title_filters(title, number, res, font)
            drawn += filters
            titles += texts
    tail = [*drawn, f"format={out_pix_fmt}"]
    lines.append(f"[{acc}]{','.join(tail)}[vout]")
    return VideoGraph(";\n".join(lines) + "\n", titles)


def _channels(channels: int) -> str:
    """The source's sound as stereo: a mono file on both sides, else its first two channels."""
    return "pan=stereo|c0=c0|c1=c0" if channels == 1 else "pan=stereo|c0=c0|c1=c1"


def _sound_chain(sound: Sound, label: str, out: str, fps: Fps) -> str:
    start = sample_at(fps, sound.src_start)
    length = sample_at(fps, sound.end) - sample_at(fps, sound.start)
    filters = [
        f"atrim=start_sample={start}:end_sample={start + length}",
        "asetpts=PTS-STARTPTS",
    ]
    if sound.gain_db:
        filters.append(f"volume=volume={sound.gain_db:.6f}dB")
    if sound.fade_in:
        n = sample_at(fps, sound.start + sound.fade_in) - sample_at(fps, sound.start)
        filters.append(f"afade=t=in:start_sample=0:nb_samples={n}:curve=tri")
    if sound.fade_out:
        at = sample_at(fps, sound.end - sound.fade_out) - sample_at(fps, sound.start)
        filters.append(f"afade=t=out:start_sample={at}:nb_samples={length - at}:curve=tri")
    filters += [f"apad=whole_len={length}", f"atrim=end_sample={length}"]
    return f"[{label}]{','.join(filters)}[{out}]"


def _silence(out: str, samples: int) -> str:
    return f"anullsrc=r={SAMPLE_RATE}:cl=stereo,atrim=end_sample={samples}[{out}]"


def mix_graph(tl: Timeline, hub: dict[str, HubSource]) -> str:
    """The mix of every sound to the timeline's exact sample length, ending in ``[mix]``."""
    fps = tl.plan.fps
    total = sample_at(fps, tl.total_frames)
    sounds = tl.sounds
    per_input = {s.input: hub[s.source_id].channels or 1 for s in sounds}
    lines: list[str] = []
    labels: dict[int, list[str]] = {}
    for k in sorted(per_input):
        count = sum(1 for s in sounds if s.input == k)
        prep = f"aresample={SAMPLE_RATE},{_channels(per_input[k])}"
        names = [f"a{k}_{j}" for j in range(count)]
        outs = "".join(f"[{name}]" for name in names)
        split = f",asplit={count}" if count > 1 else ""
        lines.append(f"[{k}:a:0]{prep}{split}{outs}")
        labels[k] = names
    taken: dict[int, int] = {}
    chains: list[str] = []
    for n, sound in enumerate(sounds):
        j = taken.get(sound.input, 0)
        taken[sound.input] = j + 1
        lines.append(_sound_chain(sound, labels[sound.input][j], f"s{n}", fps))
        chains.append(f"s{n}")
    layers = stack_lanes([(s.start, s.end) for s in sounds], 0)
    count = max(layers, default=0) + 1
    layer_labels: list[str] = []
    gaps = 0
    for layer in range(count):
        members = sorted(
            (i for i, lane in enumerate(layers) if lane == layer), key=lambda i: sounds[i].start
        )
        pieces: list[str] = []
        cursor = 0
        for i in members:
            at = sample_at(fps, sounds[i].start)
            if at > cursor:
                lines.append(_silence(f"z{gaps}", at - cursor))
                pieces.append(f"z{gaps}")
                gaps += 1
            pieces.append(chains[i])
            cursor = sample_at(fps, sounds[i].end)
        if cursor < total:
            lines.append(_silence(f"z{gaps}", total - cursor))
            pieces.append(f"z{gaps}")
            gaps += 1
        if len(pieces) == 1:
            layer_labels.append(pieces[0])
            continue
        ins = "".join(f"[{p}]" for p in pieces)
        lines.append(f"{ins}concat=n={len(pieces)}:v=0:a=1[l{layer}]")
        layer_labels.append(f"l{layer}")
    if len(layer_labels) == 1:
        lines.append(f"[{layer_labels[0]}]anull[mix]")
    else:
        ins = "".join(f"[{label}]" for label in layer_labels)
        lines.append(f"{ins}amix=inputs={len(layer_labels)}:normalize=0:duration=longest[mix]")
    return ";\n".join(lines) + "\n"
