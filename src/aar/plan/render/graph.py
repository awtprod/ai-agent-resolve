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

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from aar.media import input_url
from aar.plan.export.layout import TitleSpec
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
    return f"{max(value, Fraction(0)):.6f}"


def picture_use(pic: Picture, hub: HubSource, fps: Fps) -> Use:
    """The input for one picture: seeked half a frame before its first frame, so the seek keeps
    exactly that frame first whatever the timestamps' rounding."""
    frame = Fraction(fps.den, fps.num)
    seek = None
    if pic.src_start > 0:
        seek = hub.video_offset_s + (pic.src_start - Fraction(1, 2)) * frame
    return Use(hub.path, hub.video_index, seek, pic.frames * frame + SEEK_SLACK_S)


def _picture_chain(
    pic: Picture, label: str, out: str, fps: Fps, res: Resolution, hub: HubSource, lut: str | None
) -> str:
    filters = [
        f"trim=end_frame={pic.frames}",  # the seek kept frame src_start first
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
    lines: list[str] = []
    uses: list[Use] = []
    chains: dict[str, str] = {}
    for n, pic in enumerate(pictures):
        media = hub[pic.source_id]
        uses.append(picture_use(pic, media, fps))
        out = f"p{n}"
        label = f"{n}:{media.video_index}"
        lines.append(_picture_chain(pic, label, out, fps, res, media, luts.get(pic.source_id)))
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
    return VideoGraph(";\n".join(lines) + "\n", titles, uses)


def _channels(channels: int) -> str:
    """The source's sound as stereo: a mono file on both sides, else its first two channels."""
    return "pan=stereo|c0=c0|c1=c0" if channels == 1 else "pan=stereo|c0=c0|c1=c1"


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
                 lead: int, total: int) -> str:  # fmt: skip
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
    position = sample_at(fps, sound.start)
    if position:
        filters.append(f"adelay=delays={position}S:all=1")
    filters += [f"apad=whole_len={total}", f"atrim=end_sample={total}"]
    return f"[{label}]{','.join(filters)}[{out}]"


def _channels(channels: int) -> str:
    """The source's sound as stereo: a mono file on both sides, else its first two channels."""
    return "pan=stereo|c0=c0|c1=c0" if channels == 1 else "pan=stereo|c0=c0|c1=c1"


def mix_graph(tl: Timeline, hub: dict[str, HubSource]) -> MixGraph:
    """The mix of every sound to the timeline's exact sample length, ending in ``[mix]``. Run it
    with ``-copyts`` (each sound is cut by its timestamps)."""
    fps = tl.plan.fps
    total = sample_at(fps, tl.total_frames)
    lines: list[str] = []
    uses: list[Use] = []
    for n, sound in enumerate(tl.sounds):
        media = hub[sound.source_id]
        use, first, lead = sound_use(sound, media, fps)
        uses.append(use)
        label = f"{n}:{use.stream}"
        lines.append(
            _sound_chain(sound, label, f"s{n}", fps, media.channels or 1, first, lead, total)
        )
    count = len(tl.sounds)
    if count == 0:
        lines.append(f"anullsrc=r={SAMPLE_RATE}:cl=stereo,atrim=end_sample={total}[mix]")
    elif count == 1:
        lines.append("[s0]anull[mix]")
    else:
        ins = "".join(f"[s{n}]" for n in range(count))
        lines.append(f"{ins}amix=inputs={count}:normalize=0:duration=longest[mix]")
    return MixGraph(";\n".join(lines) + "\n", uses)
