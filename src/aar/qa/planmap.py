"""Read an EditPlan for QA: its cuts and the transcript words it keeps, in record time.

Pure functions over the v1 ``EditPlan`` API (frames, ``record_ranges()``, ``joins()``).
Transcript times are recording seconds; a source's file frame ``f`` is recording frame
``recording_origin + f``. Record time is the timeline position of the render.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass

from aar.plan.inputs import Transcript, Word
from aar.plan.model import EditPlan, Source


@dataclass(frozen=True)
class Cut:
    """A join that is a real edit: the incoming segment does not continue the outgoing one."""

    number: int  # 1-based, in plan join order
    outgoing: int
    incoming: int
    record_frame: int
    record_s: float
    source_id: str  # the incoming segment's source
    src_in_frame: int
    src_in_s: float  # file time of the incoming src_in
    out_source_id: str
    src_out_frame: int


@dataclass(frozen=True)
class KeptWord:
    """A transcript word the plan keeps, placed in record time."""

    index: int  # position in the transcript
    text: str
    segment: int
    record_start_s: float
    record_end_s: float

    @property
    def record_mid_s(self) -> float:
        return (self.record_start_s + self.record_end_s) / 2


@dataclass(frozen=True)
class StraddledWord:
    """A source word that a cut slices through by more than the tolerance."""

    cut: int  # Cut.number
    side: str  # "out" (end of the outgoing segment) or "in" (start of the incoming one)
    text: str
    start_s: float  # recording seconds
    end_s: float
    boundary_s: float


def _usable(plan: EditPlan, sources: dict[str, Source], index: int) -> bool:
    seg = plan.segments[index]
    return seg.source in sources and seg.length > 0


def is_continuous(
    plan: EditPlan, sources: dict[str, Source], outgoing: int, incoming: int, tolerance: int
) -> bool:
    """Same rule as the linter: same recording and the incoming one resumes within tolerance."""
    out_seg, in_seg = plan.segments[outgoing], plan.segments[incoming]
    out_src, in_src = sources[out_seg.source], sources[in_seg.source]
    if plan.recording_key(out_src) != plan.recording_key(in_src):
        return False
    gap = (in_src.recording_origin + in_seg.src_in) - (out_src.recording_origin + out_seg.src_out)
    return abs(gap) <= tolerance


def cuts(plan: EditPlan, tolerance_frames: int) -> list[Cut]:
    """Every non-continuous join between usable segments, in ``plan.joins()`` order."""
    sources = plan.source_map()
    ranges = plan.record_ranges()
    found: list[Cut] = []
    for outgoing, incoming in plan.joins():
        if not (_usable(plan, sources, outgoing) and _usable(plan, sources, incoming)):
            continue
        if is_continuous(plan, sources, outgoing, incoming, tolerance_frames):
            continue
        seg = plan.segments[incoming]
        out_seg = plan.segments[outgoing]
        found.append(
            Cut(
                number=len(found) + 1,
                outgoing=outgoing,
                incoming=incoming,
                record_frame=ranges[incoming][0],
                record_s=plan.fps.seconds(ranges[incoming][0]),
                source_id=seg.source,
                src_in_frame=seg.src_in,
                src_in_s=plan.fps.seconds(seg.src_in),
                out_source_id=out_seg.source,
                src_out_frame=out_seg.src_out,
            )
        )
    return found


def applies(transcript: Transcript, source: Source) -> bool:
    """A transcript naming a recording covers sources of that recording (or with that id)."""
    if transcript.recording is None:
        return True
    return transcript.recording in (source.recording_id, source.id)


class _WordIndex:
    """Words sorted by start (as a Transcript guarantees), searchable by time."""

    def __init__(self, words: list[Word]) -> None:
        self.words = words
        self.starts = [w.start_s for w in words]
        self.longest = max((w.end_s - w.start_s for w in words), default=0.0)

    def overlapping(self, lo: float, hi: float) -> range:
        """Indexes of words that may overlap ``[lo, hi]`` (a superset; callers filter)."""
        first = bisect_left(self.starts, lo - self.longest)
        last = bisect_left(self.starts, hi, lo=first)
        return range(first, min(last + 1, len(self.words)))


def kept_words(plan: EditPlan, transcript: Transcript) -> list[KeptWord]:
    """Transcript words whose midpoint lies inside a kept segment, in record order.

    A word is kept by segment ``i`` when its midpoint (recording seconds) falls in the
    segment's half-open recording range. Its record time is the segment's record start plus
    the word's offset into the segment.
    """
    sources = plan.source_map()
    ranges = plan.record_ranges()
    fps = plan.fps
    words = transcript.words
    index_of = _WordIndex(words)
    seen: set[tuple[int, int]] = set()
    kept: list[KeptWord] = []
    for i, seg in enumerate(plan.segments):
        source = sources.get(seg.source)
        if source is None or seg.length <= 0 or not applies(transcript, source):
            continue
        r0 = fps.seconds(source.recording_origin + seg.src_in)
        r1 = fps.seconds(source.recording_origin + seg.src_out)
        rec0 = fps.seconds(ranges[i][0])
        for index in index_of.overlapping(r0, r1):
            word = words[index]
            if not (r0 <= (word.start_s + word.end_s) / 2 < r1):
                continue
            start = rec0 + (word.start_s - r0)
            key = (index, ranges[i][0])
            if key in seen:
                continue
            seen.add(key)
            kept.append(
                KeptWord(
                    index=index,
                    text=word.text,
                    segment=i,
                    record_start_s=start,
                    record_end_s=rec0 + (word.end_s - r0),
                )
            )
    kept.sort(key=lambda k: (k.record_start_s, k.index))
    return kept


def _straddles(word: Word, boundary_s: float, tolerance_s: float) -> bool:
    return word.start_s + tolerance_s < boundary_s < word.end_s - tolerance_s


def straddled_words(
    plan: EditPlan, transcript: Transcript, cut: Cut, tolerance_s: float
) -> list[StraddledWord]:
    """Source words the cut slices: the outgoing end or the incoming start falls inside a word
    by more than ``tolerance_s`` on both sides."""
    sources = plan.source_map()
    fps = plan.fps
    index_of = _WordIndex(transcript.words)
    found: list[StraddledWord] = []
    sides = (
        ("out", sources[cut.out_source_id], cut.src_out_frame),
        ("in", sources[cut.source_id], cut.src_in_frame),
    )
    for side, source, frame in sides:
        if not applies(transcript, source):
            continue
        boundary = fps.seconds(source.recording_origin + frame)
        for index in index_of.overlapping(boundary, boundary):
            word = transcript.words[index]
            if _straddles(word, boundary, tolerance_s):
                found.append(
                    StraddledWord(
                        cut=cut.number,
                        side=side,
                        text=word.text,
                        start_s=word.start_s,
                        end_s=word.end_s,
                        boundary_s=boundary,
                    )
                )
    return found
