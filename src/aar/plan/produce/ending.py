"""Transform 4, end on the laugh: the content ends ``end_after_ms`` after the final laugh's onset.

R6: "end hard right after the punchline word or at the first hint of laugh, no dragging laugh".
The final laugh is the latest laugh (at least the end-on-laugh rule's ``min_strength``) whose
onset lies inside the last keep range. The out point goes to the frame nearest onset +
``end_after_ms``; content after it is dropped (with a warning when that holds transcribed words,
saying how much), and the last piece may grow into a trimmed pause of its own keep range.

It never brings back material the human excluded: the new out point stays inside the final
keep range, and the piece that ends there must belong to it (a laugh just after the range's
first frame must not stretch the previous range's last piece across the human's cut). If the
nearest frame would split a word, the nearest frame inside the rule's window (``min_after_ms``
to ``max_after_ms`` after the onset) that does not is used. Without a usable frame, or without a
laugh in the final keep range, the ending stays and a planner warning says why.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from aar.plan.inputs import LaughMap
from aar.plan.model import Fps
from aar.plan.produce.report import PlannerWarning
from aar.plan.produce.timeline import Coverage, KeepFrames, Piece, Words


@dataclass(frozen=True)
class Ending:
    onset_s: float
    from_frame: int  # the out point before (recording frames)
    to_frame: int
    dropped_frames: int  # content removed after the new out point


def _dropped_speech(
    pieces: Sequence[Piece], frame: int, words: Words, fps: Fps, ending: Ending
) -> list[PlannerWarning]:
    """A warning when the content cut after the new out point held transcribed words."""
    cut = [(max(p.rec_in, frame), p.rec_out) for p in pieces if p.rec_out > frame]
    spoken = [
        w
        for w in words.words
        if any(fps.seconds(a) < w.end_s and w.start_s < fps.seconds(b) for a, b in cut)
    ]
    if not spoken:
        return []
    seconds = fps.seconds(ending.dropped_frames)
    return [
        PlannerWarning(
            code="end-dropped-speech",
            message=(
                f"ending on the laugh at {ending.onset_s:.3f} s drops {seconds:.3f} s of kept "
                f"content holding {len(spoken)} transcribed word(s) after it "
                f"({spoken[0].text!r} first)"
            ),
            at_s=round(fps.seconds(frame), 3),
            hint="if those lines matter, end the keep list before them or after a later laugh",
        )
    ]


def end_on_laugh(
    pieces: Sequence[Piece],
    laughs: LaughMap,
    keeps: Sequence[KeepFrames],
    words: Words,
    coverage: dict[str, Coverage],
    fps: Fps,
    *,
    end_after_s: float,
    window_s: tuple[float, float],
    min_strength: float,
    need: int,
) -> tuple[list[Piece], Ending | None, list[PlannerWarning]]:
    pieces = list(pieces)
    last = pieces[-1]
    keep = keeps[last.keep]
    lo_s, hi_s = fps.seconds(keep.start), fps.seconds(keep.end)
    onsets = [
        laugh.start_s
        for laugh in laughs.laughs
        if laugh.strength >= min_strength and lo_s <= laugh.start_s <= hi_s
    ]
    if not onsets:
        return (
            pieces,
            None,
            [
                PlannerWarning(
                    code="no-final-laugh",
                    message=(
                        f"the laugh map has no laugh in the final keep range ({lo_s:.3f}-"
                        f"{hi_s:.3f} s); the content ends at the keep list's out point"
                    ),
                    at_s=round(fps.seconds(last.rec_out), 3),
                    hint="end the keep list on a laugh, or check the laugh map",
                )
            ],
        )
    onset = max(onsets)
    target = fps.frames(onset + end_after_s)
    first, final = fps.frames(onset + window_s[0]), fps.frames(onset + window_s[1])
    candidates = sorted(range(first, final + 1), key=lambda f: (abs(f - target), f)) or [target]
    for frame in candidates:
        kept = [p for p in pieces if p.rec_in < frame]
        if not kept or kept[-1].keep != last.keep or frame > keep.end:
            continue
        if words.inside(frame) is not None:
            continue
        tail = replace(kept[-1], rec_out=frame)
        if not coverage[tail.source].covers(tail.rec_in, tail.rec_out):
            continue
        if tail.length < need and not any(p.keep == tail.keep for p in kept[:-1]):
            continue
        dropped = sum(p.length for p in pieces if p.rec_in >= frame)
        ending = Ending(onset, last.rec_out, frame, dropped + max(0, kept[-1].rec_out - frame))
        return [*kept[:-1], tail], ending, _dropped_speech(pieces, frame, words, fps, ending)
    return (
        pieces,
        None,
        [
            PlannerWarning(
                code="end-not-on-laugh",
                message=(
                    f"no out point {window_s[0] * 1000:g}-{window_s[1] * 1000:g} ms after the "
                    f"final laugh's onset ({onset:.3f} s) keeps every word and the last shot "
                    "long enough; the ending is unchanged"
                ),
                at_s=round(onset, 3),
                hint="end the keep list on the laugh by hand",
            )
        ],
    )
