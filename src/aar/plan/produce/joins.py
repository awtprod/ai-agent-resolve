"""Transform 2, joins in source-audio quiet: every cut's in point starts in a pause.

R6: the in-side of a cut needs a quiet run of at least ``[qa] join_quiet_min_ms`` below
``join_quiet_db`` in the incoming source's audio, containing src_in. The test is QA's own
(``aar.qa.scan_join_quiet``), injected as ``judge``, so a cut placed here passes
``aar qa check``. When the in point fails, it moves up to ``join_nudge_ms`` either way, nearest
first (later on a tie, which adds no removed audio), to the first frame that passes, provided
the move neither adds nor drops a transcribed word, keeps the cut a cut, and leaves the piece
foldable. If no frame passes, the in point stays and a planner warning names the join.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

from aar.plan.model import Fps
from aar.plan.produce.report import PlannerWarning
from aar.plan.produce.timeline import Coverage, Piece, Words, is_cut

#: ``judge(source id, src_in frame)``: does a cut starting there begin in quiet?
Judge = Callable[[str, int], bool]


@dataclass(frozen=True)
class Nudge:
    join: int  # 1-based, in piece order
    source: str
    from_frame: int  # recording frames
    to_frame: int


def _foldable(pieces: Sequence[Piece], index: int, length: int, need: int) -> bool:
    """A piece may end up short only if a neighbour of its keep range can absorb it."""
    if length >= need:
        return True
    keep = pieces[index].keep
    neighbours = [pieces[j] for j in (index - 1, index + 1) if 0 <= j < len(pieces)]
    return any(n.keep == keep for n in neighbours)


def place_joins(
    pieces: Sequence[Piece],
    judge: Judge,
    words: Words,
    coverage: dict[str, Coverage],
    fps: Fps,
    *,
    nudge_frames: int,
    need: int,
) -> tuple[list[Piece], list[Nudge], list[PlannerWarning]]:
    out = list(pieces)
    nudges: list[Nudge] = []
    warnings: list[PlannerWarning] = []
    join = 0
    for index in range(1, len(out)):
        prev, cur = out[index - 1], out[index]
        if not is_cut(prev, cur):
            continue
        join += 1
        origin = coverage[cur.source].origin
        if judge(cur.source, cur.rec_in - origin):
            continue
        offsets = sorted(
            (d for d in range(-nudge_frames, nudge_frames + 1) if d), key=lambda d: (abs(d), -d)
        )
        for delta in offsets:
            at = cur.rec_in + delta
            if not prev.rec_out < at < cur.rec_out:
                continue
            if not coverage[cur.source].covers(at, cur.rec_out):
                continue
            lo, hi = sorted((at, cur.rec_in))
            if words.inside(at) or words.speech_between(fps.seconds(lo), fps.seconds(hi)):
                continue
            if not _foldable(out, index, cur.rec_out - at, need):
                continue
            if judge(cur.source, at - origin):
                out[index] = replace(cur, rec_in=at)
                nudges.append(Nudge(join, cur.source, cur.rec_in, at))
                break
        else:
            warnings.append(
                PlannerWarning(
                    code="join-not-quiet",
                    message=(
                        f"join {join} (recording {fps.seconds(cur.rec_in):.3f} s, source "
                        f"{cur.source!r}, src_in frame {cur.rec_in - origin}) has no quiet "
                        f"in point within {nudge_frames} frames that keeps every word"
                    ),
                    at_s=round(fps.seconds(cur.rec_in), 3),
                    source=cur.source,
                    hint="move the keep range's edge to a sentence boundary with a real pause",
                )
            )
    return out, nudges, warnings
