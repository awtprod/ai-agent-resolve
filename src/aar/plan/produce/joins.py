"""Transform 2, joins in source-audio quiet: both sides of every human cut, the in side of every
cut.

**In side, every cut** (R6): QA's join-quiet test on the incoming source's audio, a run of at
least ``[qa] join_quiet_min_ms`` below ``join_quiet_db`` containing src_in
(``aar.qa.scan_join_quiet``), so a cut placed here passes ``aar qa check``.

**Human cuts** (between keep ranges) check more, because what lies beyond them is material the
human excluded and the crossfade plays a little of it:

* the incoming crossfade handle (``handle`` frames before src_in) lies in that same quiet run;
* the out side: the outgoing source's audio is quiet from ``join_quiet_min_ms`` before src_out
  to the end of the outgoing handle after it. So the kept tail holds no real word tail that
  Whisper ended early, and the fade-out holds no onset of the next removed word. This is
  measured on the audio envelope, never taken from transcript times.

A trim cut lies inside one pause on both sides by construction, so only its in side is judged.

**Nudges.** A failing point moves up to ``join_nudge_ms``, nearest first, to the first frame
that passes, provided the move stays inside its keep range (at a human cut that means inward
only: the in point later, the out point earlier, never into excluded material), neither adds
nor drops a transcribed word, keeps the cut a cut, and leaves the piece foldable. Inside a keep
range an in point may move either way, later on a tie. With no passing frame the point stays
and a planner warning names the join.

**Media handles.** The crossfade needs ``handle`` frames of media before src_in and after
src_out. When the incoming media starts too late, the in point moves to the first frame that
has a handle if that frame passes the in-side test; when the outgoing media ends too early, the
out point moves to the last such frame if it passes the out-side test. Otherwise
``ProduceError("no-handle")`` names the cut.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Protocol

from aar.plan.model import Fps
from aar.plan.produce.errors import ProduceError
from aar.plan.produce.report import PlannerWarning
from aar.plan.produce.timeline import Coverage, KeepFrames, Piece, Words, is_cut


class Judge(Protocol):
    """The audio tests of a cut, in source frames of the source named."""

    def in_quiet(self, source: str, src_in: int, *, handle: int) -> bool:
        """QA's join-quiet test at src_in, with ``handle`` frames before it inside the run."""
        ...

    def out_quiet(self, source: str, src_out: int, *, handle: int) -> bool:
        """Quiet from ``join_quiet_min_ms`` before src_out to ``handle`` frames after it."""
        ...


@dataclass(frozen=True)
class Nudge:
    join: int  # 1-based, in piece order
    side: str  # "in" or "out"
    source: str
    from_frame: int  # recording frames
    to_frame: int


@dataclass(frozen=True)
class _Join:
    number: int
    index: int  # of the incoming piece
    human: bool


def reach_frames(nudge_ms: float, fps: Fps) -> int:
    """How many whole frames a nudge of at most ``nudge_ms`` may move (rounded down)."""
    return math.floor(Fraction(str(nudge_ms)) / 1000 * fps.num / fps.den)


def _foldable(pieces: Sequence[Piece], index: int, length: int, need: int) -> bool:
    """A piece may end up short only if a neighbour of its keep range can absorb it."""
    if length >= need:
        return True
    keep = pieces[index].keep
    neighbours = [pieces[j] for j in (index - 1, index + 1) if 0 <= j < len(pieces)]
    return any(n.keep == keep for n in neighbours)


class _Placer:
    def __init__(
        self,
        pieces: Sequence[Piece],
        judge: Judge,
        words: Words,
        coverage: dict[str, Coverage],
        keeps: Sequence[KeepFrames],
        fps: Fps,
        *,
        nudge_frames: int,
        need: int,
        handle: int,
    ) -> None:
        self.out = list(pieces)
        self.judge = judge
        self.words = words
        self.coverage = coverage
        self.keeps = keeps
        self.fps = fps
        self.nudge_frames = nudge_frames
        self.need = need
        self.handle = handle
        self.nudges: list[Nudge] = []
        self.warnings: list[PlannerWarning] = []

    def _keeps_words(self, a: int, b: int) -> bool:
        """Moving an edge between recording frames ``a`` and ``b`` adds or drops no word."""
        if a == b:
            return True
        lo, hi = sorted((a, b))
        sec = self.fps.seconds
        return self.words.inside(a) is None and self.words.speech_between(sec(lo), sec(hi)) is None

    def _no_handle(self, join: _Join, side: str, source: str, frame: int) -> ProduceError:
        return ProduceError(
            "no-handle",
            f"cut {join.number}: source {source!r} has under {self.handle} frame(s) of media "
            f"{'before its in' if side == 'in' else 'after its out'} point (source frame "
            f"{frame}) for the crossfade, and the nearest frame with a handle does not sit in "
            "source-audio quiet",
            hint="schedule a camera whose media covers the crossfade, or move the keep edge",
        )

    def in_side(self, join: _Join) -> None:
        out, i = self.out, join.index
        prev, cur = out[i - 1], out[i]
        cover = self.coverage[cur.source]
        origin = cover.origin
        pre = self.handle if join.human else 0
        keep_start = self.keeps[cur.keep].start

        def allowed(at: int) -> bool:
            return (
                prev.rec_out < at < cur.rec_out
                and at >= keep_start
                and at - origin >= self.handle
                and self._keeps_words(at, cur.rec_in)
                and _foldable(out, i, cur.rec_out - at, self.need)
            )

        def quiet(at: int) -> bool:
            return self.judge.in_quiet(cur.source, at - origin, handle=pre)

        if cur.rec_in - origin < self.handle:
            at = origin + self.handle
            if not (allowed(at) and quiet(at)):
                raise self._no_handle(join, "in", cur.source, cur.rec_in - origin)
            self._move_in(join, at)
            return
        if quiet(cur.rec_in):
            return
        offsets = sorted(
            (d for d in range(-self.nudge_frames, self.nudge_frames + 1) if d),
            key=lambda d: (abs(d), -d),
        )
        for delta in offsets:
            at = cur.rec_in + delta
            if allowed(at) and quiet(at):
                self._move_in(join, at)
                return
        self.warnings.append(
            self._warning(
                "join-not-quiet",
                join,
                cur.source,
                cur.rec_in,
                f"src_in frame {cur.rec_in - origin}) has no quiet in point within "
                f"{self.nudge_frames} frames that keeps every word"
                + (" and the crossfade's handle" if join.human else ""),
            )
        )

    def out_side(self, join: _Join) -> None:
        out, i = self.out, join.index - 1
        prev = out[i]
        cover = self.coverage[prev.source]
        origin = cover.origin

        def media_ok(at: int) -> bool:
            return cover.duration is None or at - origin + self.handle <= cover.duration

        def allowed(at: int) -> bool:
            return (
                prev.rec_in < at <= prev.rec_out
                and media_ok(at)
                and self._keeps_words(at, prev.rec_out)
                and _foldable(out, i, at - prev.rec_in, self.need)
            )

        def quiet(at: int) -> bool:
            return self.judge.out_quiet(prev.source, at - origin, handle=self.handle)

        if not media_ok(prev.rec_out):
            assert cover.duration is not None
            at = origin + cover.duration - self.handle
            if not (allowed(at) and quiet(at)):
                raise self._no_handle(join, "out", prev.source, prev.rec_out - origin)
            self._move_out(join, at)
            return
        if quiet(prev.rec_out):
            return
        for delta in range(1, self.nudge_frames + 1):
            at = prev.rec_out - delta
            if allowed(at) and quiet(at):
                self._move_out(join, at)
                return
        self.warnings.append(
            self._warning(
                "join-out-not-quiet",
                join,
                prev.source,
                prev.rec_out,
                f"src_out frame {prev.rec_out - origin}) is not followed by quiet through the "
                f"crossfade, or cuts a word's real tail, and no out point up to "
                f"{self.nudge_frames} frames earlier fixes it without dropping a word",
            )
        )

    def _move_in(self, join: _Join, at: int) -> None:
        cur = self.out[join.index]
        self.nudges.append(Nudge(join.number, "in", cur.source, cur.rec_in, at))
        self.out[join.index] = replace(cur, rec_in=at)

    def _move_out(self, join: _Join, at: int) -> None:
        prev = self.out[join.index - 1]
        self.nudges.append(Nudge(join.number, "out", prev.source, prev.rec_out, at))
        self.out[join.index - 1] = replace(prev, rec_out=at)

    def _warning(
        self, code: str, join: _Join, source: str, frame: int, detail: str
    ) -> PlannerWarning:
        at = self.fps.seconds(frame)
        kind = "human cut" if join.human else "cut"
        return PlannerWarning(
            code=code,
            message=f"join {join.number} ({kind} at recording {at:.3f} s, source {source!r}, "
            + detail,
            at_s=round(at, 3),
            source=source,
            hint="move the keep range's edge to a sentence boundary with a real pause",
        )


def place_joins(
    pieces: Sequence[Piece],
    judge: Judge,
    words: Words,
    coverage: dict[str, Coverage],
    keeps: Sequence[KeepFrames],
    fps: Fps,
    *,
    nudge_frames: int,
    need: int,
    handle: int,
) -> tuple[list[Piece], list[Nudge], list[PlannerWarning]]:
    placer = _Placer(
        pieces,
        judge,
        words,
        coverage,
        keeps,
        fps,
        nudge_frames=nudge_frames,
        need=need,
        handle=handle,
    )
    number = 0
    for index in range(1, len(placer.out)):
        prev, cur = placer.out[index - 1], placer.out[index]
        if not is_cut(prev, cur):
            continue
        number += 1
        join = _Join(number, index, prev.keep != cur.keep)
        placer.in_side(join)
        if join.human:
            placer.out_side(join)
    return placer.out, placer.nudges, placer.warnings
