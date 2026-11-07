"""The producer's working form of a cut: pieces of one recording, in recording frames.

A ``Piece`` is a half-open range ``[rec_in, rec_out)`` of recording frames shown from one
source. Pieces are kept in record order. Two neighbours are a **cut** when the second does not
start where the first ended; a camera switch at the same recording frame is not a cut, because
the audio plays on unbroken. ``keep`` is the index of the keep range a piece came from: the gap
between pieces of one keep range is a trimmed pause, the gap between keep ranges is a human cut.

The two house invariants are checked here and raise ``InvariantError``: pieces are
chronological, and no cut point lies inside a transcript word (the ``no-dialogue-splice``
test, ``aar.plan.rules.speech.word_at``).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from fractions import Fraction

from aar.plan.inputs import Word
from aar.plan.model import EditPlan, Fps, Source
from aar.plan.produce.errors import InvariantError, ProduceInputError
from aar.plan.produce.inputs import CameraSchedule, KeepList
from aar.plan.rules.speech import word_at


@dataclass(frozen=True)
class Piece:
    source: str
    rec_in: int
    rec_out: int
    keep: int

    @property
    def length(self) -> int:
        return self.rec_out - self.rec_in


@dataclass(frozen=True)
class Coverage:
    """Where a source's media lies on the recording clock, in frames."""

    origin: int
    duration: int | None

    def covers(self, rec_in: int, rec_out: int) -> bool:
        return rec_in - self.origin >= 0 and (
            self.duration is None or rec_out - self.origin <= self.duration
        )


def is_cut(prev: Piece, cur: Piece) -> bool:
    """True when ``cur`` does not resume the recording where ``prev`` ended."""
    return cur.rec_in != prev.rec_out


def min_frames(seconds: float, fps: Fps) -> int:
    """Frames needed to last at least ``seconds`` (the ``min-shot`` rule's arithmetic)."""
    return math.ceil(Fraction(str(seconds)) * fps.num / fps.den)


class Words:
    """Transcript words in recording seconds, with the splice tolerance in seconds."""

    def __init__(self, words: Sequence[Word], tolerance_s: float, fps: Fps) -> None:
        self.words = list(words)
        self.tolerance_s = tolerance_s
        self.fps = fps

    def inside(self, rec_frame: int) -> Word | None:
        """The word a cut at ``rec_frame`` would split, as ``no-dialogue-splice`` judges it."""
        return word_at(self.words, self.fps.seconds(rec_frame), self.tolerance_s)

    def speech_between(self, start_s: float, end_s: float) -> Word | None:
        """A word whose core (its span less the tolerance at each end) meets ``(start, end)``."""
        tol = self.tolerance_s
        for word in self.words:
            lo, hi = word.start_s + tol, word.end_s - tol
            if lo < hi and lo < end_s and hi > start_s:
                return word
        return None

    def cores(self, start_s: float, end_s: float) -> list[tuple[float, float]]:
        """The word cores that meet ``(start, end)``, in time order."""
        tol = self.tolerance_s
        found = []
        for word in self.words:
            lo, hi = word.start_s + tol, word.end_s - tol
            if lo < hi and lo < end_s and hi > start_s:
                found.append((lo, hi))
        return sorted(found)


@dataclass(frozen=True)
class KeepFrames:
    start: int
    end: int
    label: str | None


@dataclass(frozen=True)
class Snap:
    """A keep-list edge that fell inside a word and was moved out of it, keeping the word."""

    keep: int
    edge: str  # "in" or "out"
    word: str
    from_frame: int
    to_frame: int


def _step_out(words: Words, frame: int, step: int, limit: int) -> int:
    """Move ``frame`` by ``step`` until it is not inside any word."""
    for _ in range(limit):
        if words.inside(frame) is None:
            return frame
        frame += step
    raise InvariantError(
        "dialogue-splice",
        f"could not move recording frame {frame} out of the transcript's words",
        hint="check the transcript for words that overlap or run for minutes",
    )


def keep_frames(
    keep: KeepList, words: Words, fps: Fps, *, min_shot_s: float
) -> tuple[list[KeepFrames], list[Snap]]:
    """The keep ranges in recording frames, every edge moved out of any word it splits.

    An in point inside a word moves back to the word's start and an out point forward to its
    end, so the word is kept whole. Ranges that then overlap merge (nothing plays twice).
    Each range must last at least ``min_shot_s``, or it could not be folded anywhere.
    """
    limit = max(1, fps.nominal) * 600
    out: list[KeepFrames] = []
    snaps: list[Snap] = []
    for index, item in enumerate(keep.ranges):
        start, end = fps.frames(item.start_s), fps.frames(item.end_s)
        for edge, frame, step in (("in", start, -1), ("out", end, 1)):
            word = words.inside(frame)
            if word is None:
                continue
            moved = _step_out(words, frame, step, limit)
            snaps.append(Snap(index, edge, word.text, frame, moved))
            if edge == "in":
                start = max(0, moved)
            else:
                end = moved
        if out and start < out[-1].end:
            out[-1] = KeepFrames(out[-1].start, max(out[-1].end, end), out[-1].label)
        else:
            out.append(KeepFrames(start, end, item.label))
    need = min_frames(min_shot_s, fps)
    for number, item in enumerate(out):
        if item.end - item.start < need:
            raise ProduceInputError(
                "keep-too-short",
                f"keep range {number} ({fps.seconds(item.start):.3f}-"
                f"{fps.seconds(item.end):.3f} s) lasts {item.end - item.start} frames, under "
                f"min_shot_s ({need} frames)",
                hint="lengthen it, join it to a neighbouring range, or drop it",
            )
    return out, snaps


def same_camera(a: Source, b: Source) -> bool:
    """The ``no-same-camera-jump`` test: one source, or one camera label."""
    return a.id == b.id or (a.camera is not None and a.camera == b.camera)


def resolve_camera(plan: EditPlan, name: str) -> str:
    """A camera name to a source id: the one source with that ``camera`` label, else the
    source with that id."""
    labelled = [s for s in plan.sources if s.camera == name]
    if len(labelled) == 1:
        return labelled[0].id
    if len(labelled) > 1:
        ids = ", ".join(repr(s.id) for s in labelled)
        raise ProduceInputError(
            "ambiguous-camera",
            f"camera {name!r} names more than one source ({ids})",
            hint="give each source its own camera label, or name a source id in the schedule",
        )
    if any(s.id == name for s in plan.sources):
        return name
    known = ", ".join(sorted({s.camera or s.id for s in plan.sources})) or "none"
    raise ProduceInputError(
        "unknown-camera",
        f"camera {name!r} is neither a camera label nor a source id in the stub (known: {known})",
    )


@dataclass(frozen=True)
class ScheduleFrames:
    """The camera schedule in recording frames: ``(start, end, source id)``, chronological.
    ``None`` bounds mean open-ended (the default camera covers everything)."""

    spans: list[tuple[int | None, int | None, str]]


def schedule_frames(
    plan: EditPlan, schedule: CameraSchedule | None, default_camera: str
) -> ScheduleFrames:
    if schedule is None:
        return ScheduleFrames([(None, None, resolve_camera(plan, default_camera))])
    fps = plan.fps
    return ScheduleFrames(
        [
            (fps.frames(span.start_s), fps.frames(span.end_s), resolve_camera(plan, span.camera))
            for span in schedule.spans
        ]
    )


def split(
    start: int,
    end: int,
    keep: int,
    schedule: ScheduleFrames,
    coverage: dict[str, Coverage],
    fps: Fps,
) -> list[Piece]:
    """``[start, end)`` cut into pieces at the schedule's camera changes.

    Every frame must be scheduled, and each piece's source must hold media for it.
    """
    pieces: list[Piece] = []
    cursor = start
    for lo, hi, source in schedule.spans:
        a = start if lo is None else max(start, lo)
        b = end if hi is None else min(end, hi)
        if b <= a:
            continue
        if a > cursor:
            break
        pieces.append(Piece(source, a, b, keep))
        cursor = b
    if cursor < end or not pieces:
        gap_from = fps.seconds(cursor)
        raise ProduceInputError(
            "schedule-gap",
            f"no camera is scheduled for recording {gap_from:.3f} s inside keep range {keep}",
            hint="the camera schedule must cover every kept moment",
        )
    for piece in pieces:
        if not coverage[piece.source].covers(piece.rec_in, piece.rec_out):
            raise ProduceInputError(
                "camera-does-not-cover",
                f"source {piece.source!r} has no media for recording "
                f"{fps.seconds(piece.rec_in):.3f}-{fps.seconds(piece.rec_out):.3f} s",
                hint="schedule another camera there, or fix the source's origin or duration",
            )
    return pieces


def coalesce(pieces: Sequence[Piece]) -> list[Piece]:
    """Merge neighbours of one keep range that play one source unbroken."""
    out: list[Piece] = []
    for piece in pieces:
        if (
            out
            and out[-1].source == piece.source
            and out[-1].keep == piece.keep
            and out[-1].rec_out == piece.rec_in
        ):
            out[-1] = replace(out[-1], rec_out=piece.rec_out)
        else:
            out.append(piece)
    return out


def assert_chronological(pieces: Sequence[Piece], fps: Fps) -> None:
    """Every piece is non-empty and starts at or after the previous one's end."""
    for index, piece in enumerate(pieces):
        if piece.length <= 0:
            raise InvariantError(
                "empty-piece", f"piece {index} at recording frame {piece.rec_in} is empty"
            )
        if index and piece.rec_in < pieces[index - 1].rec_out:
            raise InvariantError(
                "not-chronological",
                f"piece {index} starts at recording {fps.seconds(piece.rec_in):.3f} s, before "
                f"piece {index - 1} ends ({fps.seconds(pieces[index - 1].rec_out):.3f} s)",
            )


def cut_points(pieces: Sequence[Piece]) -> list[tuple[int, str, int]]:
    """``(piece, "in"|"out", recording frame)`` of every edge that is an edit."""
    points: list[tuple[int, str, int]] = []
    for index, piece in enumerate(pieces):
        if index == 0 or is_cut(pieces[index - 1], piece):
            points.append((index, "in", piece.rec_in))
        if index == len(pieces) - 1 or is_cut(piece, pieces[index + 1]):
            points.append((index, "out", piece.rec_out))
    return points


def assert_no_splice(pieces: Sequence[Piece], words: Words) -> None:
    """No edit point lies inside a transcript word."""
    for index, point, frame in cut_points(pieces):
        word = words.inside(frame)
        if word is not None:
            raise InvariantError(
                "dialogue-splice",
                f"the {point} point of piece {index} (recording frame {frame}) falls inside "
                f"the word {word.text!r} ({word.start_s:.3f}-{word.end_s:.3f} s)",
            )
