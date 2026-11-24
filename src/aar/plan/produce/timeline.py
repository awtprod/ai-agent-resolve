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

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from fractions import Fraction

from aar.plan.inputs import Word
from aar.plan.model import EditPlan, Fps
from aar.plan.produce.errors import InvariantError, ProduceInputError
from aar.plan.produce.inputs import CameraSchedule, KeepList
from aar.plan.produce.report import PlannerWarning
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
    """Where a source's media lies on the recording clock, in frames. ``camera`` is the label of
    a camera of several files (S24a) the source is one part of; None otherwise."""

    origin: int
    duration: int | None
    camera: str | None = None

    @property
    def end(self) -> int | None:
        return None if self.duration is None else self.origin + self.duration

    def covers(self, rec_in: int, rec_out: int) -> bool:
        return rec_in - self.origin >= 0 and (
            self.duration is None or rec_out - self.origin <= self.duration
        )


def is_cut(prev: Piece, cur: Piece) -> bool:
    """True when ``cur`` does not resume the recording where ``prev`` ended."""
    return cur.rec_in != prev.rec_out


#: Slack, in frames, when rounding a float frame time (e.g. 581 * 1001 / 30000) to its frame.
_FRAME_SLACK = Fraction(1, 1000)


def inward_frame(seconds: float, fps: Fps, *, up: bool) -> int:
    """The frame at or after (``up``) or at or before ``seconds``: a keep edge rounded inward, so
    a range never takes in even part of a frame the human excluded."""
    exact = Fraction(repr(seconds)) * fps.num / fps.den
    return math.ceil(exact - _FRAME_SLACK) if up else math.floor(exact + _FRAME_SLACK)


def min_frames(seconds: float, fps: Fps) -> int:
    """Frames needed to last at least ``seconds`` (the ``min-shot`` rule's arithmetic)."""
    return math.ceil(Fraction(str(seconds)) * fps.num / fps.den)


class Words:
    """Transcript words in recording seconds, with the splice tolerance in seconds.

    Two views of a word. ``inside`` is the linter's splice test (``word_at``): a cut point may
    not fall strictly inside the word less ``tolerance_s`` at each end. ``speech_between`` and
    ``spans`` are what no trim, nudge or fold may remove: the word's span less the tolerance,
    but at least ``min_word_s`` long around its middle, so a zero-length or 15 ms word that
    Whisper places in a pause still occupies the pause.
    """

    def __init__(
        self, words: Sequence[Word], tolerance_s: float, fps: Fps, min_word_s: float = 0.04
    ) -> None:
        self.words = list(words)
        self.tolerance_s = tolerance_s
        self.fps = fps
        self.min_word_s = min_word_s
        occupied: list[tuple[float, float, Word]] = []
        for word in self.words:
            half = max((word.end_s - word.start_s) / 2 - tolerance_s, min_word_s / 2)
            if half > 0:
                middle = (word.start_s + word.end_s) / 2
                occupied.append((middle - half, middle + half, word))
        self._spans = sorted(occupied, key=lambda item: (item[0], item[1]))

    def inside(self, rec_frame: int) -> Word | None:
        """The word a cut at ``rec_frame`` would split, as ``no-dialogue-splice`` judges it."""
        return word_at(self.words, self.fps.seconds(rec_frame), self.tolerance_s)

    def speech_between(self, start_s: float, end_s: float) -> Word | None:
        """A word whose occupied span meets ``(start, end)``."""
        for lo, hi, word in self._spans:
            if lo < end_s and hi > start_s:
                return word
        return None

    def spans(self, start_s: float, end_s: float) -> list[tuple[float, float]]:
        """The occupied spans that meet ``(start, end)``, in time order."""
        return [(lo, hi) for lo, hi, _ in self._spans if lo < end_s and hi > start_s]

    def occupied(self, start_s: float, end_s: float) -> list[tuple[float, float, Word]]:
        """The words whose occupied span lies wholly inside ``[start, end]``."""
        return [(lo, hi, w) for lo, hi, w in self._spans if start_s <= lo and hi <= end_s]


@dataclass(frozen=True)
class KeepFrames:
    start: int
    end: int
    label: str | None


@dataclass(frozen=True)
class Snap:
    """A keep-list edge that fell inside a word and moved inward past it."""

    keep: int
    edge: str  # "in" or "out"
    word: str
    from_frame: int
    to_frame: int


_TOUCH_S = 1e-6


def _walk_inward(
    words: Words, fps: Fps, frame: int, limit: int, *, up: bool
) -> tuple[list[tuple[str, Word, int, int]], int]:
    """Each word a keep edge at ``frame`` leaves behind when it moves inward (``up``: later)
    towards ``limit``, as ``(edge, word, from_frame, to_frame)``, and the frame the edge settles on.

    Where the next word touches this one (it starts exactly where this one ends), the frame
    nearest the shared boundary on the inward side is allowed, even when it falls just inside
    the next word: otherwise a boundary with no frame within the tolerance would pass the next
    word, and the one after it, whole.
    """
    moves: list[tuple[str, Word, int, int]] = []
    step = 1 if up else -1
    first = frame
    while (frame < limit if up else frame > limit) and (word := words.inside(frame)) is not None:
        frame += step
        following = words.inside(frame)
        if following is word:
            continue
        moves.append(("in" if up else "out", word, first, frame))
        first = frame
        if (
            following is not None
            and _touching(word, following, up)
            and frame == _boundary_frame(fps, word, up)
        ):
            break
    return moves, frame


def _touching(word: Word, following: Word, up: bool) -> bool:
    gap = following.start_s - word.end_s if up else word.start_s - following.end_s
    return abs(gap) <= _TOUCH_S


def _boundary_frame(fps: Fps, word: Word, up: bool) -> int:
    return inward_frame(word.end_s if up else word.start_s, fps, up=up)


def _cuts_word(index: int, edge: str, word: Word, a: int, b: int, fps: Fps) -> PlannerWarning:
    side = "start" if edge == "in" else "end"
    return PlannerWarning(
        code="keep-edge-cuts-word",
        message=(
            f"keep range {index}: its {edge} edge ({fps.seconds(a):.3f} s) cuts through "
            f"{word.text!r} ({word.start_s:.3f}-{word.end_s:.3f} s), so the {side} of the range "
            f"moves to {fps.seconds(b):.3f} s and the word is left out"
        ),
        at_s=round(fps.seconds(a), 3),
        hint="move the edge into the pause beside the word to keep it, or leave it out",
    )


def _overlap_between(words: Words, start_s: float, end_s: float) -> bool:
    """Whether transcript words meeting ``(start, end)`` truly overlap, not just touch."""
    reach = -math.inf
    for word in sorted(words.words, key=lambda w: (w.start_s, w.end_s)):
        if word.end_s <= start_s or word.start_s >= end_s:
            continue
        if word.start_s < reach - _TOUCH_S:
            return True
        reach = max(reach, word.end_s)
    return False


def keep_frames(
    keep: KeepList, words: Words, fps: Fps, *, min_shot_s: float
) -> tuple[list[KeepFrames], list[Snap], list[PlannerWarning]]:
    """The keep ranges in recording frames, never wider than the human made them.

    Ranges that touch merge first: they play unbroken, and a word across the seam is kept. The
    edges round inward to whole frames. An edge inside a word (``no-dialogue-splice``'s test)
    moves **inward** past it: the in point to the word's end, the out point to its start. The
    partial word is left out with the material the human already excluded; the range never
    grows. Each word an edge cuts through is named in a ``keep-edge-cuts-word`` warning. Where
    transcript words touch, the frame nearest the shared boundary is allowed, so a boundary with
    no frame in the tolerance does not pass whole words. When transcript words overlap, a whole
    word can go with it, and a warning says so. A
    range that this empties is dropped with a warning. Each remaining range must last at least
    ``min_shot_s``, or no fold could keep it.
    """
    spans: list[tuple[float, float, str | None]] = []
    for item in keep.ranges:
        if spans and item.start_s <= spans[-1][1]:
            spans[-1] = (spans[-1][0], max(spans[-1][1], item.end_s), spans[-1][2])
        else:
            spans.append((item.start_s, item.end_s, item.label))
    merged = [
        (inward_frame(a, fps, up=True), inward_frame(b, fps, up=False), x) for a, b, x in spans
    ]
    out: list[KeepFrames] = []
    snaps: list[Snap] = []
    warnings: list[PlannerWarning] = []
    for index, (start, end, label) in enumerate(merged):
        moves, start = _walk_inward(words, fps, start, end, up=True)
        more, end = _walk_inward(words, fps, end, start, up=False)
        for edge, word, a, b in moves + more:
            warnings.append(_cuts_word(index, edge, word, a, b, fps))
            snaps.append(Snap(index, edge, word.text, a, b))
        lost = [
            w
            for lo, hi, w in words.occupied(spans[index][0], spans[index][1])
            if hi <= fps.seconds(start) or lo >= fps.seconds(end)
        ]
        if lost and end > start:
            overlap = _overlap_between(words, spans[index][0], spans[index][1])
            how = "overlap" if overlap else "touch"
            warnings.append(
                PlannerWarning(
                    code="keep-edge-drops-word",
                    message=(
                        f"keep range {index}: leaving out a partial word at its edge also leaves "
                        f"out {len(lost)} word(s) wholly inside it ({lost[0].text!r} "
                        f"first); the transcript's words {how} there"
                    ),
                    at_s=round(fps.seconds(merged[index][0]), 3),
                    hint="check the transcript at this edge, or move the edge to a pause",
                )
            )
        if end <= start:
            warnings.append(
                PlannerWarning(
                    code="keep-dropped",
                    message=(
                        f"keep range {index} ({fps.seconds(merged[index][0]):.3f}-"
                        f"{fps.seconds(merged[index][1]):.3f} s) lies inside a word; it is dropped"
                    ),
                    at_s=round(fps.seconds(merged[index][0]), 3),
                    hint="start and end keep ranges in the pauses around whole words",
                )
            )
            continue
        out.append(KeepFrames(start, end, label))
    need = min_frames(min_shot_s, fps)
    for number, item in enumerate(out):
        if item.end - item.start < need:
            raise ProduceInputError(
                "keep-too-short",
                f"keep range {number} ({fps.seconds(item.start):.3f}-"
                f"{fps.seconds(item.end):.3f} s) lasts {item.end - item.start} frames, under "
                f"min_shot_s ({need} frames), after leaving out any partial word at its edges",
                hint="lengthen it, join it to a neighbouring range, or drop it",
            )
    if not out:
        raise ProduceInputError("keep-empty", "every keep range lies inside a word")
    return out, snaps, warnings


def resolve_camera(plan: EditPlan, name: str) -> tuple[str, ...]:
    """A camera name to its source ids: the sources with that ``camera`` label in recording
    order (one, or a camera's consecutive files, S24a), else the source with that id. Sources
    of one label that overlap on the recording clock are two angles, not parts: ambiguous."""
    labelled = sorted(
        (s for s in plan.sources if s.camera == name),
        key=lambda s: (plan.recording_key(s), s.recording_origin),
    )
    if len(labelled) == 1:
        return (labelled[0].id,)
    if len(labelled) > 1:
        ids = ", ".join(repr(s.id) for s in labelled)
        prev = labelled[0]
        for cur in labelled[1:]:
            if (
                plan.recording_key(cur) != plan.recording_key(prev)
                or prev.duration is None
                or cur.recording_origin < prev.recording_origin + prev.duration - PART_SLACK
            ):
                raise ProduceInputError(
                    "ambiguous-camera",
                    f"camera {name!r} names more than one source ({ids}) and they are not one "
                    "camera's consecutive files (they overlap, or one has no duration)",
                    hint="give each source its own camera label, or name a source id in the "
                    "schedule",
                )
            prev = cur
        return tuple(s.id for s in labelled)
    if any(s.id == name for s in plan.sources):
        return (name,)
    known = ", ".join(sorted({s.camera or s.id for s in plan.sources})) or "none"
    raise ProduceInputError(
        "unknown-camera",
        f"camera {name!r} is neither a camera label nor a source id in the stub (known: {known})",
    )


#: Frames one part may start before the previous ends (the stub's ``PART_OVERLAP_FRAMES``); the
#: frames both cover play from the earlier part.
PART_SLACK = 2


@dataclass(frozen=True)
class ScheduleFrames:
    """The camera schedule in recording frames: ``(start, end, source ids)``, chronological; the
    ids are one source, or a camera's parts in recording order. ``None`` bounds mean open-ended
    (the default camera covers everything)."""

    spans: list[tuple[int | None, int | None, tuple[str, ...]]]


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


def _covered_text(ids: Sequence[str], coverage: dict[str, Coverage], fps: Fps) -> str:
    spans = []
    for sid in ids:
        c = coverage[sid]
        end = "end" if c.end is None else f"{fps.seconds(c.end):.3f}"
        spans.append(f"{sid} {fps.seconds(c.origin):.3f}-{end} s")
    return ", ".join(spans)


def on_camera(
    start: int,
    end: int,
    keep: int,
    camera: Sequence[str],
    coverage: dict[str, Coverage],
    fps: Fps,
) -> list[Piece]:
    """``[start, end)`` shown from ``camera``: one piece of its one source, or a piece per part
    of a camera of several files, split where one file ends and the next begins. Frames in a gap
    between two parts, a range edge in a gap too, have no file and are left out here (the
    producer warns of them and moves the edges, :func:`part_gaps`); frames before the first part
    or after the last are an error, as for one file, and so is a range with no frame in any
    part."""
    if len(camera) == 1:
        source = camera[0]
        if not coverage[source].covers(start, end):
            raise ProduceInputError(
                "camera-does-not-cover",
                f"source {source!r} has no media for recording "
                f"{fps.seconds(start):.3f}-{fps.seconds(end):.3f} s",
                hint="schedule another camera there, or fix the source's origin or duration",
            )
        return [Piece(source, start, end, keep)]
    first, last = coverage[camera[0]], coverage[camera[-1]]
    label = first.camera or camera[0]
    where = f"{fps.seconds(start):.3f}-{fps.seconds(end):.3f} s"
    if start < first.origin or (last.end is not None and end > last.end):
        raise ProduceInputError(
            "camera-does-not-cover",
            f"camera {label!r} has no media for recording {where}: its files cover "
            + _covered_text(camera, coverage, fps),
            hint="schedule another camera there, or move the range inside the camera's files",
        )
    pieces: list[Piece] = []
    cursor = start
    for sid in camera:
        cover = coverage[sid]
        lo = max(cursor, cover.origin)
        hi = end if cover.end is None else min(end, cover.end)
        if hi > lo:
            pieces.append(Piece(sid, lo, hi, keep))
            cursor = hi
    if not pieces:
        raise ProduceInputError(
            "range-in-gap",
            f"recording {where} on camera {label!r} lies wholly in a gap between its files, "
            "where it has no media; its files cover " + _covered_text(camera, coverage, fps),
            hint="move the range into one of those spans, or schedule another camera there",
        )
    return pieces


def split(
    start: int,
    end: int,
    keep: int,
    schedule: ScheduleFrames,
    coverage: dict[str, Coverage],
    fps: Fps,
) -> list[Piece]:
    """``[start, end)`` cut into pieces at the schedule's camera changes, and at the boundaries
    between a camera's files (:func:`on_camera`).

    Every frame must be scheduled, and each piece's source must hold media for it.
    """
    pieces: list[Piece] = []
    cursor = start
    for lo, hi, camera in schedule.spans:
        a = start if lo is None else max(start, lo)
        b = end if hi is None else min(end, hi)
        if b <= a:
            continue
        if a > cursor:
            break
        pieces += on_camera(a, b, keep, camera, coverage, fps)
        cursor = b
    if cursor < end or not pieces:
        gap_from = fps.seconds(cursor)
        raise ProduceInputError(
            "schedule-gap",
            f"no camera is scheduled for recording {gap_from:.3f} s inside keep range {keep}",
            hint="the camera schedule must cover every kept moment",
        )
    return pieces


def less_holes(
    spans: Sequence[tuple[int, int]], holes: Sequence[tuple[int, int]]
) -> list[tuple[int, int]]:
    """``spans`` less the frames of ``holes`` (both half-open, chronological)."""
    out: list[tuple[int, int]] = []
    for a, b in spans:
        cursor = a
        for lo, hi in holes:
            if hi <= cursor or lo >= b:
                continue
            if lo > cursor:
                out.append((cursor, lo))
            cursor = max(cursor, hi)
        if cursor < b:
            out.append((cursor, b))
    return out


@dataclass(frozen=True)
class PartGaps:
    """What :func:`part_gaps` did: the keep ranges (an edge moved off a gap when the piece
    beyond it was too short to keep), the frames to leave out of each and the warnings."""

    keeps: list[KeepFrames]
    holes: list[list[tuple[int, int]]]
    warnings: list[PlannerWarning]


def part_gaps(
    keeps: Sequence[KeepFrames],
    schedule: ScheduleFrames,
    coverage: dict[str, Coverage],
    words: Words,
    fps: Fps,
    *,
    handle: int,
    need: int,
) -> PartGaps:
    """Keep ranges that meet a gap between two files of the camera on screen (S24a): a range
    that runs across a gap is cut there (``range-crosses-gap``), and a range whose edge lies in
    a gap starts or ends at the file beside it (``range-edge-in-gap``); either warning names the
    gap's place and length. The frames with no file are left out, and the cut there is like any
    other: each file edge it meets also gives up the crossfade's ``handle`` frames (the media a
    crossfade plays past a cut) and any word the edge would split, as a keep edge does. A run of
    files left between gaps (or between a gap and the range's edge) that is shorter than ``need``
    frames is dropped too, and the warning names it."""
    out_keeps: list[KeepFrames] = []
    holes: list[list[tuple[int, int]]] = []
    warnings: list[PlannerWarning] = []
    for index, keep in enumerate(keeps):
        pieces = split(keep.start, keep.end, index, schedule, coverage, fps)
        # runs of pieces that play unbroken (one file, or files that meet with no gap)
        runs: list[list[Piece]] = []
        for piece in pieces:
            if runs and piece.rec_in <= runs[-1][-1].rec_out:
                runs[-1].append(piece)
            else:
                runs.append([piece])
        lead, trail = runs[0][0].rec_in > keep.start, runs[-1][-1].rec_out < keep.end
        if len(runs) == 1 and not lead and not trail:
            out_keeps.append(keep)
            holes.append([])
            continue
        kept: list[tuple[int, int]] = []
        dropped: list[str] = []
        for n, run in enumerate(runs):
            first, last = run[0], run[-1]
            lo, hi = first.rec_in, last.rec_out
            if n or lead:  # a file edge after a gap: the in point gives up the handle
                _, lo = _walk_inward(words, fps, min(hi, lo + handle), hi, up=True)
            if n < len(runs) - 1 or trail:  # a file edge before a gap
                _, hi = _walk_inward(words, fps, max(lo, hi - handle), lo, up=False)
            if hi - lo < need:
                names = ", ".join(repr(p.source) for p in run)
                dropped.append(
                    f"{fps.seconds(first.rec_in):.3f}-{fps.seconds(last.rec_out):.3f} s "
                    f"({names}, under {need} frames once the gap's edges are left out)"
                )
                continue
            kept.append((lo, hi))
        too_short = f"; too short to keep, {' and '.join(dropped)} went too" if dropped else ""
        gaps: list[tuple[str, int, int, str | None, str | None]] = []
        if lead:
            gaps.append(
                ("range-edge-in-gap", keep.start, runs[0][0].rec_in, None, runs[0][0].source)
            )
        for a, b in itertools.pairwise(runs):
            gaps.append(
                ("range-crosses-gap", a[-1].rec_out, b[0].rec_in, a[-1].source, b[0].source)
            )
        if trail:
            last = runs[-1][-1]
            gaps.append(("range-edge-in-gap", last.rec_out, keep.end, last.source, None))
        for code, lo, hi, before, after in gaps:
            where = []
            if before is not None:
                where.append(f"{before!r} ends at {fps.seconds(lo):.3f} s")
            if after is not None:
                where.append(f"{after!r} starts at {fps.seconds(hi):.3f} s")
            if code == "range-crosses-gap":
                what = (
                    f"keep range {index} runs across a {fps.seconds(hi - lo):.3f} s gap between "
                    f"camera files ({', '.join(where)}): it is cut there"
                )
            else:
                edge = "starts" if after is not None else "ends"
                what = (
                    f"keep range {index} {edge} in a gap between camera files, with "
                    f"{fps.seconds(hi - lo):.3f} s of it in the gap "
                    f"({', '.join(where)}): it {edge} at that file instead"
                )
            warnings.append(
                PlannerWarning(
                    code=code,
                    message=f"{what}, and the frames with no file are left out{too_short}",
                    at_s=round(fps.seconds(lo), 3),
                    source=after or before,
                    hint="start and end every range inside one camera file, in a pause "
                    "(the guide lists each file's span)",
                )
            )
        if not kept:
            raise ProduceInputError(
                "range-in-gap",
                f"keep range {index} ({fps.seconds(keep.start):.3f}-{fps.seconds(keep.end):.3f}"
                " s) meets gaps between camera files and leaves no piece of at least "
                f"{need} frames: " + " and ".join(dropped),
                hint="move the range inside one camera file (the guide lists their spans)",
            )
        out_keeps.append(replace(keep, start=kept[0][0], end=kept[-1][1]))
        holes.append([(a[1], b[0]) for a, b in itertools.pairwise(kept)])
    return PartGaps(out_keeps, holes, warnings)


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
