"""Timeline import (S20): a timeline edited in Resolve back as an EditPlan and a diff.

The flow: plan -> FCPXML (``aar plan export``) -> Resolve -> the owner edits -> File > Export >
Timeline as ``.drt`` -> :func:`read_timeline` + :func:`import_timeline` -> an edited plan and a
diff the agent reads. Pure and cold: nothing here talks to Resolve or the network, and the only
file read is the ``.drt`` (through :mod:`aar.drp`, with its own size limits).

The rule: **a wrong plan or a wrong diff is never returned without saying so.** Where the code
cannot know, it reports; it does not guess quietly.

* **Reader issues.** A reader issue that can hide or change an item refuses the import
  (``damaged_timeline``); the others are passed through as ``reader_warnings``
  (:data:`READER_WARNING_CODES`).
* **Expected timeline.** S18's :func:`~aar.plan.export.layout.build_layout` is the original plan
  as timeline clips (lanes, record ranges, source ranges); nothing here re-derives its rules.
* **Media.** An item matches a source when its path is the edit-PC path the export named for
  that source (:func:`~aar.plan.export.media.media_ref` with the export's media map). Only a
  source whose edit-PC path is unknown can match by file name, and every such match is listed.
  An item with a known source's file name but another path is foreign ("same name, different
  file"). Anything else is foreign: reported, never guessed.
* **Segments.** The picture items on each content track become the plan's segments on that
  track, frame-exact, positions from the timeline. Items are assigned to original segments as a
  whole: an exact range first, then the larger source overlap, then the nearer original record
  position, then order. A matched segment inherits the original's other fields; reframe keys are
  windowed to the new source range.
* **Timing.** Whatever exists as an item in the timeline (segments on every track, titles) takes
  its position from the timeline. What the plan positions in record frames but is never an item
  (captions, markers) is re-anchored to the content it sat over, and dropped when that content
  is gone. Every move and every drop is listed.
* **Everything else** in the plan is carried when it is still valid and listed under ``dropped``
  when it is not. What the timeline holds that a plan cannot (retimes, stills, sub-frame
  positions, audio edits, changes on tracks the plan has no lane for, transitions added in
  Resolve) is listed under ``not_carried`` with a reason code.

The assumptions that hold until a real round trip through Resolve confirms them are the functions
under "Assumptions" below (DESIGN.md, "Timeline import (S20)").
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar
from urllib.parse import unquote, urlsplit

from pydantic import ValidationError

from aar.drp import read_drp
from aar.drp.errors import DrpError
from aar.drp.model import Issue, MediaClip, Timeline, TimelineItem
from aar.plan.apply.errors import CompileError
from aar.plan.export.errors import ExportError
from aar.plan.export.layout import Layout, build_layout
from aar.plan.export.media import MediaMap, MediaRef, media_ref
from aar.plan.lint import LintReport, lint
from aar.plan.model import EditPlan, ReframeKey, Segment, Source
from aar.plan.profile import Profile

FORMAT = "aar.timeline-import/2"
#: Items read from one timeline, all tracks together; more is refused (``timeline_too_large``).
MAX_ITEMS = 20_000
#: A timeline or media rate (a float in the file) matches the plan's rational fps within this.
FPS_TOLERANCE = 1e-3
#: Reader issues listed in a refusal or in ``reader_warnings`` (the rest are counted).
MAX_LISTED_ISSUES = 50

IMPORT_CODES = frozenset(
    {
        "unreadable_timeline",
        "damaged_timeline",
        "no_timeline",
        "several_timelines",
        "timeline_not_found",
        "timeline_too_large",
        "fps_mismatch",
        "media_fps_mismatch",
        "source_range_outside",
        "output_exists",
        "output_folder_missing",
        "invalid_edited_plan",
    }
)

#: Reason codes of ``not_carried`` (timeline content a plan cannot hold) and ``dropped`` (plan
#: entries no longer valid after the edit). Pinned by the tests.
NOT_CARRIED_CODES = (
    "retimed",
    "still_image",
    "sub_frame",
    "zero_length",
    "outside_timeline",
    "negative_source_in",
    "unreadable_item",
    "audio_only_edit",
    "other_picture_track",
    "title_added",
    "transition_added",
    "transition_missing",
    "outro_missing",
    "outro_position",
    "outro_changed",
    "sting_missing",
)
DROPPED_CODES = (
    "transition_join_gone",
    "title_removed",
    "title_past_end",
    "caption_content_gone",
    "caption_past_end",
    "marker_content_gone",
    "marker_past_end",
)
#: Why an item is foreign (``foreign[].reason``).
FOREIGN_REASONS = ("unknown_file", "same_name_different_file", "ambiguous_file_name")

#: Reader issue codes that cannot hide or change a timeline item, its position, its source range
#: or its media path: passed through as ``reader_warnings``. Every other code (any ``scan-*``
#: code, a missing or unreadable member or container, a blob that did not decode, a path taken
#: from another stream, unresolved media, an uninterpreted track, a budget hit, ...) refuses the
#: import as ``damaged_timeline``, including codes this list has never seen.
READER_WARNING_CODES = frozenset(
    {
        "uninterpreted",  # Fusion comps and other payloads that are not timeline items
        "linked-files",  # a clip also names a synced external file; its own path is read
        "stale-media-file-path",  # a timeline <MediaFilePath> differs; the pool path is used
        "orphan-markers",  # markers are not read into the plan
        "marker-decode",
        "issues-truncated",  # the code it counts decides on its own
        "codec-unknown",
        "no-version-header",
        "version-mismatch",
        "orphan-timeline",
        "orphan-bin",
        "bin-cycle",
        "unknown-media",  # an item using it is caught by unresolved-media, which refuses
        "drb-best-effort",
        "no-timelines",  # no timeline at all is refused as no_timeline
    }
)

#: File name endings of still images (when the media pool does not say the clip has 1 frame).
STILL_EXTENSIONS = frozenset(
    {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif", ".webp", ".heic", ".dpx", ".exr"}
)

LIMITS = (
    "enabled_state: the reader cannot see whether an item is enabled; a disabled clip is "
    "imported as if it plays",
    "audio_transitions: transition items on audio tracks are not compared",
    "titles: title items are matched by track and record range only; their text is not read",
    "not_read: markers, grades, Fusion and Fairlight work in the .drt are not read into the plan",
)

ASSUMPTIONS = (
    {
        "id": "tracks",
        "text": "after importing aar's FCPXML, Resolve's items sit on the tracks the layout "
        "predicts: layout lane N on video track N+1, so the plan's V1 spine on V1",
        "check": "import an exported plan with a V2 clip and a title, export the timeline as "
        ".drt, and run import-timeline: the diff must be empty",
    },
    {
        "id": "source_frames",
        "text": "a moving-picture item's source in point counts frames from the media file's "
        "first frame, as the plan's src_in does (not from its start timecode); stills are not "
        "covered (a real export shows still in points near 01:00:00:00) and are not carried",
        "check": "the same round trip with sources whose start timecode is not 0: unchanged "
        "segments keep their src_in",
    },
    {
        "id": "media_path",
        "text": "an item's media path is the edit-PC path the FCPXML asset named",
        "check": "the same round trip: every item matches its source by path, none by file name",
    },
)


class TimelineImportError(CompileError):
    """A typed import refusal: ``code`` is one of :data:`IMPORT_CODES` (exit 2 in the CLI)."""

    codes: ClassVar[frozenset[str]] = IMPORT_CODES


# -- Assumptions (until a real round trip through Resolve; DESIGN "Timeline import (S20)") -------


def track_of_lane(lane: int) -> int:
    """Assumption ``tracks``: layout lane ``lane`` is video track ``lane + 1`` after import
    (dense, as S18's import test found), so the primary spine (lane 0) is V1."""
    return lane + 1


def source_frame(item: TimelineItem) -> int | None:
    """Assumption ``source_frames``: a moving-picture item's ``<In>`` is the plan's ``src_in``
    (frames from the file's first frame). Stills never reach here (:func:`is_still`)."""
    return item.source_in


def media_key(path: str) -> str:
    """Assumption ``media_path``: an item's media path is the path the FCPXML asset named, so it
    is compared with the export's edit-PC path, Windows-style (``/`` as ``\\``, case folded)."""
    return path.replace("/", "\\").casefold()


def file_name(path: str) -> str:
    """The last component of a Windows or POSIX path, case folded (the file-name fallback)."""
    return re.split(r"[\\/]", path)[-1].casefold()


def is_still(item: TimelineItem, clip: MediaClip | None) -> bool:
    """A still image: the media pool says the clip has one frame, or its file name says so."""
    if clip is not None and clip.frames == 1:
        return True
    path = item.media_path or ""
    dot = path.rfind(".")
    return dot >= 0 and path[dot:].casefold() in STILL_EXTENSIONS


# -- reading ------------------------------------------------------------------------------------


@dataclass
class TimelineFile:
    """One timeline read from a ``.drt`` with :mod:`aar.drp`, its media-pool clips and the
    reader issues that cannot change an item (``warnings``)."""

    path: str
    sha256: str
    timeline: Timeline
    media: dict[str, MediaClip]
    warnings: list[Issue] = field(default_factory=list)


def _issue_text(issue: Issue) -> str:
    member = f" in {issue.where.member}" if issue.where is not None else ""
    return f"{issue.severity} {issue.code}{member}: {issue.message}"


def read_timeline(path: Path, name: str | None = None) -> TimelineFile:
    """Read ``path`` with :func:`aar.drp.read_drp` and pick its one timeline (or ``name``)."""
    hint = "export the timeline from Resolve with File > Export > Timeline as a .drt file"
    try:
        drp = read_drp(path)
    except (DrpError, OSError) as exc:
        raise TimelineImportError(
            "unreadable_timeline", f"{path} could not be read as a .drt: {exc}", hint
        ) from None
    if not drp.is_zip:
        raise TimelineImportError(
            "unreadable_timeline", f"{path} is not a .drt (not a zip archive)", hint
        )
    damage = [i for i in drp.issues if i.code not in READER_WARNING_CODES]
    if damage:
        listed = [_issue_text(i) for i in damage[:MAX_LISTED_ISSUES]]
        if len(damage) > MAX_LISTED_ISSUES:
            listed.append(f"... {len(damage) - MAX_LISTED_ISSUES} more")
        raise TimelineImportError(
            "damaged_timeline",
            f"{path}: the reader found {len(damage)} problem(s) that can hide or change "
            f"timeline items ({', '.join(sorted({i.code for i in damage}))})",
            "export the timeline from Resolve again (File > Export > Timeline); if it repeats, "
            "check the file with `aar drp describe`",
            problems=listed,
            issues=[{"severity": i.severity, "code": i.code} for i in damage],
        )
    timelines = [t for t in drp.timelines if t.kind == "timeline"]
    names = [t.name for t in timelines]
    if not timelines:
        raise TimelineImportError("no_timeline", f"{path} holds no timeline", hint)
    if name is not None:
        timelines = [t for t in timelines if t.name == name]
        if not timelines:
            raise TimelineImportError(
                "timeline_not_found",
                f"{path} has no timeline named {name!r} (it has {names!r})",
                "give --timeline one of the names listed",
                timelines=names,
            )
    if len(timelines) > 1:
        raise TimelineImportError(
            "several_timelines",
            f"{path} holds {len(timelines)} timelines named {[t.name for t in timelines]!r}",
            "pick one with --timeline NAME (export a single timeline if names repeat)",
            timelines=names,
        )
    [timeline] = timelines
    count = sum(len(track.items) for track in timeline.tracks)
    if count > MAX_ITEMS:
        raise TimelineImportError(
            "timeline_too_large",
            f"timeline {timeline.name!r} has {count} items, over the {MAX_ITEMS} limit",
            "import a smaller timeline",
            items=count,
        )
    if timeline.frame_rate is None or timeline.start_frame is None:
        raise TimelineImportError(
            "unreadable_timeline",
            f"timeline {timeline.name!r} has no readable frame rate or start frame",
            hint,
        )
    return TimelineFile(str(path), drp.sha256, timeline, drp.media_by_id(), list(drp.issues))


# -- the import ---------------------------------------------------------------------------------


@dataclass
class _Item:
    """One timeline item in plan terms (record frames from the timeline's start)."""

    track: str
    number: int
    kind: str
    item: TimelineItem
    record_in: int = 0
    record_out: int = 0
    src_in: int | None = None
    src_out: int | None = None
    key: str | None = None
    source: str | None = None
    match: str | None = None
    foreign: str | None = None
    excluded: bool = False
    used: bool = False

    @property
    def is_video(self) -> bool:
        return self.track.startswith("V")

    @property
    def is_picture(self) -> bool:
        return self.is_video and self.kind not in ("transition", "audio")

    @property
    def is_title(self) -> bool:
        """Assumption: a media-less picture item is a title (or a generator or solid)."""
        return self.is_picture and not self.item.media_path

    def where(self) -> dict[str, Any]:
        return {
            "track": self.track,
            "record_in": self.record_in,
            "record_out": self.record_out,
            "kind": self.kind,
            "name": self.item.name,
            "media": self.item.media_path,
        }


@dataclass
class ImportResult:
    """The edited plan and the diff (a JSON-ready dict with a stable key and list order)."""

    plan: EditPlan
    diff: dict[str, Any]
    lint: LintReport | None = None

    @property
    def lint_errors(self) -> bool:
        return self.lint is not None and not self.lint.ok

    @property
    def export_errors(self) -> bool:
        return bool(self.diff.get("export_errors"))

    @property
    def has_errors(self) -> bool:
        """Exit 1 in the CLI: the plan is written but lint or the export's layout refuses it."""
        return self.lint_errors or self.export_errors


@dataclass
class _Expected:
    """A transition the edited plan puts on V1: its cut and length."""

    location: str
    cut: int
    frames: int
    found: bool = False


def _fallback_ref(source: Source) -> MediaRef:
    """A source the export rules cannot place (no map, a pool name): no edit-PC path, so it can
    match by file name only."""
    media = source.media
    path = unquote(urlsplit(media).path) if media[:5].lower() == "file:" else media
    name = re.split(r"[\\/]", path)[-1]
    return MediaRef(source.id, media, f"aar-unplaced:{source.id}", "hub-path", "", name)


def _resolver(media_map: MediaMap | None) -> Any:
    def resolve(source: Source) -> MediaRef:
        try:
            return media_ref(source, media_map)
        except ExportError:
            return _fallback_ref(source)

    return resolve


def _interpolate(keys: list[ReframeKey], time: int) -> dict[str, float]:
    """The reframe at ``time``: linear between keys, held before the first and after the last."""
    if time <= keys[0].time:
        key = keys[0]
        return {"x": key.x, "y": key.y, "zoom": key.zoom}
    for a, b in itertools.pairwise(keys):
        if a.time <= time <= b.time:
            if b.time == a.time:
                return {"x": b.x, "y": b.y, "zoom": b.zoom}
            t = (time - a.time) / (b.time - a.time)
            return {
                "x": a.x + (b.x - a.x) * t,
                "y": a.y + (b.y - a.y) * t,
                "zoom": a.zoom + (b.zoom - a.zoom) * t,
            }
    key = keys[-1]
    return {"x": key.x, "y": key.y, "zoom": key.zoom}


def window_reframe(keys: list[ReframeKey], start: int, end: int) -> list[ReframeKey]:
    """Reframe keys of a segment cut down (or extended) to ``[start, end)`` of its old frames:
    keys inside are shifted by ``-start``; a side whose keys fall outside gets one key at the new
    boundary with the interpolated value, so the framing over the kept frames is unchanged."""
    if not keys:
        return []
    inside = [k for k in keys if start <= k.time < end]
    out: list[ReframeKey] = []
    if any(k.time < start for k in keys) and not any(k.time == start for k in inside):
        out.append(ReframeKey(time=0, **_interpolate(keys, start)))
    out += [k.model_copy(update={"time": k.time - start}) for k in inside]
    last = end - 1
    if (
        any(k.time >= end for k in keys)
        and last > start
        and not any(k.time == last for k in inside)
    ):
        out.append(ReframeKey(time=last - start, **_interpolate(keys, last)))
    return out


def _with_segments(plan: EditPlan, segments: list[Segment]) -> EditPlan:
    """``plan`` with another segment list, for the model's own ``record_ranges`` and ``joins``."""
    return plan.model_copy(update={"segments": segments})


def assign_items(
    items: Sequence[tuple[str, int, int, int]],
    originals: Sequence[tuple[int, str, int, int, int]],
) -> tuple[dict[int, int], dict[int, list[int]]]:
    """Assign items to original segments as a whole. ``items`` are ``(source, src_in, src_out,
    record_in)`` in record order; ``originals`` are ``(index, source, src_in, src_out,
    record_in)``. Returns item position -> original index, and for each item matched by overlap
    the other originals it also overlaps (the diff lists them: the choice was not certain).

    1. An exact source range at the original's record position, then an exact range anywhere
       (the nearer original position first, then plan order); each original takes one exact item.
    2. Every other item goes to the original of its source it overlaps most (never one an item
       matched exactly: that item is added). Among originals it overlaps equally, one that no
       item holds yet comes first (the assignment is taken as a whole: a trimmed copy is not a
       piece of a segment that already has its own item), then the nearer original record
       position; items still tied are given out in record order to their tied originals in plan
       order. An item that overlaps nothing it can take is unassigned (added)."""
    assigned: dict[int, int] = {}
    exact: set[int] = set()
    by_range: dict[tuple[str, int, int], list[tuple[int, int]]] = {}
    by_source: dict[str, list[tuple[int, int, int, int]]] = {}
    old_in: dict[int, int] = {}
    for index, source, src_in, src_out, record_in in originals:
        by_range.setdefault((source, src_in, src_out), []).append((index, record_in))
        by_source.setdefault(source, []).append((index, src_in, src_out, record_in))
        old_in[index] = record_in
    for at_position in (True, False):
        for pos, (source, src_in, src_out, record_in) in enumerate(items):
            if pos in assigned:
                continue
            free = [
                (abs(old - record_in), index)
                for index, old in by_range.get((source, src_in, src_out), [])
                if index not in exact and (not at_position or old == record_in)
            ]
            if free:
                index = min(free)[1]
                assigned[pos] = index
                exact.add(index)
    also: dict[int, list[int]] = {}
    ties: list[tuple[int, list[int]]] = []
    for pos, (source, src_in, src_out, _) in enumerate(items):
        if pos in assigned:
            continue
        overlaps = {
            index: min(o_out, src_out) - max(o_in, src_in)
            for index, o_in, o_out, _ in by_source.get(source, [])
            if index not in exact
        }
        overlaps = {i: o for i, o in overlaps.items() if o > 0}
        if not overlaps:
            continue
        most = max(overlaps.values())
        tied = sorted(i for i, o in overlaps.items() if o == most)
        if len(tied) == 1:
            assigned[pos] = tied[0]
        else:
            ties.append((pos, tied))
        also[pos] = sorted(overlaps)
    floor = -1
    for pos, tied in ties:
        record_in = items[pos][3]
        held = set(assigned.values())
        unheld = [i for i in tied if i not in held] or tied
        near = min(abs(old_in[i] - record_in) for i in unheld)
        nearest = [i for i in unheld if abs(old_in[i] - record_in) == near]
        later = [i for i in nearest if i >= floor]
        assigned[pos] = later[0] if later else nearest[0]
        floor = assigned[pos]
    alternatives = {
        pos: [i for i in others if i != assigned[pos]]
        for pos, others in also.items()
        if len(others) > 1
    }
    return assigned, alternatives


class _Importer:
    def __init__(
        self,
        plan: EditPlan,
        timeline: Timeline,
        media: Mapping[str, MediaClip],
        media_map: MediaMap | None,
    ) -> None:
        self.plan = plan
        self.timeline = timeline
        self.media = media
        self.media_map = media_map
        self.not_carried: list[dict[str, Any]] = []
        self.dropped: list[dict[str, Any]] = []
        self.foreign: list[dict[str, Any]] = []
        self.repositioned: list[dict[str, Any]] = []

    # -- report helpers ------------------------------------------------------------------------

    def carry_not(self, code: str, detail: str, where: dict[str, Any] | None = None) -> None:
        assert code in NOT_CARRIED_CODES
        self.not_carried.append({"code": code, **(where or {}), "detail": detail})

    def drop(self, code: str, location: str, detail: str) -> None:
        assert code in DROPPED_CODES
        self.dropped.append({"code": code, "location": location, "detail": detail})

    def moved(self, location: str, by: str, old: tuple[int, int], new: tuple[int, int]) -> None:
        if old != new:
            self.repositioned.append(
                {
                    "location": location,
                    "by": by,
                    "old": {"record_in": old[0], "record_out": old[1]},
                    "new": {"record_in": new[0], "record_out": new[1]},
                }
            )

    # -- steps ---------------------------------------------------------------------------------

    def run(self) -> tuple[EditPlan, dict[str, Any]]:
        self._check_rate()
        self.layout: Layout = build_layout(
            self.plan,
            media_map=self.media_map,
            allow_unknown_timecode=True,
            resolve_media=_resolver(self.media_map),
        )
        self.primary = f"V{track_of_lane(0)}"
        self._index_sources()
        self._collect()
        self._match_media()
        self._check_matched()
        self._find_outro()
        self._match_segments()
        segments = self._segments()
        transitions = self._transitions(segments)
        outro, outro_diff = self._outro(segments)
        draft = self.plan.model_copy(
            update={"segments": segments, "transitions": transitions, "outro": outro}
        )
        at = self._content_map(draft)
        titles = self._titles(draft, at)
        captions, markers = self._reanchored(draft, at)
        try:
            edited = EditPlan.model_validate(
                draft.model_copy(
                    update={"titles": titles, "captions": captions, "markers": markers}
                ).model_dump()
            )
        except ValidationError as exc:
            raise TimelineImportError(
                "invalid_edited_plan",
                f"the timeline gives a plan the plan model refuses: {exc.error_count()} "
                "problem(s), the first at "
                + ".".join(str(p) for p in exc.errors()[0]["loc"])
                + f": {exc.errors()[0]['msg']}",
                "check the timeline in Resolve at the place named; report it if it looks right",
                problems=[
                    f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()[:20]
                ],
            ) from None
        self._leftovers()
        self._transition_items(edited)
        self._audio(edited)
        return edited, self._diff(edited, outro_diff)

    def _check_rate(self) -> None:
        rate = self.timeline.frame_rate
        fps = self.plan.fps
        if rate is None or abs(rate - fps.num / fps.den) > FPS_TOLERANCE:
            raise TimelineImportError(
                "fps_mismatch",
                f"timeline {self.timeline.name!r} runs at {rate!r} fps, the plan at "
                f"{fps.num}/{fps.den}",
                "import against the plan the timeline was exported from, at the same frame rate",
                timeline_fps=rate,
                plan_fps=f"{fps.num}/{fps.den}",
            )

    def _index_sources(self) -> None:
        """Each source's edit-PC path when the export rules place it; the file names of known
        paths (a same-name item elsewhere is foreign) and of unplaced sources (matched by name
        when the name names one file)."""
        resolve = _resolver(self.media_map)
        self.refs = {s.id: self.layout.media.get(s.id) or resolve(s) for s in self.plan.sources}
        self.by_path: dict[str, list[str]] = {}
        self.known_names: set[str] = set()
        unplaced_files: dict[str, set[str]] = {}
        self.by_name: dict[str, list[str]] = {}
        for source in self.plan.sources:
            ref = self.refs[source.id]
            name = ref.name.casefold()
            if ref.windows_path:
                self.by_path.setdefault(media_key(ref.windows_path), []).append(source.id)
                self.known_names.add(name)
            else:
                unplaced_files.setdefault(name, set()).add(source.media)
                self.by_name.setdefault(name, []).append(source.id)
        self.unique_names = {n for n, files in unplaced_files.items() if len(files) == 1}

    def _collect(self) -> None:
        start = self.timeline.start_frame
        assert start is not None
        self.items: list[_Item] = []
        counters: dict[str, int] = {}
        for track in self.timeline.tracks:
            if track.type not in ("video", "audio"):
                continue
            label = track.label
            ordered = sorted(
                track.items,
                key=lambda i: (i.record_start is None, i.record_start or 0, i.id),
            )
            for item in ordered:
                counters[label] = counters.get(label, 0) + 1
                entry = _Item(label, counters[label], item.kind, item)
                self.items.append(entry)
                self._classify(entry, start)

    def _classify(self, entry: _Item, start: int) -> None:
        """Set the item's plan frames, or exclude it with a reason code."""
        item = entry.item
        if item.record_start is None or item.duration is None:
            entry.excluded = True
            self.carry_not(
                "unreadable_item",
                "the item's start or duration could not be read",
                {"track": entry.track, "kind": item.kind, "name": item.name},
            )
            return
        entry.record_in = item.record_start - start
        entry.record_out = entry.record_in + item.duration
        reason: tuple[str, str] | None = None
        if entry.record_in < 0:
            reason = (
                "outside_timeline",
                f"the item starts {-entry.record_in} frame(s) before the timeline's start; a "
                "plan's record frames start at 0",
            )
        elif item.duration <= 0:
            reason = ("zero_length", "the item has no length")
        if reason is None and item.kind != "transition":
            src_in = source_frame(item)
            clip = self.media.get(item.media_clip_id or "")
            if item.fractions:
                fields = ", ".join(sorted(item.fractions))
                reason = (
                    "sub_frame",
                    f"the item sits at a sub-frame position ({fields}); a plan holds whole frames",
                )
            elif item.media_path and is_still(item, clip):
                reason = (
                    "still_image",
                    "the item shows a still image; its source in point is not counted from the "
                    "file's first frame, so it cannot be read as a plan range",
                )
            elif item.timemap_form == "retimed" or (item.speed is not None and item.speed != 1.0):
                reason = (
                    "retimed",
                    f"the item is retimed (speed {item.speed!r}); a plan plays sources at 100 %",
                )
            elif src_in is None and item.media_path:
                reason = ("unreadable_item", "the item's source in point could not be read")
            elif src_in is not None and src_in < 0:
                reason = (
                    "negative_source_in",
                    f"the item's source in point is {src_in}, before the file's first frame",
                )
            else:
                entry.src_in = src_in
                entry.src_out = src_in + item.duration if src_in is not None else None
        if reason is not None:
            entry.excluded = True
            self.carry_not(reason[0], reason[1], entry.where())

    def _match_media(self) -> None:
        for entry in self.items:
            path = entry.item.media_path
            if entry.kind == "transition" or not path or entry.excluded:
                continue
            entry.key = media_key(path)
            ids = self.by_path.get(entry.key)
            if ids:
                entry.match = "path"
            else:
                name = file_name(path)
                if name in self.known_names:
                    entry.foreign = "same_name_different_file"
                elif name in self.unique_names:
                    ids = self.by_name[name]
                    entry.match = "file-name"
                elif name in self.by_name:
                    entry.foreign = "ambiguous_file_name"
                else:
                    entry.foreign = "unknown_file"
            if ids:
                entry.source = self._pick(ids, entry)

    def _pick(self, ids: list[str], entry: _Item) -> str:
        """Several sources on one file: the one an original segment uses with the largest source
        overlap with the item, else the first in plan order."""
        if len(ids) == 1 or entry.src_in is None or entry.src_out is None:
            return ids[0]
        best, best_overlap = ids[0], 0
        for seg in self.plan.segments:
            if seg.source in ids:
                overlap = min(seg.src_out, entry.src_out) - max(seg.src_in, entry.src_in)
                if overlap > best_overlap:
                    best, best_overlap = seg.source, overlap
        return best

    def _check_matched(self) -> None:
        sources = self.plan.source_map()
        fps = self.plan.fps.num / self.plan.fps.den
        for entry in self.items:
            if entry.source is None or entry.excluded:
                continue
            clip = self.media.get(entry.item.media_clip_id or "")
            if (
                entry.kind == "video"
                and clip is not None
                and clip.frame_rate
                and abs(clip.frame_rate - fps) > FPS_TOLERANCE
            ):
                raise TimelineImportError(
                    "media_fps_mismatch",
                    f"{entry.track} item {entry.item.name!r} plays media at {clip.frame_rate!r} "
                    f"fps in a {fps:.3f} fps timeline; its source frames are not plan frames",
                    "conform the media to the timeline's rate in Resolve, or cut it at its own "
                    "rate",
                    track=entry.track,
                    record_in=entry.record_in,
                )
            duration = sources[entry.source].duration
            assert entry.src_out is not None
            if duration is not None and entry.src_out > duration:
                raise TimelineImportError(
                    "source_range_outside",
                    f"{entry.track} item {entry.item.name!r} reads source {entry.source!r} frames "
                    f"{entry.src_in}..{entry.src_out}, past its {duration} frames",
                    "check the source's duration in the plan, or that the item plays this file",
                    track=entry.track,
                    record_in=entry.record_in,
                    source=entry.source,
                )

    def _pictures(self, track: str) -> list[_Item]:
        return [e for e in self.items if e.track == track and not e.excluded and e.is_picture]

    def _find_outro(self) -> None:
        """Assumption: the outro is the last item of the outro's source on the outro's track."""
        self.outro_item: _Item | None = None
        outro = self.plan.outro
        if outro is None:
            return
        track = f"V{track_of_lane(self.layout.lanes[outro.track])}"
        found = [e for e in self._pictures(track) if e.source == outro.source]
        if found:
            self.outro_item = found[-1]
            self.outro_item.used = True

    def _match_segments(self) -> None:
        """Assign the picture items of each content track (one with a planned segment) to that
        track's original segments (:func:`assign_items`). Items with a file the plan does not
        know are foreign; known media on a track the plan has no segments on is not carried."""
        segs = self.plan.segments
        clips = {c.location: c for c in self.layout.clips}
        self.old_range = {
            i: (clips[f"segments[{i}]"].record_in, clips[f"segments[{i}]"].record_out)
            for i in range(len(segs))
        }
        #: video track label -> plan track number, for the tracks that hold planned segments
        self.plan_track = {f"V{track_of_lane(self.layout.lanes[s.track])}": s.track for s in segs}
        self.content: dict[str, list[_Item]] = {}
        self.assigned: dict[str, dict[int, int]] = {}
        self.alternatives: dict[int, list[int]] = {}  # id(_Item) -> other originals it overlaps
        labels = sorted(
            {e.track for e in self.items if e.is_video}, key=lambda t: (t != self.primary, t)
        )
        for label in labels:
            content: list[_Item] = []
            for entry in self._pictures(label):
                if entry.used or entry.is_title:
                    continue
                if entry.source is None:
                    entry.used = True
                    self.foreign.append({**entry.where(), "reason": entry.foreign})
                elif label in self.plan_track:
                    entry.used = True
                    content.append(entry)
            if label not in self.plan_track:
                continue
            track = self.plan_track[label]
            originals = [
                (i, s.source, s.src_in, s.src_out, self.old_range[i][0])
                for i, s in enumerate(segs)
                if s.track == track
            ]
            items = []
            for e in content:
                assert e.source is not None and e.src_in is not None and e.src_out is not None
                items.append((e.source, e.src_in, e.src_out, e.record_in))
            self.content[label] = content
            self.assigned[label], alternatives = assign_items(items, originals)
            for pos, others in alternatives.items():
                self.alternatives[id(content[pos])] = others

    def _segments(self) -> list[Segment]:
        """The edited segment list: V1 items in record order, each inheriting from its original;
        a segment on another track follows the V1 segment its original followed in the plan (an
        added one, the last V1 segment starting at or before it). Positions from the timeline."""
        segs = self.plan.segments
        self.pieces: dict[int, list[int]] = {}  # original index -> new indexes, record order
        self.content_index: dict[int, int] = {}  # id(_Item) -> new index
        primary = self.content.get(self.primary, [])
        v1_assigned = self.assigned.get(self.primary, {})
        last_piece: dict[int, int] = {}
        for pos in range(len(primary)):
            if pos in v1_assigned:
                last_piece[v1_assigned[pos]] = pos
        # the V1 original each off-V1 original followed in the plan, then its nearest survivor
        anchor_of: dict[int, int | None] = {}
        survivor: int | None = None
        for i, seg in enumerate(segs):
            if seg.track == 1:
                if i in last_piece:
                    survivor = i
                continue
            anchor_of[i] = survivor
        slots: dict[int | None, list[tuple[tuple[int, int, int], _Item, int | None, int]]] = {}
        for label, content in self.content.items():
            if label == self.primary:
                continue
            assigned = self.assigned[label]
            for pos, entry in enumerate(content):
                original = assigned.get(pos)
                if original is not None:
                    anchor = anchor_of.get(original)
                    slot = last_piece[anchor] if anchor is not None else None
                    order = (0, original, entry.record_in)
                else:
                    before = [p for p, e in enumerate(primary) if e.record_in <= entry.record_in]
                    slot = before[-1] if before else None
                    order = (1, entry.record_in, self.plan_track[label])
                slots.setdefault(slot, []).append((order, entry, original, self.plan_track[label]))

        out: list[Segment] = []
        desired: list[int] = []

        def emit(entry: _Item, original: int | None, track: int) -> None:
            assert entry.source is not None and entry.src_in is not None
            assert entry.src_out is not None
            if original is None:
                seg = Segment(
                    track=track, source=entry.source, src_in=entry.src_in, src_out=entry.src_out
                )
            else:
                base = segs[original]
                update: dict[str, Any] = {"src_in": entry.src_in, "src_out": entry.src_out}
                if (base.src_in, base.src_out) != (entry.src_in, entry.src_out):
                    update["reframe"] = window_reframe(
                        base.reframe, entry.src_in - base.src_in, entry.src_out - base.src_in
                    )
                seg = base.model_copy(update=update)
                self.pieces.setdefault(original, []).append(len(out))
            self.content_index[id(entry)] = len(out)
            out.append(seg)
            desired.append(entry.record_in)

        def emit_slot(slot: int | None) -> None:
            for _, entry, original, track in sorted(slots.get(slot, []), key=lambda t: t[0]):
                emit(entry, original, track)

        emit_slot(None)
        for pos, entry in enumerate(primary):
            emit(entry, v1_assigned.get(pos), 1)
            emit_slot(pos)
        for original in self.pieces:
            self.pieces[original].sort(key=lambda k: (desired[k], k))
        return _place(out, desired)

    def _transitions(self, segments: list[Segment]) -> list[Any]:
        """Plan transitions whose join survives, re-indexed; the others are dropped."""
        old_joins = dict(self.plan.joins())
        kept = []
        draft = _with_segments(self.plan, segments)
        new_joins = set(draft.joins())
        ranges = draft.record_ranges()
        for number, tx in enumerate(self.plan.transitions):
            nxt = old_joins.get(tx.after)
            outgoing = self.pieces.get(tx.after, [])
            incoming = self.pieces.get(nxt, []) if nxt is not None else []
            if outgoing and incoming:
                a, b = outgoing[-1], incoming[0]
                if (a, b) in new_joins and ranges[a][1] == ranges[b][0]:
                    kept.append(tx.model_copy(update={"after": a}))
                    continue
            self.drop(
                "transition_join_gone",
                f"transitions[{number}]",
                f"the join after segment {tx.after} no longer exists (its two segments are not "
                "neighbours that abut on their track)",
            )
        return kept

    def _outro(self, segments: list[Segment]) -> tuple[Any, dict[str, Any] | None]:
        outro = self.plan.outro
        if outro is None:
            return None, None
        source = self.plan.source_map()[outro.source]
        old_out = outro.src_out if outro.src_out is not None else source.duration
        content_end = _with_segments(self.plan, segments).content_frames()
        old_start = self.plan.content_frames()
        diff: dict[str, Any] = {
            "status": "unchanged",
            "old": {"src_in": outro.src_in, "src_out": old_out, "record_in": old_start},
            "new": None,
        }
        entry = self.outro_item
        if entry is None:
            diff["status"] = "missing"
            self.carry_not(
                "outro_missing",
                f"no item of the outro's source {outro.source!r} is on its track; the plan keeps "
                "its outro",
            )
            return outro, diff
        assert entry.src_in is not None and entry.src_out is not None
        diff["new"] = {
            "src_in": entry.src_in,
            "src_out": entry.src_out,
            "record_in": entry.record_in,
        }
        if entry.record_in != content_end:
            self.carry_not(
                "outro_position",
                f"the outro starts at frame {entry.record_in}, not where the content ends "
                f"({content_end}); a plan always places it there",
                entry.where(),
            )
        if (entry.src_in, entry.src_out) == (outro.src_in, old_out):
            if entry.record_in != old_start:
                diff["status"] = "moved"
            return outro, diff
        src_out = (
            None if outro.src_out is None and entry.src_out == source.duration else entry.src_out
        )
        sting = outro.sting
        if sting is not None and sting.start + sting.frames > entry.src_out - entry.src_in:
            diff["status"] = "kept"
            self.carry_not(
                "outro_changed",
                f"the outro was cut to {entry.src_out - entry.src_in} frames, shorter than its "
                f"sting (to frame {sting.start + sting.frames}); the plan keeps its outro",
                entry.where(),
            )
            return outro, diff
        diff["status"] = "trimmed"
        return outro.model_copy(update={"src_in": entry.src_in, "src_out": src_out}), diff

    # -- timing off the segments ---------------------------------------------------------------

    def _content_map(self, draft: EditPlan) -> Callable[[int], int | None]:
        """Old record frame -> new record frame of the same source frame of the same V1 segment
        (or of the outro), or None when that frame is not in the edited timeline."""
        segs = self.plan.segments
        ranges = draft.record_ranges()
        groups = self._join_groups()
        spans: list[tuple[int, int, int, list[tuple[int, int, int]]]] = []
        for i, seg in enumerate(segs):
            if seg.track != 1:
                continue
            old_in, old_out = self.old_range[i]
            pieces = [
                (draft.segments[k].src_in, draft.segments[k].src_out, ranges[k][0])
                for j in groups.get(i, [i])
                for k in self.pieces.get(j, [])
            ]
            spans.append((old_in, old_out, seg.src_in, pieces))
        old_outro, new_outro = self.plan.outro_range(), draft.outro_range()
        if old_outro is not None and new_outro is not None:
            assert self.plan.outro is not None and draft.outro is not None
            new_src_in = draft.outro.src_in
            pieces = [(new_src_in, new_src_in + new_outro[1] - new_outro[0], new_outro[0])]
            spans.append((old_outro[0], old_outro[1], self.plan.outro.src_in, pieces))

        def at(frame: int) -> int | None:
            for old_in, old_out, src_in, pieces in spans:
                if old_in <= frame < old_out:
                    source = src_in + frame - old_in
                    for a, b, record in pieces:
                        if a <= source < b:
                            return record + source - a
                    return None
            return None

        return at

    def _follow(
        self, at: Callable[[int], int | None], record_in: int, record_out: int
    ) -> tuple[int, int] | None:
        """A span re-anchored to its content: its first and last frames both survive, the same
        distance apart."""
        first, last = at(record_in), at(record_out - 1)
        if first is None or last is None or last - first != record_out - 1 - record_in:
            return None
        return first, last + 1

    def _titles(self, draft: EditPlan, at: Callable[[int], int | None]) -> list[Any]:
        """Titles are items: each plan title takes the record range of its title item (an exact
        match on its track, then its range re-anchored to its content, then, per track, the
        remaining titles in record order when as many items remain). A title with no item was
        removed in Resolve."""
        candidates = [e for e in self.items if e.is_title and not e.excluded and not e.used]
        expected: list[tuple[str, str, int, int]] = []
        for spec in self.layout.titles:
            label = f"V{track_of_lane(spec.lane)}"
            expected.append((spec.location, label, spec.record_in, spec.record_out))
        found: dict[str, _Item] = {}

        def take(location: str, entry: _Item) -> None:
            found[location] = entry
            entry.used = True

        for location, label, rin, rout in expected:
            hit = next(
                (
                    e
                    for e in candidates
                    if not e.used and (e.track, e.record_in, e.record_out) == (label, rin, rout)
                ),
                None,
            )
            if hit is not None:
                take(location, hit)
        for location, label, rin, rout in expected:
            if location in found:
                continue
            if location == "outro.sting":
                new_range = self._sting_range(draft)
            else:
                new_range = self._follow(at, rin, rout)
            if new_range is None:
                continue
            hit = next(
                (
                    e
                    for e in candidates
                    if not e.used and (e.track, e.record_in, e.record_out) == (label, *new_range)
                ),
                None,
            )
            if hit is not None:
                take(location, hit)
        for label in sorted({e[1] for e in expected}):
            missing = [e for e in expected if e[1] == label and e[0] not in found]
            left = [e for e in candidates if e.track == label and not e.used]
            if missing and len(missing) == len(left):
                for (location, *_), entry in zip(
                    sorted(missing, key=lambda e: (e[2], e[0])), left, strict=True
                ):
                    take(location, entry)
        total = draft.total_frames()
        titles = []
        for index, title in enumerate(self.plan.titles):
            location = f"titles[{index}]"
            entry = found.get(location)
            if entry is None:
                self.drop(
                    "title_removed",
                    location,
                    f"no title item on its track for {title.record_in}..{title.record_out}; "
                    "removed in Resolve",
                )
                continue
            if entry.record_out > total:
                self.drop(
                    "title_past_end",
                    location,
                    f"its item ends at frame {entry.record_out}, past the new end ({total})",
                )
                continue
            new = (entry.record_in, entry.record_out)
            self.moved(location, "timeline", (title.record_in, title.record_out), new)
            titles.append(title.model_copy(update={"record_in": new[0], "record_out": new[1]}))
        if "outro.sting" in [e[0] for e in expected]:
            entry = found.get("outro.sting")
            want = self._sting_range(draft)
            if entry is None or (entry.record_in, entry.record_out) != want:
                self.carry_not(
                    "sting_missing",
                    "no title item where the edited plan puts the outro's sting; the plan keeps "
                    "its sting",
                    {"location": "outro.sting"},
                )
        return titles

    @staticmethod
    def _sting_range(plan: EditPlan) -> tuple[int, int] | None:
        if plan.outro is None or plan.outro.sting is None:
            return None
        start = plan.content_frames() + plan.outro.sting.start
        return start, start + plan.outro.sting.frames

    def _reanchored(
        self, draft: EditPlan, at: Callable[[int], int | None]
    ) -> tuple[list[Any], list[Any]]:
        """Captions and markers are never timeline items: each moves with the content it sat
        over, and is dropped when that content is gone."""
        total = draft.total_frames()
        captions = []
        for index, caption in enumerate(self.plan.captions):
            location = f"captions[{index}]"
            new = self._follow(at, caption.record_in, caption.record_out)
            if new is None:
                self.drop(
                    "caption_content_gone",
                    location,
                    f"the content under {caption.record_in}..{caption.record_out} is not in the "
                    "edited timeline as one piece",
                )
            elif new[1] > total:
                self.drop("caption_past_end", location, f"ends past the new end ({total})")
            else:
                self.moved(location, "content", (caption.record_in, caption.record_out), new)
                captions.append(
                    caption.model_copy(update={"record_in": new[0], "record_out": new[1]})
                )
        markers = []
        for index, marker in enumerate(self.plan.markers):
            location = f"markers[{index}]"
            old = (marker.record, marker.record + marker.duration)
            new = self._follow(at, *old)
            if new is None:
                self.drop(
                    "marker_content_gone",
                    location,
                    f"the content under {old[0]}..{old[1]} is not in the edited timeline as one "
                    "piece",
                )
            elif new[1] > total:
                self.drop("marker_past_end", location, f"ends past the new end ({total})")
            else:
                self.moved(location, "content", old, new)
                markers.append(marker.model_copy(update={"record": new[0]}))
        return captions, markers

    def _join_groups(self) -> dict[int, list[int]]:
        """Neighbouring originals the owner joined into one item: each member -> the group.
        Two plan neighbours on one track, contiguous in source and record, where one has no
        piece and a piece of the other runs across their common source frame."""
        segs = self.plan.segments
        piece_src = self._piece_src
        group_of: dict[int, list[int]] = {}
        for a, b in self.plan.joins():
            sa, sb = segs[a], segs[b]
            if (
                sa.source != sb.source
                or sa.src_out != sb.src_in
                or self.old_range[a][1] != self.old_range[b][0]
            ):
                continue
            ga, gb = group_of.get(a, [a]), group_of.get(b, [b])
            pieces_a = [k for i in ga for k in self.pieces.get(i, [])]
            pieces_b = [k for i in gb for k in self.pieces.get(i, [])]
            if bool(pieces_a) == bool(pieces_b):
                continue
            boundary = sb.src_in
            ranges = [piece_src[k] for k in pieces_a + pieces_b]
            if any(lo < boundary < hi for lo, hi in ranges):
                merged = ga + [i for i in gb if i not in ga]
                for i in merged:
                    group_of[i] = merged
        return group_of

    # -- what remains in the timeline -----------------------------------------------------------

    def _leftovers(self) -> None:
        """Picture items nothing above took: known media on a track the plan has no segments on,
        and titles or generators the plan has no title for."""
        for entry in self.items:
            if entry.excluded or entry.used or not entry.is_picture:
                continue
            if entry.is_title:
                self.carry_not(
                    "title_added",
                    "a title, generator or solid the plan has no title for (its text is not read)",
                    entry.where(),
                )
            else:
                self.carry_not(
                    "other_picture_track",
                    f"an item on {entry.track}, a track the plan has no segments on; a plan "
                    "cannot place it",
                    entry.where(),
                )

    def _transition_items(self, plan: EditPlan) -> None:
        ranges = plan.record_ranges()
        expected: list[_Expected] = []
        first = next((i for i, s in enumerate(plan.segments) if s.track == 1), None)
        if (
            plan.head is not None
            and plan.head.video != "cut"
            and first is not None
            and ranges[first][0] == min(r[0] for r in ranges)
        ):
            expected.append(_Expected("head", ranges[first][0], plan.head.video_frames))
        for number, tx in enumerate(plan.transitions):
            if tx.video != "cut" and plan.segments[tx.after].track == 1:
                expected.append(
                    _Expected(f"transitions[{number}]", ranges[tx.after][1], tx.video_frames)
                )
        if (
            plan.outro is not None
            and plan.outro.track == 1
            and plan.outro.transition.video != "cut"
        ):
            expected.append(
                _Expected(
                    "outro.transition", plan.content_frames(), plan.outro.transition.video_frames
                )
            )
        for entry in self.items:
            if entry.kind != "transition" or not entry.is_video or entry.excluded:
                continue
            hit = next(
                (
                    e
                    for e in expected
                    if not e.found
                    and entry.track == self.primary
                    and entry.record_in <= e.cut <= entry.record_out
                    and entry.record_out - entry.record_in == e.frames
                ),
                None,
            )
            if hit is not None:
                hit.found = True
            else:
                self.carry_not(
                    "transition_added",
                    "a transition in the timeline that the edited plan has no matching transition "
                    "for (added in Resolve, or its length changed)",
                    entry.where(),
                )
        for e in expected:
            if not e.found:
                self.carry_not(
                    "transition_missing",
                    f"{e.location}: no transition item of {e.frames} frames at frame {e.cut}; the "
                    "plan keeps it (Resolve's import of FCPXML transitions is unconfirmed)",
                    {
                        "track": self.primary,
                        "record_in": e.cut,
                        "record_out": e.cut,
                        "location": e.location,
                    },
                )

    def _audio(self, plan: EditPlan) -> None:
        """Each sound must be a picture item's own linked sound: the same file, in sync, over the
        picture's range (widened only by an audio-only crossfade the plan carries at that cut)."""
        split_after = {
            tx.after: tx.audio.frames
            for tx in plan.transitions
            if tx.video == "cut" and tx.audio is not None and tx.audio.frames > 0
        }
        joins = dict(plan.joins())
        split_before = {joins[a]: f for a, f in split_after.items() if a in joins}
        pictures = [
            e
            for e in self.items
            if e.is_video and e.kind == "video" and not e.excluded and e.key is not None
        ]
        in_sync: dict[tuple[str | None, int], list[_Item]] = {}  # (file, source - record)
        for pic in pictures:
            assert pic.src_in is not None
            in_sync.setdefault((pic.key, pic.src_in - pic.record_in), []).append(pic)
        heard: set[int] = set()
        for entry in self.items:
            if entry.is_video or entry.excluded or entry.kind == "transition":
                continue
            ok = False
            offset = entry.src_in - entry.record_in if entry.src_in is not None else None
            for pic in in_sync.get((entry.key, offset), []) if offset is not None else []:
                index = self.content_index.get(id(pic))
                widen_in = split_before.get(index, 0) if index is not None else 0
                widen_out = split_after.get(index, 0) if index is not None else 0
                if (
                    pic.record_in - widen_in <= entry.record_in <= pic.record_in
                    and pic.record_out <= entry.record_out <= pic.record_out + widen_out
                ):
                    ok = True
                    heard.add(id(pic))
                    break
            if not ok:
                self.carry_not(
                    "audio_only_edit",
                    "a sound that is not a picture item's own linked sound (a separate audio "
                    "edit); the plan carries sound only with its pictures",
                    entry.where(),
                )
        for pic in pictures:
            clip = self.media.get(pic.item.media_clip_id or "")
            if pic.source is None or clip is None or not clip.audio_tracks or id(pic) in heard:
                continue
            self.carry_not(
                "audio_only_edit",
                "the picture's own sound is not in the timeline (removed or moved on its own); "
                "the plan plays it",
                pic.where(),
            )

    # -- the diff ------------------------------------------------------------------------------

    def _diff(self, edited: EditPlan, outro: dict[str, Any] | None) -> dict[str, Any]:
        new_ranges = edited.record_ranges()
        segs = self.plan.segments
        match_of = {self.content_index[id(e)]: e.match for c in self.content.values() for e in c}
        also_of = {
            self.content_index[i]: others
            for i, others in self.alternatives.items()
            if i in self.content_index
        }
        groups = self._join_groups()
        rows: list[dict[str, Any]] = []
        for i, seg in enumerate(segs):
            old_in, old_out = self.old_range[i]
            row: dict[str, Any] = {
                "index": i,
                "status": "",
                "track": seg.track,
                "source": seg.source,
                "old": {
                    "src_in": seg.src_in,
                    "src_out": seg.src_out,
                    "record_in": old_in,
                    "record_out": old_out,
                },
                "new": [
                    {
                        "index": k,
                        "src_in": edited.segments[k].src_in,
                        "src_out": edited.segments[k].src_out,
                        "record_in": new_ranges[k][0],
                        "record_out": new_ranges[k][1],
                        "match": match_of.get(k),
                        "also_overlaps": also_of.get(k, []),
                    }
                    for k in self.pieces.get(i, [])
                ],
                "head_frames": None,
                "tail_frames": None,
                "joined": None,
            }
            new = row["new"]
            group = groups.get(i)
            if group is not None:
                keeper = next(j for j in group if self.pieces.get(j))
                row["joined"] = {"segments": group, "fields_from": keeper}
            if group is not None and (not new or len(new) == 1):
                row["status"] = "joined"
                if new:
                    row["head_frames"] = new[0]["src_in"] - segs[group[0]].src_in
                    row["tail_frames"] = segs[group[-1]].src_out - new[0]["src_out"]
            elif not new:
                row["status"] = "removed"
            elif len(new) > 1:
                row["status"] = "split"
            elif (new[0]["src_in"], new[0]["src_out"]) != (seg.src_in, seg.src_out):
                row["status"] = "trimmed"
                row["head_frames"] = new[0]["src_in"] - seg.src_in
                row["tail_frames"] = seg.src_out - new[0]["src_out"]
            elif new[0]["record_in"] != old_in:
                row["status"] = "moved"
            else:
                row["status"] = "unchanged"
            rows.append(row)
        added = []
        for label, content in self.content.items():
            for pos, entry in enumerate(content):
                if pos in self.assigned[label]:
                    continue
                k = self.content_index[id(entry)]
                added.append(
                    {
                        "index": k,
                        "track": edited.segments[k].track,
                        "source": entry.source,
                        "match": entry.match,
                        "src_in": entry.src_in,
                        "src_out": entry.src_out,
                        "record_in": new_ranges[k][0],
                        "record_out": new_ranges[k][1],
                    }
                )
        added.sort(key=lambda a: a["index"])
        uncertain = sorted(
            (
                {"index": k, "assigned_to": i, "also_overlaps": also_of[k]}
                for i, ks in self.pieces.items()
                for k in ks
                if also_of.get(k)
            ),
            key=lambda u: u["index"],
        )
        by_name = [
            {**e.where(), "source": e.source}
            for e in self.items
            if e.match == "file-name" and e.source is not None
        ]
        matches = sorted(
            {e.match for e in self.items if e.source is not None and e.match is not None}
        )
        unchanged = (
            all(r["status"] == "unchanged" for r in rows)
            and not added
            and not self.foreign
            and not self.not_carried
            and not self.dropped
            and not self.repositioned
            and (outro is None or outro["status"] == "unchanged")
        )
        tl = self.timeline
        return {
            "format": FORMAT,
            "unchanged": unchanged,
            "timeline": {
                "name": tl.name,
                "frame_rate": tl.frame_rate,
                "start_frame": tl.start_frame,
                "items": len(self.items),
            },
            "primary_track": self.primary,
            "media_matched_by": matches,
            "matched_by_file_name": by_name,
            "segments": rows,
            "uncertain_matches": uncertain,
            "added": added,
            "outro": outro,
            "repositioned": self.repositioned,
            "foreign": self.foreign,
            "not_carried": self.not_carried,
            "dropped": self.dropped,
            "limits": list(LIMITS),
            "assumptions": [dict(a) for a in ASSUMPTIONS],
        }

    @property
    def _piece_src(self) -> dict[int, tuple[int, int]]:
        """New segment index -> its source range (from the matched items)."""
        out: dict[int, tuple[int, int]] = {}
        for content in self.content.values():
            for entry in content:
                assert entry.src_in is not None and entry.src_out is not None
                out[self.content_index[id(entry)]] = (entry.src_in, entry.src_out)
        return out


def _place(segments: list[Segment], desired: list[int]) -> list[Segment]:
    """Give each segment the record start it has in the timeline: its own ``record_in`` when that
    is still right, none when it follows the previous segment on its track, else explicit."""
    cursor: dict[int, int] = {}
    out = []
    for seg, start in zip(segments, desired, strict=True):
        if seg.record_in is not None and seg.record_in == start:
            placed = seg
        elif cursor.get(seg.track, 0) == start:
            placed = seg if seg.record_in is None else seg.model_copy(update={"record_in": None})
        else:
            placed = seg.model_copy(update={"record_in": start})
        cursor[seg.track] = start + placed.length
        out.append(placed)
    return out


def export_check(plan: EditPlan, media_map: MediaMap | None) -> dict[str, Any]:
    """The export's layout step on ``plan``, with the options the import laid out the original
    with: ``{"ok": True, "error": None}`` or the export's refusal."""
    try:
        build_layout(
            plan,
            media_map=media_map,
            allow_unknown_timecode=True,
            resolve_media=_resolver(media_map),
        )
    except ExportError as exc:
        return {
            "ok": False,
            "error": {"code": exc.code, "message": exc.message, "fix_hint": exc.fix_hint},
        }
    return {"ok": True, "error": None}


def import_timeline(
    plan: EditPlan,
    timeline: Timeline,
    media: Mapping[str, MediaClip],
    *,
    media_map: MediaMap | None = None,
    profile: Profile | None = None,
    reader_warnings: Sequence[Issue] = (),
) -> ImportResult:
    """The edited plan and the diff of ``timeline`` against ``plan``, or a typed refusal
    (:class:`TimelineImportError`, or the export's :class:`ExportError` when the plan itself
    cannot be laid out). With ``profile`` the edited plan is linted. Lint errors and an export
    layout refusal of the edited plan do not stop the import (the owner's edit stands); they set
    ``lint_errors`` / ``export_errors`` in the diff."""
    importer = _Importer(plan, timeline, media, media_map)
    edited, diff = importer.run()
    report = lint(edited, profile) if profile is not None else None
    diff["lint"] = report.model_dump(mode="json") if report is not None else None
    diff["lint_errors"] = report is not None and not report.ok
    check = export_check(edited, media_map)
    diff["export_check"] = check
    diff["export_errors"] = not check["ok"]
    listed = [
        {"severity": i.severity, "code": i.code, "message": i.message}
        for i in list(reader_warnings)[:MAX_LISTED_ISSUES]
    ]
    diff["reader_warnings"] = {"count": len(reader_warnings), "listed": listed}
    return ImportResult(edited, diff, report)
