"""Timeline import (S20): a timeline edited in Resolve back as an EditPlan and a diff.

The flow: plan -> FCPXML (``aar plan export``) -> Resolve -> the owner edits -> File > Export >
Timeline as ``.drt`` -> :func:`read_timeline` + :func:`import_timeline` -> an edited plan and a
diff the agent reads. Pure and cold: nothing here talks to Resolve or the network, and the only
file read is the ``.drt`` (through :mod:`aar.drp`, with its own size limits).

* **Expected timeline.** S18's :func:`~aar.plan.export.layout.build_layout` is the original plan
  as timeline clips (lanes, record ranges, source ranges); nothing here re-derives its rules.
* **Media.** An item's media path is compared with the edit-PC path the export named for each
  plan source (:func:`~aar.plan.export.media.media_ref` with the export's media map, hub root ->
  edit-PC root), separators unified and case folded. With no path hit, a file name unique among
  the plan's media files (case folded). Anything else is foreign: reported, never guessed.
* **Segments.** The picture items on the primary track (V1), in record order, become the plan's
  segments, frame-exact. Each is matched to the original V1 segment of its source that it
  overlaps most (an exact range first); it inherits that segment's other fields. Reframe keys are
  measured from the segment's start, so a segment whose source range changed gets its keys
  windowed to the new range: shifted by the head trim, and a side whose keys were cut away gets
  one key interpolated (linearly, as the model says) at the new boundary.
* **Everything else** in the plan is carried when it is still valid, and listed under ``dropped``
  when it is not. What the timeline holds that a plan cannot (retimes, sub-frame positions, audio
  edits, changes off V1, transitions added in Resolve) is listed under ``not_carried`` with a
  reason code; the plan keeps its own version there.

The three assumptions that hold until a real round trip through Resolve confirms them are the
three functions under "Assumptions" below (DESIGN.md, "Timeline import (S20)").
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar
from urllib.parse import unquote, urlsplit

from aar.drp import read_drp
from aar.drp.errors import DrpError
from aar.drp.model import MediaClip, Timeline, TimelineItem
from aar.plan.apply.errors import CompileError
from aar.plan.export.errors import ExportError
from aar.plan.export.layout import Layout, build_layout
from aar.plan.export.media import MediaMap, MediaRef, media_ref
from aar.plan.lint import LintReport, lint
from aar.plan.model import EditPlan, ReframeKey, Segment, Source
from aar.plan.profile import Profile

FORMAT = "aar.timeline-import/1"
#: Items read from one timeline, all tracks together; more is refused (``timeline_too_large``).
MAX_ITEMS = 20_000
#: A timeline or media rate (a float in the file) matches the plan's rational fps within this.
FPS_TOLERANCE = 1e-3

IMPORT_CODES = frozenset(
    {
        "unreadable_timeline",
        "no_timeline",
        "several_timelines",
        "timeline_not_found",
        "timeline_too_large",
        "fps_mismatch",
        "media_fps_mismatch",
        "source_range_outside",
        "output_exists",
    }
)

#: Reason codes of ``not_carried`` (timeline changes a plan cannot hold) and ``dropped`` (plan
#: entries no longer valid after the edit). Pinned by the tests.
NOT_CARRIED_CODES = (
    "retimed",
    "sub_frame",
    "unreadable_item",
    "audio_only_edit",
    "other_picture_track",
    "transition_added",
    "transition_missing",
    "outro_missing",
    "outro_position",
    "outro_changed",
)
DROPPED_CODES = ("transition_join_gone", "title_past_end", "caption_past_end", "marker_past_end")

LIMITS = (
    "enabled_state: the reader cannot see whether an item is enabled; a disabled clip is "
    "imported as if it plays",
    "audio_transitions: transition items on audio tracks are not compared",
    "titles: title items are compared by track and record range only; their text is not read",
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
        "text": "an item's source in point counts frames from the media file's first frame, as "
        "the plan's src_in does (not from its start timecode)",
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
    """Assumption ``source_frames``: the item's ``<In>`` is the plan's ``src_in`` (frames from the
    file's first frame)."""
    return item.source_in


def media_key(path: str) -> str:
    """Assumption ``media_path``: an item's media path is the path the FCPXML asset named, so it
    is compared with the export's edit-PC path, Windows-style (``/`` as ``\\``, case folded)."""
    return path.replace("/", "\\").casefold()


# -- reading ------------------------------------------------------------------------------------


@dataclass
class TimelineFile:
    """One timeline read from a ``.drt`` with :mod:`aar.drp`, and its media-pool clips."""

    path: str
    sha256: str
    timeline: Timeline
    media: dict[str, MediaClip]


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
    return TimelineFile(str(path), drp.sha256, timeline, drp.media_by_id())


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
    excluded: bool = False

    @property
    def is_video(self) -> bool:
        return self.track.startswith("V")

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

    # -- report helpers ------------------------------------------------------------------------

    def carry_not(self, code: str, detail: str, where: dict[str, Any] | None = None) -> None:
        assert code in NOT_CARRIED_CODES
        self.not_carried.append({"code": code, **(where or {}), "detail": detail})

    def drop(self, code: str, location: str, detail: str) -> None:
        assert code in DROPPED_CODES
        self.dropped.append({"code": code, "location": location, "detail": detail})

    # -- steps ---------------------------------------------------------------------------------

    def run(self) -> tuple[EditPlan, dict[str, Any]]:
        self._check_rate()
        self.layout: Layout = build_layout(
            self.plan,
            media_map=self.media_map,
            allow_unknown_timecode=True,
            resolve_media=_resolver(self.media_map),
        )
        self._index_sources()
        self._collect()
        self._match_media()
        self._check_matched()
        self.primary = f"V{track_of_lane(0)}"
        self._find_outro()
        self._match_segments()
        segments = self._segments()
        transitions = self._transitions(segments)
        outro, outro_diff = self._outro(segments)
        draft = self.plan.model_copy(
            update={"segments": segments, "transitions": transitions, "outro": outro}
        )
        titles, captions, markers = self._timed(draft)
        edited = EditPlan.model_validate(
            draft.model_copy(
                update={"titles": titles, "captions": captions, "markers": markers}
            ).model_dump()
        )
        self._other_tracks()
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
        resolve = _resolver(self.media_map)
        self.refs = {s.id: self.layout.media.get(s.id) or resolve(s) for s in self.plan.sources}
        self.by_path: dict[str, list[str]] = {}
        files_by_name: dict[str, set[str]] = {}
        self.by_name: dict[str, list[str]] = {}
        for source in self.plan.sources:
            ref = self.refs[source.id]
            if ref.windows_path:
                self.by_path.setdefault(media_key(ref.windows_path), []).append(source.id)
            name = ref.name.casefold()
            ident = media_key(ref.windows_path) if ref.windows_path else source.media
            files_by_name.setdefault(name, set()).add(ident)
            self.by_name.setdefault(name, []).append(source.id)
        self.unique_names = {n for n, files in files_by_name.items() if len(files) == 1}

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
                if item.record_start is None or item.duration is None:
                    entry.excluded = True
                    self.carry_not(
                        "unreadable_item",
                        "the item's start or duration could not be read",
                        {"track": label, "kind": item.kind, "name": item.name},
                    )
                    continue
                entry.record_in = item.record_start - start
                entry.record_out = entry.record_in + item.duration
                if item.kind == "transition":
                    continue
                src_in = source_frame(item)
                entry.src_in = src_in
                entry.src_out = src_in + item.duration if src_in is not None else None
                if item.fractions:
                    entry.excluded = True
                    fields = ", ".join(sorted(item.fractions))
                    self.carry_not(
                        "sub_frame",
                        f"the item sits at a sub-frame position ({fields}); a plan holds whole "
                        "frames",
                        entry.where(),
                    )
                elif item.timemap_form == "retimed" or (
                    item.speed is not None and item.speed != 1.0
                ):
                    entry.excluded = True
                    self.carry_not(
                        "retimed",
                        f"the item is retimed (speed {item.speed!r}); a plan plays sources at "
                        "100 %",
                        entry.where(),
                    )
                elif src_in is None and item.media_path:
                    entry.excluded = True
                    self.carry_not(
                        "unreadable_item",
                        "the item's source in point could not be read",
                        entry.where(),
                    )

    def _match_media(self) -> None:
        for entry in self.items:
            path = entry.item.media_path
            if entry.kind == "transition" or not path:
                continue
            entry.key = media_key(path)
            ids = self.by_path.get(entry.key)
            if ids:
                entry.match = "path"
            else:
                name = re.split(r"[\\/]", path)[-1].casefold()
                if name in self.unique_names:
                    ids = self.by_name[name]
                    entry.match = "file-name"
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

    def _picture(self, track: str) -> list[_Item]:
        return [
            e
            for e in self.items
            if e.track == track and not e.excluded and e.kind not in ("transition", "audio")
        ]

    def _find_outro(self) -> None:
        self.outro_item: _Item | None = None
        outro = self.plan.outro
        if outro is None:
            return
        track = f"V{track_of_lane(self.layout.lanes[outro.track])}"
        found = [e for e in self._picture(track) if e.source == outro.source]
        if found:
            self.outro_item = found[-1]

    def _match_segments(self) -> None:
        """Assign each V1 picture item to an original V1 segment: an exact range at its expected
        position, then an exact range anywhere, then the largest overlap (preferring segments not
        yet matched exactly). An item that matches nothing of its source is added."""
        self.originals = [i for i, s in enumerate(self.plan.segments) if s.track == 1]
        clips = {c.location: c for c in self.layout.clips}
        self.old_range = {
            i: (clips[f"segments[{i}]"].record_in, clips[f"segments[{i}]"].record_out)
            for i in range(len(self.plan.segments))
        }
        self.content: list[_Item] = []
        for entry in self._picture(self.primary):
            if entry is self.outro_item:
                continue
            if entry.source is None:
                self.foreign.append(entry.where())
                continue
            self.content.append(entry)
        self.assigned: dict[int, int] = {}  # content position -> original segment index
        exact: set[int] = set()
        segs = self.plan.segments

        def same(i: int, e: _Item) -> bool:
            s = segs[i]
            return (s.source, s.src_in, s.src_out) == (e.source, e.src_in, e.src_out)

        for at_position in (True, False):
            for pos, entry in enumerate(self.content):
                if pos in self.assigned:
                    continue
                for i in self.originals:
                    if i in exact or not same(i, entry):
                        continue
                    if at_position and self.old_range[i][0] != entry.record_in:
                        continue
                    self.assigned[pos] = i
                    exact.add(i)
                    break
        for pos, entry in enumerate(self.content):
            if pos in self.assigned:
                continue
            assert entry.src_in is not None and entry.src_out is not None
            best: tuple[int, int, int] | None = None  # (not exact, overlap, -index)
            for i in self.originals:
                s = segs[i]
                if s.source != entry.source:
                    continue
                overlap = min(s.src_out, entry.src_out) - max(s.src_in, entry.src_in)
                if overlap <= 0:
                    continue
                rank = (i not in exact, overlap, -i)
                if best is None or rank > best:
                    best = rank
            if best is not None and best[0]:
                self.assigned[pos] = -best[2]

    def _segments(self) -> list[Segment]:
        """The edited segment list: V1 items in record order, each inheriting from its original,
        with the off-V1 originals kept after the V1 segment they followed in the plan."""
        segs = self.plan.segments
        self.pieces: dict[int, list[int]] = {}  # original index -> new indexes, record order
        self.content_index: dict[int, int] = {}  # id(_Item) -> new index
        anchored: dict[int | None, list[int]] = {}
        anchor: int | None = None
        for i, seg in enumerate(segs):
            if seg.track == 1:
                anchor = i
            else:
                anchored.setdefault(anchor, []).append(i)
        last_piece: dict[int, int] = {}
        for pos, _ in enumerate(self.content):
            if pos in self.assigned:
                last_piece[self.assigned[pos]] = pos
        # an off-V1 segment whose anchor left the timeline follows the nearest earlier survivor
        survivors: dict[int | None, int | None] = {None: None}
        alive: int | None = None
        for i in self.originals:
            if i in last_piece:
                alive = i
            survivors[i] = alive
        after_pos: dict[int | None, list[int]] = {}
        for key, ids in anchored.items():
            target = survivors.get(key)
            pos_key = last_piece[target] if target is not None else None
            after_pos.setdefault(pos_key, []).extend(ids)

        out: list[Segment] = []
        desired: list[int] = []

        def keep_off_v1(ids: Iterable[int]) -> None:
            for i in sorted(ids):
                out.append(segs[i])
                desired.append(self.old_range[i][0])
                self.pieces.setdefault(i, []).append(len(out) - 1)

        keep_off_v1(after_pos.get(None, []))
        for pos, entry in enumerate(self.content):
            assert entry.source is not None and entry.src_in is not None
            assert entry.src_out is not None
            original = self.assigned.get(pos)
            if original is None:
                seg = Segment(
                    track=1, source=entry.source, src_in=entry.src_in, src_out=entry.src_out
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
            keep_off_v1(after_pos.get(pos, []))
        return _place(out, desired)

    def _transitions(self, segments: list[Segment]) -> list[Any]:
        """Plan transitions whose join survives, re-indexed; the others are dropped."""
        old_joins = dict(self.plan.joins())
        kept = []
        new_joins = set(_joins(segments))
        ranges = _ranges(segments)
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
        content_end = max((end for _, end in _ranges(segments)), default=0)
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

    def _timed(self, draft: EditPlan) -> tuple[list[Any], list[Any], list[Any]]:
        total = draft.total_frames()
        titles, captions, markers = [], [], []
        for kind, items, out, code in (
            ("titles", self.plan.titles, titles, "title_past_end"),
            ("captions", self.plan.captions, captions, "caption_past_end"),
        ):
            for index, entry in enumerate(items):
                if entry.record_out <= total:
                    out.append(entry)
                else:
                    self.drop(
                        code,
                        f"{kind}[{index}]",
                        f"ends at frame {entry.record_out}, past the new end ({total})",
                    )
        for index, marker in enumerate(self.plan.markers):
            if marker.record + marker.duration <= total:
                markers.append(marker)
            else:
                self.drop(
                    "marker_past_end",
                    f"markers[{index}]",
                    f"ends at frame {marker.record + marker.duration}, past the new end ({total})",
                )
        return titles, captions, markers

    def _other_tracks(self) -> None:
        """Off V1 the plan keeps its own version: an item there must be one the layout predicts
        (a clip with the same source and ranges, or a title at the same range)."""
        expected: list[tuple[str, str, int, int, str | None, int | None, int | None]] = []
        for clip in self.layout.clips:
            if clip.lane == 0 or clip.location == "outro":
                continue
            index = int(clip.location[len("segments[") : -1])
            seg = self.plan.segments[index]
            expected.append(
                (
                    clip.location,
                    f"V{track_of_lane(clip.lane)}",
                    clip.record_in,
                    clip.record_out,
                    seg.source,
                    clip.src_in,
                    clip.src_out,
                )
            )
        for title in self.layout.titles:
            expected.append(
                (
                    title.location,
                    f"V{track_of_lane(title.lane)}",
                    title.record_in,
                    title.record_out,
                    None,
                    None,
                    None,
                )
            )
        found: set[int] = set()
        self.off_v1_changed: set[int] = set()
        for entry in self.items:
            if (
                not entry.is_video
                or entry.track == self.primary
                or entry.excluded
                or entry.kind == "transition"
                or entry is self.outro_item
            ):
                continue
            hit = None
            for n, (_, track, rin, rout, source, sin, sout) in enumerate(expected):
                if (
                    n in found
                    or track != entry.track
                    or (rin, rout) != (entry.record_in, entry.record_out)
                ):
                    continue
                if (
                    source is None
                    and entry.item.media_path is not None
                    and entry.source is not None
                ):
                    continue
                if source is not None and (source, sin, sout) != (
                    entry.source,
                    entry.src_in,
                    entry.src_out,
                ):
                    continue
                hit = n
                break
            if hit is not None:
                found.add(hit)
            elif entry.item.media_path and entry.source is None:
                self.foreign.append(entry.where())
            else:
                self.carry_not(
                    "other_picture_track",
                    f"an item on {entry.track} that the plan does not have there; changes off "
                    f"{self.primary} are not carried",
                    entry.where(),
                )
        for n, (location, track, rin, rout, _, _, _) in enumerate(expected):
            if n in found:
                continue
            if location.startswith("segments["):
                self.off_v1_changed.add(int(location[len("segments[") : -1]))
            self.carry_not(
                "other_picture_track",
                f"{location} is not on {track} at {rin}..{rout} as planned; the plan keeps it",
                {"track": track, "record_in": rin, "record_out": rout, "location": location},
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
        heard: set[int] = set()
        for entry in self.items:
            if entry.is_video or entry.excluded or entry.kind == "transition":
                continue
            ok = False
            for pic in pictures:
                if pic.key != entry.key or entry.src_in is None or pic.src_in is None:
                    continue
                if entry.src_in - entry.record_in != pic.src_in - pic.record_in:
                    continue
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
                    }
                    for k in self.pieces.get(i, [])
                ],
                "head_frames": None,
                "tail_frames": None,
            }
            new = row["new"]
            if seg.track != 1:
                row["status"] = "kept_off_primary" if i in self.off_v1_changed else "unchanged"
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
        for pos, entry in enumerate(self.content):
            if pos in self.assigned:
                continue
            k = self.content_index[id(entry)]
            added.append(
                {
                    "index": k,
                    "source": entry.source,
                    "match": entry.match,
                    "src_in": entry.src_in,
                    "src_out": entry.src_out,
                    "record_in": new_ranges[k][0],
                    "record_out": new_ranges[k][1],
                }
            )
        matches = sorted(
            {e.match for e in self.items if e.source is not None and e.match is not None}
        )
        unchanged = (
            all(r["status"] == "unchanged" for r in rows)
            and not added
            and not self.foreign
            and not self.not_carried
            and not self.dropped
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
            "segments": rows,
            "added": added,
            "outro": outro,
            "foreign": self.foreign,
            "not_carried": self.not_carried,
            "dropped": self.dropped,
            "limits": list(LIMITS),
            "assumptions": [dict(a) for a in ASSUMPTIONS],
        }


def _joins(segments: list[Segment]) -> list[tuple[int, int]]:
    last: dict[int, int] = {}
    pairs = []
    for index, seg in enumerate(segments):
        if seg.track in last:
            pairs.append((last[seg.track], index))
        last[seg.track] = index
    return pairs


def _ranges(segments: list[Segment]) -> list[tuple[int, int]]:
    cursor: dict[int, int] = {}
    out = []
    for seg in segments:
        start = seg.record_in if seg.record_in is not None else cursor.get(seg.track, 0)
        cursor[seg.track] = start + seg.length
        out.append((start, start + seg.length))
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


def import_timeline(
    plan: EditPlan,
    timeline: Timeline,
    media: Mapping[str, MediaClip],
    *,
    media_map: MediaMap | None = None,
    profile: Profile | None = None,
) -> ImportResult:
    """The edited plan and the diff of ``timeline`` against ``plan``, or a typed refusal
    (:class:`TimelineImportError`, or the export's :class:`ExportError` when the plan itself
    cannot be laid out). With ``profile`` the edited plan is linted; lint errors do not stop the
    import (the owner's edit stands), they set ``lint_errors`` in the diff."""
    importer = _Importer(plan, timeline, media, media_map)
    edited, diff = importer.run()
    report = lint(edited, profile) if profile is not None else None
    diff["lint"] = report.model_dump(mode="json") if report is not None else None
    diff["lint_errors"] = report is not None and not report.ok
    return ImportResult(edited, diff, report)
