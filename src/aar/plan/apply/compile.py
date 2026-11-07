"""Compile a linted EditPlan v2 into ordered Resolve build-op steps (phase A and phase B).

Nothing here talks to Resolve. Every op's params are built from the op's own validated model,
so a step that compiles is a step the op (and ``aar hub job submit``) will accept.

Why two phases
--------------
The finishing ops (``transition.add``, ``item.set_color`` and the rest) address a timeline item
by its ``item_id``, which Resolve only mints when the clip is appended. So phase A branches,
imports, ensures tracks and appends; phase B addresses the items phase A created, matched from
phase A's *verified* append results by ``(track, record_in)``.

Run-time-id seams (see PROGRESS.md gaps)
----------------------------------------
``ops.sequence`` runs each step with static params and does not thread one step's result into
the next. Three ids are therefore run-time values the compiler cannot know from the plan alone:

* the **template's** ``unique_id`` that ``timeline.branch`` needs -- supplied by
  ``--template-describe`` (a ``timeline.describe`` of the template);
* the **branched working timeline's** identity, which every later step must name -- taken from a
  phase-A append result or a ``--branch-describe``; without one the dry-run uses a named
  placeholder and ``--submit`` refuses (see :func:`~aar.plan.apply.submit.build_submission`);
* each **imported clip's** ``unique_id`` that ``timeline.append_ranges`` needs -- which
  ``media.import`` only returns at run time. The compiler keys append media by the plan's
  ``Source.media`` string (what the rest of ``aar`` already treats as the clip reference); a
  faithful run needs a job kind that threads the import result. Listed as a gap; no op is added.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, get_args

from pydantic import TypeAdapter, ValidationError

from aar.node.ops.build import (
    AudioFormat,  # noqa: F401  (imported so build.py registers before we read its models)
    CdlSpec,
    ColorParams,
    ColorTarget,
    EditPath,
    ImportParams,
    TitleParams,
    TitleSpec,
    TrackParams,
    TrackSpec,
    TransformParams,
    TransformTarget,
    TransitionParams,
    TransitionTarget,
    path_key,
)
from aar.node.ops.collab import BranchParams
from aar.node.ops.framework import ItemRef, MediaRef, TimelineRef
from aar.node.ops.read import DescribeResult
from aar.node.ops.write import (
    AppendParams,
    AppendRange,
    AppendResult,
    FadesParams,
    FadeTarget,
    GainParams,
    GainTarget,
    MarkerColor,
    MarkersParams,
    MarkerSpec,
    PlacedItem,
)
from aar.plan.apply.errors import CompileError
from aar.plan.apply.model import FinishIntent, SegmentPlacement, Step
from aar.plan.model import Cdl, EditPlan, Source

DEFAULT_BATCH = 10
MAX_BATCH = 10
"""``AppendToTimeline`` hung Resolve 21.1 on large batches (2026-11-05): never more than 10
ranges per ``timeline.append_ranges`` step."""

TEMPLATE_MAX_ITEMS = 30
"""``DuplicateTimeline`` hung 21.1 on a 91-item timeline (2026-11-04). A template is small;
refuse to branch one over 30 items."""

IMPORT_ROOT = "aar"
"""Top media-pool bin the compiler imports source media under: ``aar/<camera>``."""

_WORKING_SUFFIX = "agent branch"

_EDIT_PATH = TypeAdapter(EditPath)
_VIDEO_TRANSITIONS = {
    "dissolve": "Cross Dissolve",
    "dip": "Dip To Color Dissolve",
    "wipe": "Wipe",
    "push": "Push",
}
_MARKER_COLORS = frozenset(get_args(MarkerColor))


@dataclass(frozen=True)
class BranchRef:
    """The branched working timeline's identity, from a phase-A result or ``--branch-describe``."""

    name: str
    unique_id: str

    def ref(self) -> TimelineRef:
        return TimelineRef(name=self.name, unique_id=self.unique_id)


@dataclass
class BuildCompilation:
    """Phase A: the ordered steps plus where each segment landed and any gaps found."""

    steps: list[Step]
    placements: list[SegmentPlacement]
    notes: list[str]
    working: TimelineRef
    template_items: int | None
    batch_size: int


@dataclass
class FinishCompilation:
    """Phase B: the finishing intents, the steps (empty until an item id resolves) and gaps."""

    intents: list[FinishIntent]
    steps: list[Step]
    notes: list[str]
    resolved: bool


def plan_hash(plan: EditPlan) -> str:
    """A stable sha256 over the plan's canonical JSON, for idempotency keys."""
    canonical = json.dumps(
        plan.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def template_item_count(describe: DescribeResult) -> int:
    """Total timeline items the template describe reports, across all tracks."""
    return sum(track.item_count for track in describe.tracks)


# -- phase A -----------------------------------------------------------------------------------


def compile_build(
    plan: EditPlan,
    *,
    template: DescribeResult | None = None,
    branch: BranchRef | None = None,
    batch_size: int = DEFAULT_BATCH,
) -> BuildCompilation:
    """Compile phase A: branch the template, import media, ensure tracks, append the ranges."""
    if not 1 <= batch_size <= MAX_BATCH:
        raise CompileError(
            "batch_size_invalid",
            f"batch size {batch_size} is out of range (1..{MAX_BATCH})",
            f"AppendToTimeline hung 21.1 on large batches; keep it at 1..{MAX_BATCH}",
            batch_size=batch_size,
        )
    if plan.template is None:
        raise CompileError(
            "missing_template",
            "the plan has no template to build from",
            "set the plan's 'template' (the timeline timeline.branch starts from)",
        )

    notes: list[str] = []
    template_items = _check_template(template, notes)
    working = _working_ref(plan, branch)
    if branch is None:
        notes.append(
            "working timeline named by placeholder; the real branch id is a run-time value "
            "threaded from a phase-A result or --branch-describe"
        )

    steps: list[Step] = []
    steps.append(_branch_step(plan, template))
    steps.extend(_import_steps(plan, notes))
    steps.append(_track_step(plan, working, notes))

    append_steps, placements = _append_steps(plan, working, batch_size, notes)
    steps.extend(append_steps)
    return BuildCompilation(steps, placements, notes, working, template_items, batch_size)


def _check_template(template: DescribeResult | None, notes: list[str]) -> int | None:
    if template is None:
        notes.append("template size unchecked (no --template-describe)")
        return None
    items = template_item_count(template)
    if items > TEMPLATE_MAX_ITEMS:
        raise CompileError(
            "template_too_large",
            f"the template has {items} items, over the {TEMPLATE_MAX_ITEMS} limit",
            "DuplicateTimeline hung 21.1 on a 91-item timeline (2026-11-04); branch a small "
            "template",
            items=items,
            limit=TEMPLATE_MAX_ITEMS,
        )
    return items


def _working_ref(plan: EditPlan, branch: BranchRef | None) -> TimelineRef:
    if branch is not None:
        return branch.ref()
    return TimelineRef(name=f"{plan.template} · {_WORKING_SUFFIX}")


def _branch_step(plan: EditPlan, template: DescribeResult | None) -> Step:
    if template is not None:
        source = TimelineRef(name=template.timeline.name, unique_id=template.timeline.unique_id)
        where = f"unique_id {template.timeline.unique_id}"
    else:
        source = TimelineRef(name=plan.template)
        where = f"name {plan.template!r} (no unique_id: dry-run only)"
    params = BranchParams(source=source, if_exists="reuse")
    return _step("A", "timeline.branch", "branch", params, f"branch {where}, reuse")


def _import_bin(source: Source) -> tuple[str, ...]:
    return (IMPORT_ROOT, source.camera or "sources")


def _is_edit_path(media: str) -> bool:
    try:
        _EDIT_PATH.validate_python(media)
    except ValidationError:
        return False
    return True


def _import_steps(plan: EditPlan, notes: list[str]) -> list[Step]:
    bins: dict[tuple[str, ...], list[str]] = {}
    for source in plan.sources:
        if not _is_edit_path(source.media):
            continue
        paths = bins.setdefault(_import_bin(source), [])
        if all(path_key(p) != path_key(source.media) for p in paths):
            paths.append(source.media)
    if not bins:
        notes.append("no sources are edit-PC paths; media is assumed present in the pool")
    steps: list[Step] = []
    for bin_path, paths in bins.items():
        for batch_index, batch in enumerate(_chunk(paths, 50)):
            params = ImportParams(
                paths=batch, bin_path=list(bin_path), create_bins=True, if_present="reuse"
            )
            name = f"import:{'/'.join(bin_path)}:{batch_index}"
            summary = f"import {len(batch)} file(s) into {'/'.join(bin_path)}"
            steps.append(_step("A", "media.import", name, params, summary))
    return steps


def _video_tracks(plan: EditPlan) -> list[int]:
    tracks = {seg.track for seg in plan.segments}
    if plan.outro is not None:
        tracks.add(plan.outro.track)
    return sorted(tracks)


def _title_track(plan: EditPlan) -> int:
    """Titles and the sting overlay above the content: one track past the top content track."""
    content = _video_tracks(plan) or [1]
    return max(content) + 1


def _track_step(plan: EditPlan, working: TimelineRef, notes: list[str]) -> Step:
    specs: list[TrackSpec] = []
    top_video = _video_tracks(plan)
    for index in range(1, (max(top_video, default=0) or 1) + 1):
        specs.append(TrackSpec(type="video", index=index))
    if plan.titles or (plan.outro is not None and plan.outro.sting is not None):
        specs.append(TrackSpec(type="video", index=_title_track(plan)))
    manual: list[str] = []
    for index, track in enumerate(plan.audio_tracks, start=1):
        specs.append(TrackSpec(type="audio", index=index, name=track.name, enabled=not track.mute))
        if track.gain_db or track.bus is not None:
            manual.append(track.name)
    if manual:
        notes.append(
            f"audio track gain_db/bus/bounce are a manual Fairlight step for {', '.join(manual)} "
            "(track.ensure sets only name and enabled; 21.1 has no Fairlight scripting)"
        )
    params = TrackParams(timeline=working, tracks=specs)
    summary = (
        f"ensure {sum(s.type == 'video' for s in specs)} video and "
        f"{sum(s.type == 'audio' for s in specs)} audio track(s)"
    )
    return _step("A", "track.ensure", "tracks", params, summary)


def _media_ref(plan: EditPlan, source: Source) -> MediaRef:
    bin_path = _import_bin(source) if _is_edit_path(source.media) else None
    return MediaRef(unique_id=source.media, bin_path=bin_path)


def _append_steps(
    plan: EditPlan, working: TimelineRef, batch_size: int, notes: list[str]
) -> tuple[list[Step], list[SegmentPlacement]]:
    ranges = plan.record_ranges()
    sources = plan.source_map()
    steps: list[Step] = []
    placements: list[SegmentPlacement] = []
    # Per track, in record order (segment list order is record order).
    for track in _video_tracks(plan):
        on_track: list[tuple[int | None, AppendRange]] = []
        for index, seg in enumerate(plan.segments):
            if seg.track != track:
                continue
            source = sources.get(seg.source)
            if source is None:
                raise CompileError(
                    "unknown_source",
                    f"segment {index} names source {seg.source!r}, which is not in the plan",
                    "add the source, or fix the segment's 'source' id",
                    segment=index,
                    source=seg.source,
                )
            record_in = ranges[index][0]
            rng = AppendRange(
                media=_media_ref(plan, source),
                src_in=seg.src_in,
                src_out=seg.src_out,
                track=track,
                media_type="av",
                record_in=record_in,
                label=f"segment {index}",
            )
            on_track.append((index, rng))
        if plan.outro is not None and plan.outro.track == track:
            on_track.append((None, _outro_range(plan, working)))
        _batch_track(track, working, on_track, batch_size, ranges, steps, placements)
    if not placements:
        notes.append("no video segments to append")
    return steps, placements


def _outro_range(plan: EditPlan, _working: TimelineRef) -> AppendRange:
    assert plan.outro is not None
    outro = plan.outro
    source = plan.source_map()[outro.source]
    end = outro.src_out if outro.src_out is not None else source.duration
    assert end is not None, "a validated plan knows the outro's length"
    return AppendRange(
        media=_media_ref(plan, source),
        src_in=outro.src_in,
        src_out=end,
        track=outro.track,
        media_type="av",
        record_in=plan.content_frames(),
        label="outro",
    )


def _batch_track(
    track: int,
    working: TimelineRef,
    on_track: list[tuple[int | None, AppendRange]],
    batch_size: int,
    ranges: list[tuple[int, int]],
    steps: list[Step],
    placements: list[SegmentPlacement],
) -> None:
    for batch_index, batch in enumerate(_chunk(on_track, batch_size)):
        name = f"append:v{track}:{batch_index}"
        rngs = [rng for _, rng in batch]
        params = AppendParams(timeline=working, ranges=rngs)
        first, last = rngs[0].record_in, rngs[-1].record_in
        summary = f"append {len(rngs)} range(s) on video track {track} (record {first}..{last})"
        steps.append(_step("A", "timeline.append_ranges", name, params, summary))
        for seg_index, rng in batch:
            if seg_index is None:
                continue
            placements.append(
                SegmentPlacement(
                    segment=seg_index,
                    track=track,
                    record_in=ranges[seg_index][0],
                    record_out=ranges[seg_index][1],
                    frames=rng.src_out - rng.src_in,
                    source=rng.media.unique_id,
                    append_step=name,
                )
            )


# -- phase B -----------------------------------------------------------------------------------


@dataclass
class _Resolver:
    """Maps ``(video track, record_in)`` to the appended items phase A verified."""

    by_key: dict[tuple[int, int], PlacedItem]
    audio_by_key: dict[tuple[int, int], list[PlacedItem]]
    ambiguous: set[tuple[int, int]]
    timeline: TimelineRef

    def video(self, track: int, record_in: int) -> PlacedItem:
        key = (track, record_in)
        if key in self.ambiguous:
            raise CompileError(
                "segment_double_matched",
                f"two appended ranges match video track {track} at record {record_in}",
                "phase A appended the same slot twice; re-run phase A cleanly",
                track=track,
                record_in=record_in,
            )
        item = self.by_key.get(key)
        if item is None:
            raise CompileError(
                "segment_unmatched",
                f"no appended item on video track {track} at record {record_in}",
                "give the phase-A result that appended this segment",
                track=track,
                record_in=record_in,
            )
        return item

    def audio(self, track: int, record_in: int) -> list[PlacedItem]:
        return self.audio_by_key.get((track, record_in), [])


def _resolver(phase_a_results: list[AppendResult]) -> _Resolver:
    by_key: dict[tuple[int, int], PlacedItem] = {}
    audio_by_key: dict[tuple[int, int], list[PlacedItem]] = {}
    ambiguous: set[tuple[int, int]] = set()
    timeline: TimelineRef | None = None
    for result in phase_a_results:
        timeline = TimelineRef(name=result.timeline.name, unique_id=result.timeline.unique_id)
        for appended in result.ranges:
            if appended.video is not None:
                key = (appended.video.track, appended.planned_record_in)
                if key in by_key:
                    ambiguous.add(key)
                by_key[key] = appended.video
                audio_by_key[key] = list(appended.audio)
    if timeline is None:
        raise CompileError(
            "missing_phase_a_result",
            "the phase-A result holds no append results, so no timeline is known",
            "give a phase-A result from a run of compile_build's steps",
        )
    return _Resolver(by_key, audio_by_key, ambiguous, timeline)


def compile_finish(
    plan: EditPlan,
    phase_a_results: list[AppendResult] | None = None,
    *,
    batch_size: int = DEFAULT_BATCH,
) -> FinishCompilation:
    """Compile phase B. With ``phase_a_results`` it emits steps addressing resolved item ids;
    without, it emits only the intents for a dry-run."""
    notes: list[str] = []
    resolver = _resolver(phase_a_results) if phase_a_results else None
    builder = _FinishBuilder(plan, resolver, notes)
    builder.run()
    steps = builder.to_steps(batch_size) if resolver is not None else []
    return FinishCompilation(builder.intents, steps, notes, resolver is not None)


@dataclass
class _Action:
    kind: str
    op: str
    target: str
    summary: str
    # The op target model, with its item ref filled once resolved (None until then).
    make_target: Any


class _FinishBuilder:
    """Walks the plan once, producing a :class:`FinishIntent` (and, when resolved, an op target)
    for every colour, transform, transition, title, gain, fade and marker."""

    def __init__(self, plan: EditPlan, resolver: _Resolver | None, notes: list[str]) -> None:
        self.plan = plan
        self.resolver = resolver
        self.notes = notes
        self.ranges = plan.record_ranges()
        self.intents: list[FinishIntent] = []
        self._targets: dict[str, list[Any]] = {
            "transition.add": [],
            "item.set_color": [],
            "item.set_transform": [],
            "item.set_audio_gain": [],
            "item.set_fades": [],
            "title.place": [],
            "markers.add": [],
        }

    # -- walk ----------------------------------------------------------------------------------

    def run(self) -> None:
        self._colours()
        self._transforms()
        self._transitions()
        self._head_and_end_fades()
        self._audio_gain()
        self._titles()
        self._markers()

    def _video_item(self, track: int, record_in: int) -> ItemRef | None:
        if self.resolver is None:
            return None
        placed = self.resolver.video(track, record_in)
        return ItemRef(track_type="video", track=placed.track, item_id=placed.item_id)

    def _audio_items(self, track: int, record_in: int) -> list[ItemRef]:
        if self.resolver is None:
            return []
        return [
            ItemRef(track_type="audio", track=p.track, item_id=p.item_id)
            for p in self.resolver.audio(track, record_in)
        ]

    def _emit(self, op: str, target: Any | None, intent: FinishIntent) -> None:
        self.intents.append(intent)
        if target is not None:
            self._targets[op].append(target)

    # -- colour --------------------------------------------------------------------------------

    def _colours(self) -> None:
        for index, seg in enumerate(self.plan.segments):
            source = self.plan.source_map().get(seg.source)
            if source is None or source.color is None:
                continue
            color = source.color
            if color.input_color_space is not None:
                self.notes.append(
                    f"source {source.id!r} input_color_space is not compiled: item.set_color has "
                    "no such field (21.1); set it by hand, never on BRAW"
                )
            if color.cdl is None and color.lut is None:
                continue
            record_in = self.ranges[index][0]
            item = self._video_item(seg.track, record_in)
            target = (
                ColorTarget(item=item, cdl=_cdl(color.cdl), lut=color.lut)
                if item is not None
                else None
            )
            self._emit(
                "item.set_color",
                target,
                FinishIntent(
                    kind="colour",
                    op="item.set_color",
                    target=_seg_target(index, seg.track, record_in),
                    summary=_colour_summary(color.cdl, color.lut),
                    resolved=item is not None,
                ),
            )

    # -- transform -----------------------------------------------------------------------------

    def _transforms(self) -> None:
        for index, seg in enumerate(self.plan.segments):
            if seg.transform is None:
                continue
            tf = seg.transform
            record_in = self.ranges[index][0]
            item = self._video_item(seg.track, record_in)
            target = (
                TransformTarget(
                    item=item,
                    zoom=tf.zoom,
                    pan=tf.pan,
                    tilt=tf.tilt,
                    rotation_deg=tf.rotation_deg,
                    flip_h=tf.flip_h,
                    flip_v=tf.flip_v,
                )
                if item is not None
                else None
            )
            self._emit(
                "item.set_transform",
                target,
                FinishIntent(
                    kind="transform",
                    op="item.set_transform",
                    target=_seg_target(index, seg.track, record_in),
                    summary=(
                        f"zoom {tf.zoom:g}, pan {tf.pan:g}, tilt {tf.tilt:g}, "
                        f"rot {tf.rotation_deg:g}, flip_h {tf.flip_h}, flip_v {tf.flip_v}"
                    ),
                    resolved=item is not None,
                ),
            )

    # -- transitions ---------------------------------------------------------------------------

    def _transitions(self) -> None:
        for tx in self.plan.transitions:
            self._content_transition(tx)
        self._outro_transition()

    def _content_transition(self, tx: Any) -> None:
        seg = self.plan.segments[tx.after]
        record_in = self.ranges[tx.after][0]
        if tx.video != "cut":
            name = _VIDEO_TRANSITIONS.get(tx.video)
            if name is None:
                self.notes.append(
                    f"segment {tx.after}: video transition {tx.video!r} has no 21.1 name; "
                    "left as a manual step"
                )
            else:
                item = self._video_item(seg.track, record_in)
                target = (
                    TransitionTarget(
                        item=item,
                        position="end",
                        type=name,
                        category="simple",
                        alignment="center",
                        frames=tx.video_frames,
                    )
                    if item is not None
                    else None
                )
                self._emit(
                    "transition.add",
                    target,
                    FinishIntent(
                        kind="transition",
                        op="transition.add",
                        target=f"after segment {tx.after} (video track {seg.track})",
                        summary=f"{name}, center, {tx.video_frames} frames",
                        resolved=item is not None,
                    ),
                )
        if tx.audio is not None and tx.audio.frames > 0:
            self._audio_crossfade(tx.after, seg.track, record_in, tx.audio)

    def _audio_crossfade(self, after: int, track: int, record_in: int, audio: Any) -> None:
        items = self._audio_items(track, record_in)
        made_target = items[0] if items else None
        target = (
            TransitionTarget(
                item=made_target,
                position="end",
                type=audio.type,
                category="audio",
                alignment="center",
                frames=audio.frames,
            )
            if made_target is not None
            else None
        )
        self._emit(
            "transition.add",
            target,
            FinishIntent(
                kind="audio_crossfade",
                op="transition.add",
                target=f"after segment {after} (audio under track {track})",
                summary=f"{audio.type}, center, {audio.frames} frames",
                resolved=made_target is not None,
            ),
        )

    def _outro_transition(self) -> None:
        outro = self.plan.outro
        if outro is None or outro.transition.video == "cut":
            return
        name = _VIDEO_TRANSITIONS.get(outro.transition.video)
        content_end = self.plan.content_frames()
        # The transition sits on the last content item that ends at the content boundary.
        last = _last_content_segment(self.plan)
        if name is None or last is None:
            if name is None:
                self.notes.append(
                    f"outro transition {outro.transition.video!r} has no 21.1 name; manual step"
                )
            return
        index, track, record_in = last
        item = self._video_item(track, record_in)
        target = (
            TransitionTarget(
                item=item,
                position="end",
                type=name,
                category="simple",
                alignment="left",
                frames=outro.transition.video_frames,
            )
            if item is not None
            else None
        )
        self._emit(
            "transition.add",
            target,
            FinishIntent(
                kind="outro_transition",
                op="transition.add",
                target=f"content into outro (segment {index}, record {content_end})",
                summary=f"{name}, left, {outro.transition.video_frames} frames",
                resolved=item is not None,
            ),
        )

    # -- head / end fades ----------------------------------------------------------------------

    def _head_and_end_fades(self) -> None:
        self._head_fade()
        self._end_fade()

    def _head_fade(self) -> None:
        head = self.plan.head
        first = _first_content_segment(self.plan)
        if head is None or first is None:
            return
        index, track, record_in = first
        video_frames = head.video_frames if head.video != "cut" else 0
        if video_frames > 0:
            item = self._video_item(track, record_in)
            target = FadeTarget(item=item, fade_in=video_frames) if item is not None else None
            self._emit(
                "item.set_fades",
                target,
                FinishIntent(
                    kind="head_fade",
                    op="item.set_fades",
                    target=f"first segment {index} (video track {track})",
                    summary=f"video fade-in {video_frames} frames (from black)",
                    resolved=item is not None,
                ),
            )
        if head.audio_fade_frames > 0:
            for audio in self._audio_items(track, record_in) or [None]:
                target = (
                    FadeTarget(item=audio, fade_in=head.audio_fade_frames)
                    if audio is not None
                    else None
                )
                self._emit(
                    "item.set_fades",
                    target,
                    FinishIntent(
                        kind="head_fade",
                        op="item.set_fades",
                        target=f"first segment {index} (audio under track {track})",
                        summary=f"audio fade-in {head.audio_fade_frames} frames",
                        resolved=audio is not None,
                    ),
                )

    def _end_fade(self) -> None:
        if self.plan.end_fade_frames is None or self.plan.end_fade_frames <= 0:
            return
        if self.plan.outro is not None:
            track, record_in = self.plan.outro.track, self.plan.content_frames()
            where = "outro"
        else:
            last = _last_content_segment(self.plan)
            if last is None:
                return
            index, track, record_in = last
            where = f"last segment {index}"
        for audio in self._audio_items(track, record_in) or [None]:
            target = (
                FadeTarget(item=audio, fade_out=self.plan.end_fade_frames)
                if audio is not None
                else None
            )
            self._emit(
                "item.set_fades",
                target,
                FinishIntent(
                    kind="end_fade",
                    op="item.set_fades",
                    target=f"{where} (audio under track {track})",
                    summary=f"audio fade-out {self.plan.end_fade_frames} frames (de-click)",
                    resolved=audio is not None,
                ),
            )

    # -- audio gain ----------------------------------------------------------------------------

    def _audio_gain(self) -> None:
        for index, seg in enumerate(self.plan.segments):
            if seg.gain_db == 0.0:
                continue
            record_in = self.ranges[index][0]
            for audio in self._audio_items(seg.track, record_in) or [None]:
                target = GainTarget(item=audio, gain_db=seg.gain_db) if audio is not None else None
                self._emit(
                    "item.set_audio_gain",
                    target,
                    FinishIntent(
                        kind="audio_gain",
                        op="item.set_audio_gain",
                        target=f"segment {index} (audio under track {seg.track})",
                        summary=f"{seg.gain_db:+g} dB",
                        resolved=audio is not None,
                    ),
                )

    # -- titles --------------------------------------------------------------------------------

    def _titles(self) -> None:
        track = _title_track(self.plan)
        for index, title in enumerate(self.plan.titles):
            self._one_title(
                f"title {index}",
                title.template,
                track,
                title.record_in,
                title.record_out,
                title.lines,
                kind="title",
            )
        outro = self.plan.outro
        if outro is not None and outro.sting is not None:
            start = self.plan.content_frames() + outro.sting.start
            self._one_title(
                "sting",
                outro.sting.template,
                track,
                start,
                start + outro.sting.frames,
                outro.sting.lines,
                kind="sting",
            )

    def _one_title(
        self,
        label: str,
        template: str | None,
        track: int,
        record_in: int,
        record_out: int,
        lines: list[str],
        *,
        kind: str,
    ) -> None:
        if template is None:
            self.notes.append(
                f"{label} has no template name; cannot place it (title.place needs one)"
            )
            return
        if len(lines) > 4:
            self.notes.append(f"{label} has {len(lines)} lines; title.place allows at most 4")
            return
        spec = TitleSpec(
            template=MediaRef(unique_id=template),
            track=track,
            record_in=record_in,
            record_out=record_out,
            lines=lines,
        )
        self._emit(
            "title.place",
            spec,
            FinishIntent(
                kind=kind,
                op="title.place",
                target=f"{label} on video track {track} (record {record_in}..{record_out})",
                summary=f"template {template!r}: {' / '.join(lines)}",
                resolved=True,
            ),
        )

    # -- markers -------------------------------------------------------------------------------

    def _markers(self) -> None:
        for index, marker in enumerate(self.plan.markers):
            color = marker.color
            if color is None:
                color = "Blue"
            elif color not in _MARKER_COLORS:
                self.notes.append(
                    f"marker {index} colour {color!r} is not a Resolve marker colour; using Blue"
                )
                color = "Blue"
            spec = MarkerSpec(
                frame=marker.record,
                color=color,
                name=marker.name,
                note=marker.note or "",
                duration=marker.duration,
            )
            self._emit(
                "markers.add",
                spec,
                FinishIntent(
                    kind="marker",
                    op="markers.add",
                    target=f"record {marker.record}",
                    summary=f"{color}{(' ' + marker.name) if marker.name else ''}",
                    resolved=True,
                ),
            )

    # -- steps ---------------------------------------------------------------------------------

    def to_steps(self, batch_size: int) -> list[Step]:
        assert self.resolver is not None
        timeline = self.resolver.timeline
        self._targets["item.set_fades"] = _merge_fades(self._targets["item.set_fades"])
        steps: list[Step] = []
        steps.extend(self._grouped("transition.add", TransitionParams, "targets", timeline, 50))
        steps.extend(self._grouped("item.set_color", ColorParams, "targets", timeline, 50))
        steps.extend(self._grouped("item.set_transform", TransformParams, "targets", timeline, 50))
        steps.extend(self._grouped("item.set_audio_gain", GainParams, "targets", timeline, 500))
        steps.extend(self._grouped("item.set_fades", FadesParams, "targets", timeline, 500))
        steps.extend(self._grouped("title.place", TitleParams, "titles", timeline, 50))
        steps.extend(self._grouped("markers.add", MarkersParams, "markers", timeline, 1000))
        return steps

    def _grouped(
        self, op: str, params_model: Any, field_name: str, timeline: TimelineRef, cap: int
    ) -> list[Step]:
        targets = self._targets[op]
        steps: list[Step] = []
        for batch_index, batch in enumerate(_chunk(targets, cap)):
            kwargs: dict[str, Any] = {field_name: batch, "timeline": timeline}
            params = params_model(**kwargs)
            name = f"{op}:{batch_index}"
            summary = f"{op} on {len(batch)} target(s)"
            steps.append(_step("B", op, name, params, summary))
        return steps


# -- helpers -----------------------------------------------------------------------------------


def _cdl(cdl: Cdl | None) -> CdlSpec | None:
    if cdl is None:
        return None
    return CdlSpec(
        slope=tuple(cdl.slope),
        offset=tuple(cdl.offset),
        power=tuple(cdl.power),
        saturation=cdl.saturation,
    )


def _colour_summary(cdl: Cdl | None, lut: str | None) -> str:
    parts = []
    if cdl is not None:
        parts.append(f"CDL slope {tuple(round(v, 4) for v in cdl.slope)} sat {cdl.saturation:g}")
    if lut is not None:
        parts.append(f"LUT {lut!r}")
    return ", ".join(parts)


def _seg_target(index: int, track: int, record_in: int) -> str:
    return f"segment {index} (video track {track} @{record_in})"


def _first_content_segment(plan: EditPlan) -> tuple[int, int, int] | None:
    """The earliest content item: least record_in, then lowest track."""
    ranges = plan.record_ranges()
    best: tuple[int, int, int] | None = None
    for index, seg in enumerate(plan.segments):
        key = (ranges[index][0], seg.track)
        if best is None or key < (best[2], best[1]):
            best = (index, seg.track, ranges[index][0])
    return best


def _last_content_segment(plan: EditPlan) -> tuple[int, int, int] | None:
    """The latest content item by record_out, then highest track."""
    ranges = plan.record_ranges()
    best: tuple[int, int, int] | None = None
    best_key: tuple[int, int] | None = None
    for index, seg in enumerate(plan.segments):
        key = (ranges[index][1], seg.track)
        if best_key is None or key > best_key:
            best_key = key
            best = (index, seg.track, ranges[index][0])
    return best


def _merge_fades(targets: list[FadeTarget]) -> list[FadeTarget]:
    """One ``FadeTarget`` per item: a clip at the very end of a one-item timeline can take both a
    head fade-in and an end fade-out, and ``item.set_fades`` refuses a repeated item."""
    merged: dict[str, FadeTarget] = {}
    for target in targets:
        prior = merged.get(target.item.item_id)
        if prior is None:
            merged[target.item.item_id] = target
            continue
        merged[target.item.item_id] = FadeTarget(
            item=target.item,
            fade_in=target.fade_in if target.fade_in is not None else prior.fade_in,
            fade_out=target.fade_out if target.fade_out is not None else prior.fade_out,
        )
    return list(merged.values())


def _chunk(items: list[Any], size: int) -> list[list[Any]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _step(phase: str, op: str, name: str, params: Any, summary: str) -> Step:
    return Step(
        phase=phase,  # type: ignore[arg-type]
        op=op,
        name=name,
        params=params.model_dump(mode="json"),
        summary=summary,
    )
