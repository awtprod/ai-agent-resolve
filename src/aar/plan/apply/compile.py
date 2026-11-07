"""Compile a linted EditPlan v2 into Resolve build-op steps, in three phases that each compile
only from the real results of the phases before them (round 2).

* **prepare** -- ``timeline.branch`` (from the template's real unique_id), ``media.import`` once
  per edit-PC file, ``media.find`` over the whole pool for every pool-name source and title
  template. Inputs: plan, template describe.
* **assemble** -- ``track.ensure`` + ``timeline.append_ranges`` against the *real* branch
  timeline and *real* pool unique_ids from the prepare result. Inputs: plan, prepare result.
* **finish** -- transitions, ``clip.set_input_color_space``, colour, transform, ``title.place``
  (real template pool ids), gain, fades, markers, each addressing an item resolved from the
  assemble result by ``(track, record_in)``. Inputs: plan, prepare result, assemble result.

A reference an earlier phase has not produced is a typed :mod:`~aar.plan.apply.refs` unresolved
marker, never a guessed value. A dry-run prints it; a submission is refused if any survives. Every
step has a stable name unique in its phase; a finished job's results are mapped by it when
:func:`~aar.plan.apply.jobs.phase_result_from_job` turns them into the next phase's input.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, get_args

from aar.node.ops import get_op
from aar.node.ops.build import (
    AudioFormat,  # noqa: F401  (import registers build.py's ops before we read them)
    path_key,
)
from aar.node.ops.framework import TimelineRef
from aar.node.ops.read import DescribeResult
from aar.node.ops.write import MarkerColor
from aar.plan.apply.errors import CompileError
from aar.plan.apply.inputs import (
    BRANCH_STEP,
    AssembleResult,
    PrepareResult,
    check_media_classifiable,
    classify_media,
    find_source_step,
    find_title_step,
    import_bin,
    import_owners,
    import_step,
    plan_hash,
    title_templates,
    validate_assemble,
    validate_prepare,
)
from aar.plan.apply.model import FinishIntent, Phase, SegmentPlacement, Step
from aar.plan.apply.refs import find_unresolved, unresolved
from aar.plan.model import Cdl, EditPlan, Source

DEFAULT_BATCH = 10
MAX_BATCH = 10
"""AppendToTimeline hung 21.1 on large batches (2026-11-05): at most 10 ranges per step."""

TEMPLATE_MAX_ITEMS = 30
"""DuplicateTimeline hung 21.1 on a 91-item timeline (2026-11-04)."""

MAX_STEPS = 500
"""``OpSequenceParams`` caps a job at 500 steps."""

_VIDEO_TRANSITIONS = {
    "dissolve": "Cross Dissolve",
    "dip": "Dip To Color Dissolve",
    "wipe": "Wipe",
    "push": "Push",
}
_MARKER_COLORS = frozenset(get_args(MarkerColor))


@dataclass
class Compilation:
    """One phase's compiled steps, with the dry-run extras and any gaps found."""

    phase: Phase
    steps: list[Step]
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    placements: list[SegmentPlacement] = field(default_factory=list)
    intents: list[FinishIntent] = field(default_factory=list)
    batch_size: int = DEFAULT_BATCH

    @property
    def unresolved(self) -> int:
        return sum(len(find_unresolved(step.params)) for step in self.steps)


def template_item_count(describe: DescribeResult) -> int:
    return sum(track.item_count for track in describe.tracks)


# -- shared plan geometry ----------------------------------------------------------------------


def video_tracks(plan: EditPlan) -> list[int]:
    tracks = {seg.track for seg in plan.segments}
    if plan.outro is not None:
        tracks.add(plan.outro.track)
    return sorted(tracks)


def title_track(plan: EditPlan) -> int:
    """The one overlay track for titles and the sting: above the top content track. The single
    source of truth, shared with the renderer (F11)."""
    return max(video_tracks(plan) or [0]) + 1


def _check_title_track(plan: EditPlan) -> None:
    if (plan.titles or _has_sting(plan)) and title_track(plan) > 16:
        raise CompileError(
            "title_track_out_of_range",
            f"titles would land on video track {title_track(plan)}, over the 16-track limit",
            "reduce the number of content video tracks",
            track=title_track(plan),
        )


def _audio_index(plan: EditPlan, name: str | None) -> int | None:
    for index, track in enumerate(plan.audio_tracks, start=1):
        if track.name == name:
            return index
    return None


# -- phase 1: prepare --------------------------------------------------------------------------


def compile_prepare(plan: EditPlan, *, template: DescribeResult | None = None) -> Compilation:
    if plan.template is None:
        raise CompileError(
            "missing_template",
            "the plan has no template to build from",
            "set the plan's 'template' (the timeline timeline.branch starts from)",
        )
    check_media_classifiable(plan)
    notes: list[str] = []
    steps: list[Step] = []

    if template is not None:
        if template.timeline.name != plan.template:
            raise CompileError(
                "template_name_mismatch",
                f"the template describe is of {template.timeline.name!r}, not the plan's "
                f"template {plan.template!r}",
                "describe the timeline the plan names as its template",
                described=template.timeline.name,
                expected=plan.template,
            )
        items = template_item_count(template)
        if items > TEMPLATE_MAX_ITEMS:
            raise CompileError(
                "template_too_large",
                f"the template has {items} items, over the {TEMPLATE_MAX_ITEMS} limit",
                "DuplicateTimeline hung 21.1 on a 91-item timeline; branch a small template",
                items=items,
                limit=TEMPLATE_MAX_ITEMS,
            )
        source: Any = {"name": template.timeline.name, "unique_id": template.timeline.unique_id}
        notes.append(f"template {plan.template!r} has {items} item(s) (<= {TEMPLATE_MAX_ITEMS})")
    else:
        source = unresolved(
            "timeline",
            f"template {plan.template!r} unique_id; pass --template-describe (also size-checks it)",
        )
        notes.append("template size unchecked (no --template-describe)")

    steps.append(
        _step(
            "prepare",
            "timeline.branch",
            BRANCH_STEP,
            {"source": source, "if_exists": "reuse"},
            f"branch {plan.template!r} (reuse)",
        )
    )
    steps.extend(_import_steps(plan, notes))
    steps.extend(_find_source_steps(plan))
    for name in title_templates(plan):
        steps.append(
            _step(
                "prepare",
                "media.find",
                find_title_step(name),
                # Recursive over the whole pool; the op matches the name exactly (case-sensitive)
                # and the job builder re-checks the returned name (N1/N5).
                {"name": name, "bin_path": [], "recursive": True, "limit": 200},
                f"find title template {name!r}",
            )
        )
    if not title_templates(plan) and (plan.titles or _has_sting(plan)):
        notes.append(
            "a title/sting has no template name; it cannot be placed (title.place needs one)"
        )
    _check_steps(steps)
    return Compilation("prepare", steps, notes)


def _find_source_steps(plan: EditPlan) -> list[Step]:
    """A media.find for each pool-name source (e.g. a CCN multicam clip), recursively over the
    WHOLE media pool, keyed to the source id so the result maps by name, never by position (N1,
    N5). It requires exactly one exact, case-sensitive name match; the editor resolves ambiguity
    in Resolve."""
    steps: list[Step] = []
    for source in plan.sources:
        if classify_media(source.media) != "pool":
            continue
        steps.append(
            _step(
                "prepare",
                "media.find",
                find_source_step(source.id),
                {"name": source.media, "bin_path": [], "recursive": True, "limit": 200},
                f"find pool source {source.media!r} for {source.id}",
            )
        )
    return steps


def _has_sting(plan: EditPlan) -> bool:
    return plan.outro is not None and plan.outro.sting is not None


def _import_steps(plan: EditPlan, notes: list[str]) -> list[Step]:
    """``media.import`` batches (<= 50 paths, the op's cap) per camera bin. Each distinct file (by
    ``path_key``) is imported once, into the bin of the first source that uses it; a source that
    shares the file maps to the same pool item (N7)."""
    owners = import_owners(plan)
    bins: dict[tuple[str, ...], list[str]] = {}
    for owner in owners.values():
        bins.setdefault(import_bin(owner), []).append(owner.media)
    pool = [s.id for s in plan.sources if classify_media(s.media) != "import"]
    if pool:
        notes.append(f"pool-name sources resolved by media.find in prepare: {', '.join(pool)}")
    shared = [
        f"{s.id} (= {owners[path_key(s.media)].id})"
        for s in plan.sources
        if classify_media(s.media) == "import" and owners[path_key(s.media)] is not s
    ]
    if shared:
        notes.append(f"sources sharing a file are imported once: {', '.join(shared)}")
    steps: list[Step] = []
    for bin_path, paths in bins.items():
        for batch_index, batch in enumerate(_chunk(paths, 50)):
            steps.append(
                _step(
                    "prepare",
                    "media.import",
                    import_step(bin_path, batch_index),
                    {
                        "paths": batch,
                        "bin_path": list(bin_path),
                        "create_bins": True,
                        "if_present": "reuse",
                    },
                    f"import {len(batch)} file(s) into {'/'.join(bin_path)}",
                )
            )
    return steps


# -- phase 2: assemble -------------------------------------------------------------------------


def compile_assemble(
    plan: EditPlan, *, prepare: PrepareResult | None = None, batch_size: int = DEFAULT_BATCH
) -> Compilation:
    _check_batch(batch_size)
    _check_title_track(plan)
    check_media_classifiable(plan)
    if prepare is not None:
        validate_prepare(prepare, plan)
    notes: list[str] = []
    warnings: list[str] = []
    working = prepare.branch_ref() if prepare is not None else None
    tl = _tl(working, "the branch; run phase 'prepare' and pass --prepare-result")
    source_media = prepare.source_media() if prepare is not None else {}

    steps: list[Step] = [_track_step(plan, tl, warnings)]
    append_steps, placements = _append_steps(plan, tl, source_media, batch_size, notes)
    steps.extend(append_steps)
    _check_steps(steps)
    return Compilation(
        "assemble", steps, notes, warnings, placements=placements, batch_size=batch_size
    )


def _track_step(plan: EditPlan, tl: Any, warnings: list[str]) -> Step:
    specs: list[dict[str, Any]] = []
    for index in range(1, (max(video_tracks(plan), default=0) or 1) + 1):
        specs.append({"type": "video", "index": index})
    if plan.titles or _has_sting(plan):
        specs.append({"type": "video", "index": title_track(plan)})
    for index, track in enumerate(plan.audio_tracks, start=1):
        # Never apply mute: it is template state in 21.1, and disabling a bus-fed track would
        # render the show silent (F5). track.ensure only creates and names the track here.
        specs.append({"type": "audio", "index": index, "name": track.name})
        if track.mute:
            warnings.append(
                f"audio track {track.name!r} is marked mute in the plan; mute is template state "
                "in 21.1, so it is not applied -- verify it in the template"
            )
    summary = (
        f"ensure {sum(s['type'] == 'video' for s in specs)} video and "
        f"{sum(s['type'] == 'audio' for s in specs)} audio track(s)"
    )
    return _step("assemble", "track.ensure", "tracks", {"timeline": tl, "tracks": specs}, summary)


def _media_ref(plan: EditPlan, source: Source, source_media: dict[str, Any]) -> Any:
    """The pool media ref for a source: ONLY from the prepare result. A source never passes its
    own ``media`` string through as a pool id (R1); without a prepare result it is Unresolved,
    which makes submit refuse."""
    imported = source_media.get(source.id)
    if imported is not None:
        return imported.media.model_dump(mode="json")
    return unresolved(
        "media",
        f"pool unique_id for source {source.id!r} ({source.media}); run phase 'prepare' and pass "
        "--prepare-result",
    )


def _append_steps(
    plan: EditPlan, tl: Any, source_media: dict[str, Any], batch_size: int, notes: list[str]
) -> tuple[list[Step], list[SegmentPlacement]]:
    ranges = plan.record_ranges()
    sources = plan.source_map()
    steps: list[Step] = []
    placements: list[SegmentPlacement] = []
    for track in video_tracks(plan):
        on_track: list[tuple[int | None, dict[str, Any]]] = []
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
            if seg.length <= 0:
                raise CompileError(
                    "zero_length_segment",
                    f"segment {index} has a non-positive length ({seg.src_in}..{seg.src_out})",
                    "give src_out greater than src_in",
                    segment=index,
                )
            on_track.append(
                (
                    index,
                    _range(
                        _media_ref(plan, source, source_media),
                        seg.src_in,
                        seg.src_out,
                        track,
                        "av",
                        ranges[index][0],
                        f"segment {index}",
                    ),
                )
            )
        _batch_track(track, tl, on_track, batch_size, ranges, steps, placements)
    _outro_append(plan, tl, source_media, steps, notes)
    if not placements:
        notes.append("no video segments to append")
    return steps, placements


def _outro_append(
    plan: EditPlan, tl: Any, source_media: dict[str, Any], steps: list[Step], notes: list[str]
) -> None:
    outro = plan.outro
    if outro is None:
        return
    source = plan.source_map()[outro.source]
    end = outro.src_out if outro.src_out is not None else source.duration
    assert end is not None, "a validated plan knows the outro's length"
    start = plan.content_frames()
    media = _media_ref(plan, source, source_media)
    # The outro's picture on its video track, and its sound on the named audio track (the A5 bus
    # in CCN), never on A1 (F4).
    steps.append(
        _step(
            "assemble",
            "timeline.append_ranges",
            f"append:outro-v{outro.track}",
            {
                "timeline": tl,
                "ranges": [_range(media, outro.src_in, end, outro.track, "video", start, "outro")],
            },
            f"append outro video on track {outro.track} (record {start})",
        )
    )
    audio_index = _audio_index(plan, outro.audio_track)
    if audio_index is None:
        # Refuse rather than silently drop the outro's sound onto A1 (R4): the outro clip has
        # audio, so it needs a named audio track to carry it.
        raise CompileError(
            "outro_audio_track_missing",
            f"the outro has no audio_track in audio_tracks (audio_track={outro.audio_track!r})",
            "set outro.audio_track to an audio_tracks entry (e.g. the A5 bus) so the outro's "
            "sound has a track; it must not fall back to A1",
            audio_track=outro.audio_track,
        )
    steps.append(
        _step(
            "assemble",
            "timeline.append_ranges",
            f"append:outro-a{audio_index}",
            {
                "timeline": tl,
                "ranges": [
                    _range(media, outro.src_in, end, audio_index, "audio", start, "outro-audio")
                ],
            },
            f"append outro audio on audio track {audio_index} ({outro.audio_track})",
        )
    )


def _range(
    media: Any, src_in: int, src_out: int, track: int, media_type: str, record_in: int, label: str
) -> dict[str, Any]:
    return {
        "media": media,
        "src_in": src_in,
        "src_out": src_out,
        "track": track,
        "media_type": media_type,
        "record_in": record_in,
        "label": label,
    }


def _batch_track(
    track: int,
    tl: Any,
    on_track: list[tuple[int | None, dict[str, Any]]],
    batch_size: int,
    ranges: list[tuple[int, int]],
    steps: list[Step],
    placements: list[SegmentPlacement],
) -> None:
    for batch_index, batch in enumerate(_chunk(on_track, batch_size)):
        rngs = [rng for _, rng in batch]
        name = f"append:v{track}:{batch_index}"
        first, last = rngs[0]["record_in"], rngs[-1]["record_in"]
        steps.append(
            _step(
                "assemble",
                "timeline.append_ranges",
                name,
                {"timeline": tl, "ranges": rngs},
                f"append {len(rngs)} range(s) on video track {track} (record {first}..{last})",
            )
        )
        for seg_index, rng in batch:
            if seg_index is None:
                continue
            placements.append(
                SegmentPlacement(
                    segment=seg_index,
                    track=track,
                    record_in=ranges[seg_index][0],
                    record_out=ranges[seg_index][1],
                    frames=rng["src_out"] - rng["src_in"],
                    source=rng["label"],
                    append_step=name,
                )
            )


# -- phase 3: finish ---------------------------------------------------------------------------


@dataclass
class _Resolver:
    video: dict[tuple[int, int], str]
    content_audio: dict[tuple[int, int], list[str]]
    audio_only: dict[tuple[int, int], list[str]]
    meta: dict[tuple[int, int], tuple[int | None, int | None]]
    ambiguous: set[tuple[int, int]]
    timeline: TimelineRef

    def video_item(self, track: int, record_in: int) -> dict[str, Any]:
        key = (track, record_in)
        if key in self.ambiguous:
            raise CompileError(
                "segment_double_matched",
                f"two appended ranges match video track {track} at record {record_in}",
                "phase 2 appended the same slot twice; re-run it cleanly",
                track=track,
                record_in=record_in,
            )
        item = self.video.get(key)
        if item is None:
            raise CompileError(
                "segment_unmatched",
                f"no appended item on video track {track} at record {record_in}",
                "give the phase-2 (assemble) result that appended this segment",
                track=track,
                record_in=record_in,
            )
        return {"track_type": "video", "track": track, "item_id": item}

    def content_audio_items(self, track: int, record_in: int) -> list[dict[str, Any]]:
        return [
            {"track_type": "audio", "track": track, "item_id": i}
            for i in self.content_audio.get((track, record_in), [])
        ]

    def audio_only_items(self, track: int, record_in: int) -> list[dict[str, Any]]:
        return [
            {"track_type": "audio", "track": track, "item_id": i}
            for i in self.audio_only.get((track, record_in), [])
        ]


def _resolver(plan: EditPlan, assemble: AssembleResult) -> _Resolver:
    video: dict[tuple[int, int], str] = {}
    content_audio: dict[tuple[int, int], list[str]] = {}
    audio_only: dict[tuple[int, int], list[str]] = {}
    meta: dict[tuple[int, int], tuple[int | None, int | None]] = {}
    ambiguous: set[tuple[int, int]] = set()
    timeline = assemble.branch_ref()
    # Every append is on the branch: validate_assemble has refused a mixed result before this
    # runs (one copy of that check, so it cannot drift).
    for result in assemble.appends:
        for appended in result.ranges:
            if appended.video is not None:
                key = (appended.video.track, appended.planned_record_in)
                if key in video:
                    ambiguous.add(key)
                video[key] = appended.video.item_id
                content_audio[key] = [p.item_id for p in appended.audio]
                meta[key] = (appended.video.source_start, appended.video.source_end)
            elif appended.audio:
                key = (appended.audio[0].track, appended.planned_record_in)
                audio_only.setdefault(key, []).extend(p.item_id for p in appended.audio)
    return _Resolver(video, content_audio, audio_only, meta, ambiguous, timeline)


def compile_finish(
    plan: EditPlan,
    *,
    prepare: PrepareResult | None = None,
    assemble: AssembleResult | None = None,
    batch_size: int = DEFAULT_BATCH,
) -> Compilation:
    _check_batch(batch_size)
    _check_title_track(plan)
    check_media_classifiable(plan)
    if assemble is not None and prepare is None:
        # Finish needs the prepare envelope too: title pool ids and the BRAW re-check come from
        # it, and we must cross-check the shared branch (R2).
        raise CompileError(
            "missing_prepare_result",
            "finish needs the prepare result as well as the assemble result",
            "pass --prepare-result alongside --assemble-result",
        )
    if prepare is not None:
        validate_prepare(prepare, plan)
    if assemble is not None:
        validate_assemble(assemble, plan)
    if (
        prepare is not None
        and assemble is not None
        and prepare.branch.unique_id != assemble.branch.unique_id
    ):
        raise CompileError(
            "phase_result_mismatch",
            "the prepare and assemble results are from different branch timelines",
            "pass the prepare and assemble results of the same build",
        )
    resolver = _resolver(plan, assemble) if assemble is not None else None
    if resolver is not None:
        _check_all_segments_matched(plan, resolver)
    builder = _Finish(plan, prepare, resolver)
    builder.run()
    steps = builder.to_steps(plan)
    _check_steps(steps)
    return Compilation("finish", steps, builder.notes, builder.warnings, intents=builder.intents)


SOURCE_READBACK_TOLERANCE = 1
"""Trap ``source-frame-readback-rounding`` (confirmed, aar/node/traps.py): GetSourceStartFrame /
GetSourceEndFrame read back up to one frame off the in point used, and the append op itself
accepts a start one frame off (write.py, note ``source_readback_off_by_one``). A tighter check
would refuse honest results; a stale range differs by far more."""


def _check_all_segments_matched(plan: EditPlan, resolver: _Resolver) -> None:
    """Every content segment must be matched exactly once in the assemble result, finishing op or
    not; and the matched append must be of this plan's range, not a stale one (F8, R2).

    ``record_in`` is matched exactly (it is the resolver key, a timeline frame). The source in/out
    are matched to the read-back tolerance above. ``record_out`` is NOT compared: the op
    rate-converts it (write.py), so a mixed-rate source would read falsely stale (N8). A source
    range that was not read back fails closed as ``unverified_phase_result`` (N6)."""
    ranges = plan.record_ranges()
    for index, seg in enumerate(plan.segments):
        record_in = ranges[index][0]
        resolver.video_item(seg.track, record_in)  # raises unmatched / double_matched
        src_start, src_end = resolver.meta[(seg.track, record_in)]
        if src_start is None or src_end is None:
            raise CompileError(
                "unverified_phase_result",
                f"segment {index}: the appended item's source range was not read back, so it "
                "cannot be checked against the plan",
                "re-run phase 'assemble' against a Resolve that reports source frames",
                segment=index,
            )
        if (
            abs(src_start - seg.src_in) > SOURCE_READBACK_TOLERANCE
            or abs(src_end - seg.src_out) > SOURCE_READBACK_TOLERANCE
        ):
            raise CompileError(
                "stale_phase_result",
                f"segment {index} appended source {src_start}..{src_end}, not the plan's "
                f"{seg.src_in}..{seg.src_out}",
                "the assemble result is from an older plan; re-run phase 'assemble'",
                segment=index,
            )


class _Finish:
    def __init__(self, plan: EditPlan, prepare: PrepareResult | None, resolver: _Resolver | None):
        self.plan = plan
        self.prepare = prepare
        self.resolver = resolver
        self.titles = prepare.title_media() if prepare is not None else {}
        self.imports = prepare.source_media() if prepare is not None else {}
        self.ranges = plan.record_ranges()
        self.notes: list[str] = []
        self.warnings: list[str] = []
        self.intents: list[FinishIntent] = []
        self._t: dict[str, list[dict[str, Any]]] = {
            op: []
            for op in (
                "transition.add",
                "clip.set_input_color_space",
                "item.set_color",
                "item.set_transform",
                "item.set_audio_gain",
                "item.set_fades",
                "title.place",
                "markers.add",
            )
        }

    # -- resolution helpers --------------------------------------------------------------------

    def _video(self, track: int, record_in: int, what: str) -> Any:
        if self.resolver is None:
            return unresolved("item", f"{what}; run phase 'assemble' and pass --assemble-result")
        return self.resolver.video_item(track, record_in)

    def _content_audio(self, track: int, record_in: int) -> list[Any] | None:
        if self.resolver is None:
            return None
        return self.resolver.content_audio_items(track, record_in)

    def _resolved(self, ref: Any) -> bool:
        return not find_unresolved(ref)

    def _emit(self, op: str, target: Any, intent: FinishIntent) -> None:
        self.intents.append(intent)
        self._t[op].append(target)

    # -- walk ----------------------------------------------------------------------------------

    def run(self) -> None:
        self._colour_spaces()
        self._colours()
        self._transforms()
        self._transitions()
        self._head_and_end_fades()
        self._audio_gain()
        self._titles()
        self._markers()
        self._captions()

    # -- F6: input colour space ----------------------------------------------------------------

    def _colour_spaces(self) -> None:
        by_space: dict[str, list[tuple[str, Any]]] = {}
        for source in self.plan.sources:
            if source.color is None or source.color.input_color_space is None:
                continue
            space = source.color.input_color_space
            imported = self.imports.get(source.id)
            if imported is not None:
                if _is_braw(imported.file_path):
                    self.notes.append(
                        f"source {source.id!r} is BRAW ({imported.file_path}); input colour space "
                        "not set (house safety rule)"
                    )
                    continue
                item: Any = imported.media.model_dump(mode="json")
            elif _is_braw(source.media):
                self.notes.append(
                    f"source {source.id!r} looks BRAW ({source.media}); input colour space not set"
                )
                continue
            else:
                item = unresolved(
                    "media", f"pool item for source {source.id!r}; run phase 'prepare'"
                )
            by_space.setdefault(space, []).append((source.id, item))
        for space, items in by_space.items():
            resolved = all(self._resolved(i) for _, i in items)
            self._emit(
                "clip.set_input_color_space",
                {"items": [i for _, i in items], "color_space": space},
                FinishIntent(
                    kind="input_colour_space",
                    op="clip.set_input_color_space",
                    target=f"sources {', '.join(s for s, _ in items)}",
                    summary=f"input colour space {space!r} (non-BRAW only)",
                    resolved=resolved,
                ),
            )

    # -- colour --------------------------------------------------------------------------------

    def _colours(self) -> None:
        for index, seg in enumerate(self.plan.segments):
            source = self.plan.source_map().get(seg.source)
            if source is None or source.color is None:
                continue
            color = source.color
            if color.cdl is None and color.lut is None:
                continue
            record_in = self.ranges[index][0]
            item = self._video(seg.track, record_in, f"segment {index} colour")
            target: Any = {"item": item}
            if color.cdl is not None:
                target["cdl"] = _cdl_dict(color.cdl)
            if color.lut is not None:
                target["lut"] = color.lut
            self._emit(
                "item.set_color",
                target,
                FinishIntent(
                    kind="colour",
                    op="item.set_color",
                    target=_seg_target(index, seg.track, record_in),
                    summary=_colour_summary(color.cdl, color.lut),
                    resolved=self._resolved(item),
                ),
            )

    # -- transform -----------------------------------------------------------------------------

    def _transforms(self) -> None:
        for index, seg in enumerate(self.plan.segments):
            if seg.transform is None:
                continue
            tf = seg.transform
            record_in = self.ranges[index][0]
            item = self._video(seg.track, record_in, f"segment {index} transform")
            self._emit(
                "item.set_transform",
                {
                    "item": item,
                    "zoom": tf.zoom,
                    "pan": tf.pan,
                    "tilt": tf.tilt,
                    "rotation_deg": tf.rotation_deg,
                    "flip_h": tf.flip_h,
                    "flip_v": tf.flip_v,
                },
                FinishIntent(
                    kind="transform",
                    op="item.set_transform",
                    target=_seg_target(index, seg.track, record_in),
                    summary=f"zoom {tf.zoom:g}, pan {tf.pan:g}, tilt {tf.tilt:g}, "
                    f"rot {tf.rotation_deg:g}, flip_h {tf.flip_h}, flip_v {tf.flip_v}",
                    resolved=self._resolved(item),
                ),
            )

    # -- transitions ---------------------------------------------------------------------------

    def _transitions(self) -> None:
        self._head_transition()
        for tx in self.plan.transitions:
            self._content_transition(tx)
        self._outro_transition()

    def _head_transition(self) -> None:
        head = self.plan.head
        first = _first_content_segment(self.plan)
        if head is None or first is None or head.video == "cut":
            return
        name = _VIDEO_TRANSITIONS.get(head.video)
        if name is None:
            self.notes.append(f"head transition {head.video!r} has no 21.1 name; manual step")
            return
        index, track, record_in = first
        # Head from black: transition.add on the first item's start, right-aligned, so the
        # missing outgoing (black) needs no handle (API fact build.py _handles: right=(f,0)).
        item = self._video(track, record_in, f"head on segment {index}")
        self._emit(
            "transition.add",
            {
                "item": item,
                "position": "start",
                "type": name,
                "category": "simple",
                "alignment": "right",
                "frames": head.video_frames,
            },
            FinishIntent(
                kind="head_transition",
                op="transition.add",
                target=f"head from black on segment {index} (video track {track})",
                summary=f"{name}, right (from black), {head.video_frames} frames",
                resolved=self._resolved(item),
            ),
        )

    def _content_transition(self, tx: Any) -> None:
        seg = self.plan.segments[tx.after]
        record_in = self.ranges[tx.after][0]
        if tx.video != "cut":
            name = _VIDEO_TRANSITIONS.get(tx.video)
            if name is None:
                self.notes.append(f"segment {tx.after}: transition {tx.video!r} has no 21.1 name")
            else:
                align = self._alignment(tx.after, tx.video_frames, f"segment {tx.after} dissolve")
                item = self._video(seg.track, record_in, f"segment {tx.after} dissolve")
                self._emit(
                    "transition.add",
                    {
                        "item": item,
                        "position": "end",
                        "type": name,
                        "category": "simple",
                        "alignment": align,
                        "frames": tx.video_frames,
                    },
                    FinishIntent(
                        kind="transition",
                        op="transition.add",
                        target=f"after segment {tx.after} (video track {seg.track})",
                        summary=f"{name}, {align}, {tx.video_frames} frames",
                        resolved=self._resolved(item),
                    ),
                )
        if tx.audio is not None and tx.audio.frames > 0:
            self._audio_crossfade(tx.after, seg.track, record_in, tx.audio)

    def _audio_crossfade(self, after: int, track: int, record_in: int, audio: Any) -> None:
        align = self._alignment(after, audio.frames, f"segment {after} crossfade")
        items = self._content_audio(track, record_in)
        refs = (
            items
            if items is not None
            else [unresolved("item", f"audio of segment {after}; run phase 'assemble'")]
        )
        if items is not None and not items:
            self.notes.append(f"segment {after}: no audio item to crossfade")
            return
        for position, ref in enumerate(refs):
            self._emit(
                "transition.add",
                {
                    "item": ref,
                    "position": "end",
                    "type": audio.type,
                    "category": "audio",
                    "alignment": align,
                    "frames": audio.frames,
                },
                FinishIntent(
                    kind="audio_crossfade",
                    op="transition.add",
                    target=f"after segment {after} (audio item {position})",
                    summary=f"{audio.type}, {align}, {audio.frames} frames",
                    resolved=self._resolved(ref),
                ),
            )

    def _outro_transition(self) -> None:
        outro = self.plan.outro
        if outro is None or outro.transition.video == "cut":
            return
        name = _VIDEO_TRANSITIONS.get(outro.transition.video)
        last = _last_content_segment(self.plan)
        if name is None or last is None:
            if name is None:
                self.notes.append(f"outro transition {outro.transition.video!r} has no 21.1 name")
            return
        index, track, record_in = last
        align = self._alignment(index, outro.transition.video_frames, "outro transition")
        item = self._video(track, record_in, "outro transition")
        self._emit(
            "transition.add",
            {
                "item": item,
                "position": "end",
                "type": name,
                "category": "simple",
                "alignment": align,
                "frames": outro.transition.video_frames,
            },
            FinishIntent(
                kind="outro_transition",
                op="transition.add",
                target=f"content into outro (segment {index})",
                summary=f"{name}, {align}, {outro.transition.video_frames} frames",
                resolved=self._resolved(item),
            ),
        )

    def _alignment(self, outgoing_index: int, frames: int, what: str) -> str:
        """Choose an alignment from the handles the plan's source ranges actually have, and
        refuse when none fits (F7). Outgoing tail = source.duration - src_out; incoming head =
        the next same-track segment's src_in (0 for the outro, which starts at source frame 0)."""
        sources = self.plan.source_map()
        out_seg = self.plan.segments[outgoing_index]
        out_src = sources.get(out_seg.source)
        tail = (
            None
            if out_src is None or out_src.duration is None
            else out_src.duration - out_seg.src_out
        )
        head = self._incoming_head(outgoing_index)
        half = frames // 2
        if _fits(tail, half) and _fits(head, half):
            return "center"
        if _fits(tail, frames):
            return "right"
        if _fits(head, frames):
            return "left"
        raise CompileError(
            "no_handle_for_transition",
            f"{what}: no alignment fits a {frames}-frame transition (outgoing tail {tail}, "
            f"incoming head {head})",
            "shorten the transition, give the source ranges more handle, or set the source "
            "'duration' (an unknown duration counts as no handle)",
            outgoing_tail=tail,
            incoming_head=head,
            frames=frames,
        )

    def _incoming_head(self, outgoing_index: int) -> int | None:
        track = self.plan.segments[outgoing_index].track
        for index in range(outgoing_index + 1, len(self.plan.segments)):
            if self.plan.segments[index].track == track:
                return self.plan.segments[index].src_in
        if self.plan.outro is not None and self.plan.outro.track == track:
            return self.plan.outro.src_in
        return None

    # -- head / end fades ----------------------------------------------------------------------

    def _head_and_end_fades(self) -> None:
        head = self.plan.head
        first = _first_content_segment(self.plan)
        if head is not None and first is not None and head.audio_fade_frames > 0:
            index, track, record_in = first
            # [I] inferred: the facts doc does not say an audio AddTransition needs a neighbour,
            # so we do NOT claim 21.1 lacks an audio-from-silence transition. We use the clip fade
            # handle (item.set_fades, which the op reads back) for the head audio fade-in.
            self.notes.append(
                "[I] head audio fade uses the clip fade handle (item.set_fades), not an audio "
                "transition from silence (inferred: the facts doc doesn't cover this)"
            )
            self._fades(
                track, record_in, fade_in=head.audio_fade_frames, target=f"first segment {index}"
            )
        if self.plan.end_fade_frames and self.plan.end_fade_frames > 0:
            track, record_in, where = self._end_item()
            if track is None:
                raise CompileError(
                    "end_fade_without_audio",
                    f"end_fade_frames is set ({self.plan.end_fade_frames}) but there is no audio "
                    "track to fade at the end of the timeline",
                    "give the outro an audio_track, or add a content segment with audio, or drop "
                    "end_fade_frames",
                )
            self._fades(
                track,
                record_in,
                fade_out=self.plan.end_fade_frames,
                target=where,
                audio_only=self.plan.outro is not None,
            )

    def _end_item(self) -> tuple[int | None, int, str]:
        if self.plan.outro is not None:
            index = _audio_index(self.plan, self.plan.outro.audio_track)
            if index is None:
                return None, 0, "outro"
            return index, self.plan.content_frames(), "outro audio"
        last = _last_content_segment(self.plan)
        if last is None:
            return None, 0, ""
        idx, track, record_in = last
        return track, record_in, f"last segment {idx}"

    def _fades(
        self,
        track: int,
        record_in: int,
        *,
        fade_in: int | None = None,
        fade_out: int | None = None,
        target: str,
        audio_only: bool = False,
    ) -> None:
        if self.resolver is None:
            items: list[Any] = [unresolved("item", f"audio of {target}; run phase 'assemble'")]
        elif audio_only:
            items = self.resolver.audio_only_items(track, record_in)
        else:
            items = self.resolver.content_audio_items(track, record_in)
        if self.resolver is not None and not items:
            self.notes.append(f"{target}: no audio item to fade")
            return
        for position, ref in enumerate(items):
            fade: dict[str, Any] = {"item": ref}
            if fade_in is not None:
                fade["fade_in"] = fade_in
            if fade_out is not None:
                fade["fade_out"] = fade_out
            self._emit(
                "item.set_fades",
                fade,
                FinishIntent(
                    kind="fade",
                    op="item.set_fades",
                    target=f"{target} (audio item {position})",
                    summary=f"fade_in {fade_in} fade_out {fade_out}",
                    resolved=self._resolved(ref),
                ),
            )

    # -- audio gain ----------------------------------------------------------------------------

    def _audio_gain(self) -> None:
        for index, seg in enumerate(self.plan.segments):
            if seg.gain_db == 0.0:
                continue
            if not -100.0 <= seg.gain_db <= 30.0:
                raise CompileError(
                    "gain_out_of_range",
                    f"segment {index} gain {seg.gain_db:+g} dB is outside -100..+30",
                    "item.set_audio_gain clamps clip gain to [-100, 30] dB",
                    segment=index,
                    gain_db=seg.gain_db,
                )
            record_in = self.ranges[index][0]
            items = self._content_audio(seg.track, record_in)
            refs = (
                items
                if items is not None
                else [unresolved("item", f"audio of segment {index}; run phase 'assemble'")]
            )
            if items is not None and not items:
                self.notes.append(f"segment {index}: no audio item for gain")
                continue
            for position, ref in enumerate(refs):
                self._emit(
                    "item.set_audio_gain",
                    {"item": ref, "gain_db": seg.gain_db},
                    FinishIntent(
                        kind="audio_gain",
                        op="item.set_audio_gain",
                        target=f"segment {index} (audio item {position})",
                        summary=f"{seg.gain_db:+g} dB",
                        resolved=self._resolved(ref),
                    ),
                )

    # -- titles --------------------------------------------------------------------------------

    def _titles(self) -> None:
        _check_title_track(self.plan)
        track = title_track(self.plan)
        for index, t in enumerate(self.plan.titles):
            self._one_title(
                f"title {index}",
                t.template,
                track,
                t.record_in,
                t.record_out,
                list(t.lines),
                kind="title",
            )
        if _has_sting(self.plan):
            sting = self.plan.outro.sting  # type: ignore[union-attr]
            start = self.plan.content_frames() + sting.start
            self._one_title(
                "sting",
                sting.template,
                track,
                start,
                start + sting.frames,
                list(sting.lines),
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
            self.notes.append(f"{label} has no template name; cannot place it")
            return
        if len(lines) > 4:
            # Refuse, do not silently drop the lower third (R8).
            raise CompileError(
                "title_too_many_lines",
                f"{label} has {len(lines)} lines; title.place allows at most 4",
                "split the title into at most 4 lines",
                lines=len(lines),
            )
        media = self.titles.get(template)
        if media is not None:
            ref: Any = media.model_dump(mode="json")
        else:
            # A template not (yet) in the prepare result is Unresolved, so submit refuses rather
            # than dropping the title (R8, W29).
            ref = unresolved(
                "title_template",
                f"pool id of title template {template!r}; run phase 'prepare' and pass "
                "--prepare-result",
            )
        self._emit(
            "title.place",
            {
                "template": ref,
                "track": track,
                "record_in": record_in,
                "record_out": record_out,
                "lines": lines,
            },
            FinishIntent(
                kind=kind,
                op="title.place",
                target=f"{label} on video track {track} (record {record_in}..{record_out})",
                summary=f"template {template!r}: {' / '.join(lines)}",
                resolved=self._resolved(ref),
            ),
        )

    # -- markers -------------------------------------------------------------------------------

    def _markers(self) -> None:
        for index, marker in enumerate(self.plan.markers):
            color = marker.color or "Blue"
            if color not in _MARKER_COLORS:
                self.notes.append(f"marker {index} colour {color!r} is not a Resolve colour; Blue")
                color = "Blue"
            self._emit(
                "markers.add",
                {
                    "frame": marker.record,
                    "color": color,
                    "name": marker.name,
                    "note": marker.note or "",
                    "duration": marker.duration,
                },
                FinishIntent(
                    kind="marker",
                    op="markers.add",
                    target=f"record {marker.record}",
                    summary=f"{color}{(' ' + marker.name) if marker.name else ''} "
                    f"({marker.duration}f)",
                    resolved=True,
                ),
            )

    # -- captions (F12) ------------------------------------------------------------------------

    def _captions(self) -> None:
        if self.plan.captions:
            self.notes.append(
                f"{len(self.plan.captions)} caption(s) are not applied by plan.apply "
                "(ffmpeg compose, S13e)"
            )
            self.intents.append(
                FinishIntent(
                    kind="captions",
                    op="(none)",
                    target=f"{len(self.plan.captions)} caption(s)",
                    summary="not applied by plan.apply (ffmpeg compose, S13e)",
                    resolved=True,
                )
            )

    # -- steps ---------------------------------------------------------------------------------

    def to_steps(self, plan: EditPlan) -> list[Step]:
        if self.resolver is None:
            tl: Any = unresolved("timeline", "the branch; run phase 'assemble'")
        else:
            tl = self.resolver.timeline.model_dump(mode="json")
        self._t["item.set_fades"] = _merge_fades(self._t["item.set_fades"])
        steps: list[Step] = []
        # transitions first: colour (incl. input colour space) is set after grades, titles land
        # on their own track, the rest address the resolved items.
        steps.extend(self._grouped("transition.add", "targets", tl, 50))
        steps.extend(self._colour_space_steps())
        for op, field_name, cap in (
            ("item.set_color", "targets", 50),
            ("item.set_transform", "targets", 50),
            ("item.set_audio_gain", "targets", 500),
            ("item.set_fades", "targets", 500),
            ("title.place", "titles", 50),
            ("markers.add", "markers", 1000),
        ):
            steps.extend(self._grouped(op, field_name, tl, cap))
        return steps

    def _colour_space_steps(self) -> list[Step]:
        # One step per colour space: clip.set_input_color_space takes one color_space per call
        # and no timeline (it keys live pool items).
        steps = []
        for index, payload in enumerate(self._t["clip.set_input_color_space"]):
            steps.append(
                _step(
                    "finish",
                    "clip.set_input_color_space",
                    f"clip.set_input_color_space:{index}",
                    payload,
                    f"input colour space {payload['color_space']!r} on "
                    f"{len(payload['items'])} item(s)",
                )
            )
        return steps

    def _grouped(self, op: str, field_name: str, tl: Any, cap: int) -> list[Step]:
        steps: list[Step] = []
        for batch_index, batch in enumerate(_chunk(self._t[op], cap)):
            params: dict[str, Any] = {field_name: batch, "timeline": tl}
            steps.append(
                _step(
                    "finish", op, f"{op}:{batch_index}", params, f"{op} on {len(batch)} target(s)"
                )
            )
        return steps


# -- helpers -----------------------------------------------------------------------------------


def _tl(working: TimelineRef | None, hint: str) -> Any:
    if working is not None:
        return working.model_dump(mode="json")
    return unresolved("timeline", hint)


def _fits(available: int | None, need: int) -> bool:
    # An unknown handle (no source duration / no neighbour) counts as NO handle: the conservative
    # choice, so a transition that might not build is refused rather than silently emitted (R7).
    return available is not None and available >= need


def _is_braw(path: str) -> bool:
    return path.strip().lower().split("?", 1)[0].endswith(".braw")


def _cdl_dict(cdl: Cdl) -> dict[str, Any]:
    return {
        "slope": list(cdl.slope),
        "offset": list(cdl.offset),
        "power": list(cdl.power),
        "saturation": cdl.saturation,
    }


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
    ranges = plan.record_ranges()
    best: tuple[int, int, int] | None = None
    for index, seg in enumerate(plan.segments):
        key = (ranges[index][0], seg.track)
        if best is None or key < (best[2], best[1]):
            best = (index, seg.track, ranges[index][0])
    return best


def _last_content_segment(plan: EditPlan) -> tuple[int, int, int] | None:
    ranges = plan.record_ranges()
    best: tuple[int, int, int] | None = None
    best_key: tuple[int, int] | None = None
    for index, seg in enumerate(plan.segments):
        key = (ranges[index][1], seg.track)
        if best_key is None or key > best_key:
            best_key, best = key, (index, seg.track, ranges[index][0])
    return best


def _merge_fades(targets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One fade target per item: an item at the timeline end can take both a head fade-in and an
    end fade-out, and item.set_fades refuses a repeated item."""
    merged: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for target in targets:
        item = target["item"]
        key = json.dumps(item, sort_keys=True)
        if key not in merged:
            merged[key] = dict(target)
            order.append(key)
        else:
            merged[key].update({k: v for k, v in target.items() if k != "item"})
    return [merged[k] for k in order]


def _chunk(items: list[Any], size: int) -> list[list[Any]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _check_batch(batch_size: int) -> None:
    if not 1 <= batch_size <= MAX_BATCH:
        raise CompileError(
            "batch_size_invalid",
            f"batch size {batch_size} is out of range (1..{MAX_BATCH})",
            f"AppendToTimeline hung 21.1 on large batches; keep it at 1..{MAX_BATCH}",
            batch_size=batch_size,
        )


def _check_steps(steps: list[Step]) -> None:
    if len(steps) > MAX_STEPS:
        raise CompileError(
            "too_many_steps",
            f"the phase compiled to {len(steps)} steps, over the {MAX_STEPS} ops.sequence limit",
            "split the build, or raise fewer per-segment finishing ops",
            steps=len(steps),
            limit=MAX_STEPS,
        )


def _step(phase: Phase, op: str, name: str, params: dict[str, Any], summary: str) -> Step:
    if not find_unresolved(params):
        params = get_op(op).params.model_validate(params).model_dump(mode="json")
    return Step(phase=phase, op=op, name=name, params=params, summary=summary)


# expose for the renderer / submit
__all__ = [
    "DEFAULT_BATCH",
    "MAX_BATCH",
    "MAX_STEPS",
    "TEMPLATE_MAX_ITEMS",
    "Compilation",
    "compile_assemble",
    "compile_finish",
    "compile_prepare",
    "plan_hash",
    "template_item_count",
    "title_track",
    "video_tracks",
]
