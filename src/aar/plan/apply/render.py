"""Render a compiled plan as the stable dry-run diff (text and JSON).

The text form is golden-tested, so every line here is a contract. With a ``--branch-describe``
(a ``timeline.describe`` of an existing agent branch) the diff also reports idempotency: an
append, title or marker already on the branch is "present", not re-added, so a re-apply only
does what is missing. (Transitions, colour, transforms, gain and fades have no reliable
read-back in a describe -- 21.1 has no CDL getter -- so the ops confirm those as present by
their own read-back at run time; the preview marks them "re-sent".)
"""

from __future__ import annotations

import json
from typing import Any

from aar.node.ops.read import DescribeResult
from aar.plan.apply.compile import BuildCompilation, FinishCompilation
from aar.plan.model import EditPlan

_RESENT_KINDS = frozenset(
    {
        "colour",
        "transform",
        "transition",
        "audio_crossfade",
        "outro_transition",
        "head_fade",
        "end_fade",
        "audio_gain",
    }
)


def _appended(build: BuildCompilation) -> list[tuple[int, int, int]]:
    """Every appended range (content segments and the outro) as ``(track, record_in,
    record_out)``, read from the steps so previews match the ``timeline.append_ranges`` step."""
    rows: list[tuple[int, int, int]] = []
    for step in build.steps:
        if step.op != "timeline.append_ranges":
            continue
        for rng in step.params["ranges"]:
            record_in = rng["record_in"]
            rows.append((rng["track"], record_in, record_in + (rng["src_out"] - rng["src_in"])))
    return rows


def _append_groups(build: BuildCompilation) -> dict[int, list[tuple[int, int]]]:
    groups: dict[int, list[tuple[int, int]]] = {}
    for track, record_in, record_out in _appended(build):
        groups.setdefault(track, []).append((record_in, record_out))
    return groups


def _present_items(describe: DescribeResult) -> set[tuple[int, int]]:
    """``(track, record_in)`` of every item already on a video track of the branch."""
    present: set[tuple[int, int]] = set()
    for track in describe.tracks:
        if track.type != "video":
            continue
        for item in track.items:
            present.add((track.index, item.record_in))
    return present


def _present_markers(describe: DescribeResult) -> set[int]:
    return {marker.frame for marker in describe.markers}


def idempotency(
    plan: EditPlan,
    build: BuildCompilation,
    finish: FinishCompilation,
    describe: DescribeResult,
) -> dict[str, Any]:
    """What a re-apply would skip (present) versus do (missing), from the branch describe."""
    present = _present_items(describe)
    marker_frames = _present_markers(describe)
    title_track = max((p.track for p in build.placements), default=0) + 1

    appends = {"present": 0, "missing": 0}
    for track, record_in, _ in _appended(build):
        appends["present" if (track, record_in) in present else "missing"] += 1

    markers = {"present": 0, "missing": 0}
    titles = {"present": 0, "missing": 0}
    resent = 0
    for intent in finish.intents:
        if intent.kind == "marker":
            frame = int(intent.target.rsplit(" ", 1)[1])
            markers["present" if frame in marker_frames else "missing"] += 1
        elif intent.kind in ("title", "sting"):
            here = (title_track, _title_record_in(intent)) in present
            titles["present" if here else "missing"] += 1
        elif intent.kind in _RESENT_KINDS:
            resent += 1
    return {"appends": appends, "titles": titles, "markers": markers, "re_sent": resent}


def _title_record_in(intent: Any) -> int:
    # target looks like "... (record <in>..<out>)"
    span = intent.target.rsplit("record ", 1)[1].rstrip(")")
    return int(span.split("..", 1)[0])


# -- text --------------------------------------------------------------------------------------


def render_text(
    plan: EditPlan,
    build: BuildCompilation,
    finish: FinishCompilation,
    *,
    branch_describe: DescribeResult | None = None,
) -> str:
    lines: list[str] = []
    fps = plan.fps
    lines.append(
        f"plan apply: {plan.project} / {plan.timeline}  "
        f"(template {plan.template!r}, {plan.kind}, {fps.num}/{fps.den} fps)"
    )
    lines.append(
        f"profile {plan.profile}: {len(plan.segments)} segment(s), "
        f"{plan.total_frames()} frame(s); batch size {build.batch_size}"
    )
    lines.append("")

    lines.append(f"PHASE A - build ({len(build.steps)} steps)")
    for number, step in enumerate(build.steps, start=1):
        lines.append(f"  {number}. {step.op}  [{step.name}]")
        lines.append(f"       {step.summary}")
    lines.append("  append ranges per video track:")
    groups = _append_groups(build)
    for track in sorted(groups):
        ranges = groups[track]
        batches = -(-len(ranges) // build.batch_size)
        shown = ", ".join(f"{a}..{b}" for a, b in ranges)
        lines.append(f"    track {track}: {shown}  ({len(ranges)} range(s), {batches} batch(es))")
    lines.append("")

    lines.append(f"PHASE B - finish ({len(finish.intents)} actions)")
    for intent in finish.intents:
        lines.append(f"  {intent.kind:16} {intent.target}")
        lines.append(f"       {intent.summary}")
    if finish.resolved:
        lines.append(f"  (resolved into {len(finish.steps)} grouped step(s))")
    lines.append("")

    notes = [*build.notes, *finish.notes]
    lines.append(f"NOTES / GAPS ({len(notes)})")
    for note in notes:
        lines.append(f"  - {note}")

    if branch_describe is not None:
        lines.append("")
        status = idempotency(plan, build, finish, branch_describe)
        lines.append(f"IDEMPOTENCY (vs branch {branch_describe.timeline.name!r})")
        lines.append(
            f"  appends: {status['appends']['present']} present, "
            f"{status['appends']['missing']} to add"
        )
        lines.append(
            f"  titles:  {status['titles']['present']} present, "
            f"{status['titles']['missing']} to add"
        )
        lines.append(
            f"  markers: {status['markers']['present']} present, "
            f"{status['markers']['missing']} to add"
        )
        lines.append(f"  colour/transforms/transitions/fades: {status['re_sent']} re-sent")

    return "\n".join(lines) + "\n"


# -- json --------------------------------------------------------------------------------------


def render_json(
    plan: EditPlan,
    build: BuildCompilation,
    finish: FinishCompilation,
    *,
    branch_describe: DescribeResult | None = None,
) -> dict[str, Any]:
    groups = _append_groups(build)
    body: dict[str, Any] = {
        "plan": {
            "project": plan.project,
            "timeline": plan.timeline,
            "template": plan.template,
            "kind": plan.kind,
            "fps": {"num": plan.fps.num, "den": plan.fps.den},
            "segments": len(plan.segments),
            "frames": plan.total_frames(),
        },
        "batch_size": build.batch_size,
        "phase_a": {
            "steps": [
                {"op": s.op, "name": s.name, "summary": s.summary, "params": s.params}
                for s in build.steps
            ],
            "appends": [
                {"track": track, "ranges": [[a, b] for a, b in groups[track]]}
                for track in sorted(groups)
            ],
            "placements": [p.model_dump() for p in build.placements],
        },
        "phase_b": {
            "resolved": finish.resolved,
            "intents": [i.model_dump() for i in finish.intents],
            "steps": [
                {"op": s.op, "name": s.name, "summary": s.summary, "params": s.params}
                for s in finish.steps
            ],
        },
        "notes": [*build.notes, *finish.notes],
    }
    if branch_describe is not None:
        body["idempotency"] = idempotency(plan, build, finish, branch_describe)
    return body


def dumps_json(body: dict[str, Any]) -> str:
    return json.dumps(body, indent=2, ensure_ascii=False) + "\n"
