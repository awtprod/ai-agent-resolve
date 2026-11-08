"""Render the compiled three-phase build as the stable dry-run diff (text and JSON).

Unresolved references (ids an earlier phase has not yet produced) are printed, clearly marked,
so a preview of a later phase is still useful; a submission of such a phase is refused. With a
``--branch-describe`` the diff reports idempotency the way decision 5 asks: it lists only what is
missing, and "present" means a full identity match (track, record_in, source in/out, and media
when it is resolved).
"""

from __future__ import annotations

import json
from typing import Any

from aar.node.ops.read import DescribeResult
from aar.plan.apply.compile import Compilation, title_track
from aar.plan.apply.refs import find_unresolved, is_unresolved, show
from aar.plan.model import EditPlan


def _appended(assemble: Compilation) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for step in assemble.steps:
        if step.op != "timeline.append_ranges":
            continue
        for rng in step.params["ranges"]:
            rows.append(rng)
    return rows


def _media_id(media: Any) -> str | None:
    if is_unresolved(media):
        return None
    return media.get("unique_id")


# -- idempotency (F11) -------------------------------------------------------------------------


def _present_items(describe: DescribeResult) -> dict[tuple[str, int], list[Any]]:
    # Key by (track_type, index): a linked audio item on A1 must not read as a video append on V1
    # (R5).
    by_track: dict[tuple[str, int], list[Any]] = {}
    for track in describe.tracks:
        for item in track.items:
            by_track.setdefault((track.type, track.index), []).append(item)
    return by_track


def _item_present(
    items: list[Any], record_in: int, src_in: int, src_out: int, media_id: str | None
) -> bool:
    for item in items:
        if item.record_in != record_in:
            continue
        if item.source_start not in (None, src_in) or item.source_end not in (None, src_out):
            continue
        if media_id is not None and item.media_id not in (None, media_id):
            continue
        return True
    return False


def idempotency(
    plan: EditPlan, assemble: Compilation, finish: Compilation, describe: DescribeResult
) -> dict[str, Any]:
    by_track = _present_items(describe)
    marker_frames = {m.frame for m in describe.markers}
    missing_appends: list[str] = []
    present_appends = 0
    for rng in _appended(assemble):
        kind = "video" if rng["media_type"] in ("av", "video") else "audio"
        items = by_track.get((kind, rng["track"]), [])
        if _item_present(
            items, rng["record_in"], rng["src_in"], rng["src_out"], _media_id(rng["media"])
        ):
            present_appends += 1
        else:
            missing_appends.append(
                f"{kind[0].upper()}{rng['track']} @ {rng['record_in']} ({rng['label']})"
            )

    t_track = title_track(plan)
    missing_titles: list[str] = []
    present_titles = 0
    for intent in finish.intents:
        if intent.kind not in ("title", "sting"):
            continue
        record_in = _title_record_in(intent)
        if any(i.record_in == record_in for i in by_track.get(("video", t_track), [])):
            present_titles += 1
        else:
            missing_titles.append(intent.target)

    missing_markers = [f"record {m.record}" for m in plan.markers if m.record not in marker_frames]
    present_markers = len(plan.markers) - len(missing_markers)
    return {
        "appends": {"present": present_appends, "missing": missing_appends},
        "titles": {"present": present_titles, "missing": missing_titles},
        "markers": {"present": present_markers, "missing": missing_markers},
    }


def _title_record_in(intent: Any) -> int:
    span = intent.target.rsplit("record ", 1)[1].rstrip(")")
    return int(span.split("..", 1)[0])


# -- text --------------------------------------------------------------------------------------


def _phase_block(lines: list[str], comp: Compilation, label: str) -> None:
    lines.append(
        f"PHASE {comp.phase} - {label} ({len(comp.steps)} steps, "
        f"{comp.unresolved} unresolved ref(s))"
    )
    for number, step in enumerate(comp.steps, start=1):
        marks = [show(inner_wrap(i)) for _, i in find_unresolved(step.params)]
        suffix = f"  [UNRESOLVED: {', '.join(marks)}]" if marks else ""
        lines.append(f"  {number}. {step.op}  [{step.name}]{suffix}")
        lines.append(f"       {step.summary}")


def inner_wrap(inner: dict[str, Any]) -> dict[str, Any]:
    return {"__unresolved__": inner}


def render_text(
    plan: EditPlan,
    prepare: Compilation,
    assemble: Compilation,
    finish: Compilation,
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
        f"{plan.total_frames()} frame(s); batch size {assemble.batch_size}"
    )
    lines.append("")

    _phase_block(lines, prepare, "prepare")
    lines.append("")
    _phase_block(lines, assemble, "assemble")
    lines.append("  append ranges per video/audio track:")
    for track_line in _append_lines(assemble):
        lines.append(f"    {track_line}")
    lines.append("")

    lines.append(
        f"PHASE finish - finish ({len(finish.steps)} steps, {finish.unresolved} unresolved ref(s))"
    )
    for intent in finish.intents:
        flag = "" if intent.resolved else "  [unresolved]"
        lines.append(f"  {intent.kind:20} {intent.target}{flag}")
        lines.append(f"       {intent.summary}")
    lines.append("")

    notes = [*prepare.notes, *assemble.notes, *finish.notes]
    warnings = [*prepare.warnings, *assemble.warnings, *finish.warnings]
    lines.append(f"WARNINGS ({len(warnings)})")
    for w in warnings:
        lines.append(f"  ! {w}")
    lines.append(f"NOTES / GAPS ({len(notes)})")
    for note in notes:
        lines.append(f"  - {note}")

    if branch_describe is not None:
        lines.append("")
        status = idempotency(plan, assemble, finish, branch_describe)
        lines.append(
            f"IDEMPOTENCY (vs branch {branch_describe.timeline.name!r}) - only what is missing"
        )
        for kind in ("appends", "titles", "markers"):
            miss = status[kind]["missing"]
            lines.append(f"  {kind}: {status[kind]['present']} present, {len(miss)} to add")
            for item in miss:
                lines.append(f"    + {item}")
    return "\n".join(lines) + "\n"


def _append_lines(assemble: Compilation) -> list[str]:
    groups: dict[tuple[str, int], list[tuple[int, int]]] = {}
    for rng in _appended(assemble):
        kind = "V" if rng["media_type"] in ("av", "video") else "A"
        rec_out = rng["record_in"] + (rng["src_out"] - rng["src_in"])
        groups.setdefault((kind, rng["track"]), []).append((rng["record_in"], rec_out))
    out = []
    for (kind, track), ranges in sorted(groups.items()):
        shown = ", ".join(f"{a}..{b}" for a, b in ranges)
        out.append(f"{kind}{track}: {shown}  ({len(ranges)} range(s))")
    return out


# -- json --------------------------------------------------------------------------------------


def _phase_json(comp: Compilation) -> dict[str, Any]:
    return {
        "phase": comp.phase,
        "unresolved": comp.unresolved,
        "steps": [
            {"op": s.op, "name": s.name, "summary": s.summary, "params": s.params}
            for s in comp.steps
        ],
        "notes": comp.notes,
        "warnings": comp.warnings,
    }


def render_json(
    plan: EditPlan,
    prepare: Compilation,
    assemble: Compilation,
    finish: Compilation,
    *,
    branch_describe: DescribeResult | None = None,
) -> dict[str, Any]:
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
        "prepare": _phase_json(prepare),
        "assemble": _phase_json(assemble),
        "finish": {
            **_phase_json(finish),
            "intents": [i.model_dump() for i in finish.intents],
            "placements": [p.model_dump() for p in assemble.placements],
        },
    }
    if branch_describe is not None:
        body["idempotency"] = idempotency(plan, assemble, finish, branch_describe)
    return body


def dumps_json(body: dict[str, Any]) -> str:
    return json.dumps(body, indent=2, ensure_ascii=False) + "\n"
