"""Structural rules: valid ranges, chronological order, shortest shot, a short's duration and
its final fade."""

from __future__ import annotations

import math
from collections.abc import Mapping
from fractions import Fraction
from typing import Any

from aar.plan.lint import Finding, LintContext, check_number, rule


@rule(
    "ranges-valid",
    severity="error",
    summary="Ranges are positive, inside the media, and do not overlap on a track.",
)
def ranges_valid(ctx: LintContext) -> list[Finding]:
    plan = ctx.plan
    out: list[Finding] = []
    seen: set[str] = set()
    for source in plan.sources:
        if source.id in seen:
            out.append(
                ctx.finding(
                    "error",
                    f"Source id {source.id!r} is defined more than once.",
                    "Give every source a unique id.",
                )
            )
        seen.add(source.id)
    for index, seg in enumerate(plan.segments):
        source = ctx.source(index)
        if source is None:
            out.append(
                ctx.finding(
                    "error",
                    f"Segment {index} uses unknown source {seg.source!r}.",
                    "Add the source to `sources` or fix the id.",
                    [ctx.ref(index)],
                )
            )
        if seg.src_out <= seg.src_in:
            out.append(
                ctx.finding(
                    "error",
                    f"Segment {index} has src_out {seg.src_out} <= src_in {seg.src_in}; "
                    "ranges are [in, out) so out must be greater than in.",
                    "Swap or correct the in and out frames.",
                    [ctx.ref(index, "in"), ctx.ref(index, "out")],
                )
            )
        elif source is not None and source.duration is not None and seg.src_out > source.duration:
            out.append(
                ctx.finding(
                    "error",
                    f"Segment {index} ends at source frame {seg.src_out}, past the end of "
                    f"source {source.id!r} ({source.duration} frames).",
                    f"Trim src_out to at most {source.duration}.",
                    [ctx.ref(index, "out")],
                )
            )
        if seg.length > 0:
            for key in seg.reframe:
                if key.time > seg.length:
                    out.append(
                        ctx.finding(
                            "error",
                            f"Segment {index} has a reframe keyframe at frame {key.time}, "
                            f"after the segment ends ({seg.length} frames).",
                            "Keyframe times are frames from the segment start.",
                            [ctx.ref(index)],
                        )
                    )
    for outgoing, incoming in plan.joins():
        end, start = ctx.ranges[outgoing][1], ctx.ranges[incoming][0]
        if start < end:
            out.append(
                ctx.finding(
                    "error",
                    f"Segment {incoming} starts at record frame {start}, before segment "
                    f"{outgoing} ends at {end}, on the same track.",
                    "Move one segment to another track or fix record_in.",
                    [ctx.ref(outgoing, "out"), ctx.ref(incoming, "in")],
                )
            )
    joins = set(plan.joins())
    for number, transition in enumerate(plan.transitions):
        after = transition.after
        if after >= len(plan.segments) or not any(a == after for a, _ in joins):
            out.append(
                ctx.finding(
                    "error",
                    f"Transition {number} follows segment {after}, which has no following "
                    "segment on its track.",
                    "Set `after` to the index of the outgoing segment of a join.",
                )
            )
    for kind, items in (("title", plan.titles), ("caption", plan.captions)):
        for number, item in enumerate(items):
            if item.record_out <= item.record_in:
                out.append(
                    ctx.finding(
                        "error",
                        f"{kind.capitalize()} {number} ends at {item.record_out}, not after "
                        f"its start {item.record_in}.",
                        "record_out is exclusive and must be greater than record_in.",
                        [ctx.record_ref(item.record_in, f"{kind}[{number}]")],
                    )
                )
    return out


@rule(
    "chronological",
    severity="error",
    summary="Segments from one recording play in source-time order.",
    params=("tolerance_frames",),
)
def chronological(ctx: LintContext) -> list[Finding]:
    tolerance = int(ctx.params["tolerance_frames"])
    fps = ctx.plan.fps
    out: list[Finding] = []
    last: dict[str, int] = {}
    for index in range(len(ctx.plan.segments)):
        if not ctx.usable(index):
            continue
        key = ctx.recording(index)
        assert key is not None
        previous = last.get(key)
        last[key] = index
        if previous is None:
            continue
        prev_in, prev_out = ctx.recording_range(previous)
        now_in, _ = ctx.recording_range(index)
        if now_in >= prev_out - tolerance:
            continue
        how = "goes back in time" if now_in < prev_in else "repeats material already played"
        out.append(
            ctx.finding(
                "error",
                f"Segment {index} {how}: it starts at {fps.timecode(now_in)} of recording "
                f"{key!r} but segment {previous} already played up to {fps.timecode(prev_out)}.",
                "Reorder or drop segments so each recording plays forward only; "
                "do not import a cold open from later in the recording.",
                [ctx.ref(previous, "out"), ctx.ref(index, "in")],
            )
        )
    return out


@rule(
    "short-max-duration",
    severity="error",
    summary="A short runs under the maximum duration.",
    params=("max_seconds",),
)
def short_max_duration(ctx: LintContext) -> list[Finding]:
    if ctx.plan.kind != "short":
        return []
    fps = ctx.plan.fps
    limit = Fraction(str(ctx.params["max_seconds"]))
    total = ctx.plan.total_frames()
    if Fraction(total * fps.den, fps.num) < limit:
        return []
    allowed = math.ceil(limit * fps.num / fps.den) - 1
    return [
        ctx.finding(
            "error",
            f"The short runs {fps.seconds(total):.3f} s ({total} frames); it must be under "
            f"{float(limit):g} s.",
            f"Cut at least {total - allowed} frames so the total is at most {allowed} frames.",
            [ctx.record_ref(total, "end")],
        )
    ]


def _min_shot_params(given: Mapping[str, Any], merged: dict[str, Any]) -> dict[str, Any]:
    check_number(merged, "min_seconds", minimum=0)
    return merged


@rule(
    "min-shot",
    severity="warn",
    summary="No slivers: every content segment runs at least the minimum record time.",
    params=("min_seconds",),
    prepare=_min_shot_params,
)
def min_shot(ctx: LintContext) -> list[Finding]:
    plan = ctx.plan
    fps = plan.fps
    limit = Fraction(str(ctx.params["min_seconds"]))
    need = math.ceil(limit * fps.num / fps.den)
    before = {incoming: outgoing for outgoing, incoming in plan.joins()}
    after = {outgoing: incoming for outgoing, incoming in plan.joins()}
    out: list[Finding] = []
    for index, seg in enumerate(plan.segments):
        if not ctx.usable(index) or Fraction(seg.length * fps.den, fps.num) >= limit:
            continue
        neighbours = [n for n in (before.get(index), after.get(index)) if n is not None]
        grow = f"lengthen it to at least {need} frames"
        if neighbours:
            names = " or ".join(f"segment {n}" for n in neighbours)
            hint = (
                f"Fold it into a neighbour: extend {names} over its {seg.length} frames, or {grow}."
            )
        else:
            hint = f"It has no neighbour on its track to fold into: {grow}."
        out.append(
            ctx.finding(
                "warn",
                f"Segment {index} runs {fps.seconds(seg.length):.3f} s ({seg.length} frames); "
                f"shots under {float(limit):g} s read as slivers.",
                hint[0].upper() + hint[1:],
                [ctx.ref(index, "in"), ctx.ref(index, "out")],
            )
        )
    return out


def _short_end_fade_params(given: Mapping[str, Any], merged: dict[str, Any]) -> dict[str, Any]:
    check_number(merged, "max_frames", integer=True, minimum=0)
    return merged


@rule(
    "short-end-fade",
    severity="warn",
    summary="A short's final audio fade is a de-click only, never a fade to silence.",
    params=("max_frames",),
    prepare=_short_end_fade_params,
)
def short_end_fade(ctx: LintContext) -> list[Finding]:
    plan = ctx.plan
    frames = plan.end_fade_frames
    limit = int(ctx.params["max_frames"])
    if plan.kind != "short" or frames is None or frames <= limit:
        return []
    total = plan.total_frames()
    return [
        ctx.finding(
            "warn",
            f"The short's final audio fade is {frames} frames "
            f"({plan.fps.seconds(frames):.2f} s); a short ends hard on the laugh, so the fade "
            f"is only a de-click of at most {limit} frame(s), never a fade to silence.",
            f"Set end_fade_frames to {limit} or less.",
            [ctx.record_ref(total, "end")],
        )
    ]
