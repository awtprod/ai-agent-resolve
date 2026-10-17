"""Rules about joins between segments: jump cuts, shot sizes, transitions and reframing."""

from __future__ import annotations

import math

from aar.plan.lint import Finding, LintContext, Ref, rule
from aar.plan.model import Segment, Transition

_SIZE_RANK = {"close": 0, "medium": 1, "wide": 2}


def _zoom_out(seg: Segment) -> float:
    return seg.reframe[-1].zoom if seg.reframe else 1.0


def _zoom_in(seg: Segment) -> float:
    return seg.reframe[0].zoom if seg.reframe else 1.0


def _join_refs(ctx: LintContext, outgoing: int, incoming: int) -> list[Ref]:
    return [ctx.ref(outgoing, "out"), ctx.ref(incoming, "in")]


@rule(
    "no-same-camera-jump",
    severity="error",
    summary="No jump cut: same camera with a time jump needs a different framing.",
    params=("min_zoom_delta", "continuity_tolerance_frames"),
)
def no_same_camera_jump(ctx: LintContext) -> list[Finding]:
    min_delta = float(ctx.params["min_zoom_delta"])
    tolerance = int(ctx.params["continuity_tolerance_frames"])
    out: list[Finding] = []
    for outgoing, incoming in ctx.plan.joins():
        if not (ctx.usable(outgoing) and ctx.usable(incoming)):
            continue
        first, second = ctx.source(outgoing), ctx.source(incoming)
        assert first is not None and second is not None
        same_camera = first.id == second.id or (
            first.camera is not None and first.camera == second.camera
        )
        if not same_camera or not ctx.time_jump(outgoing, incoming, tolerance):
            continue
        z_out = _zoom_out(ctx.plan.segments[outgoing])
        z_in = _zoom_in(ctx.plan.segments[incoming])
        delta = abs(z_in - z_out)
        if delta >= min_delta:
            continue
        out.append(
            ctx.finding(
                "error",
                f"Jump cut: segments {outgoing} and {incoming} are the same camera "
                f"({first.camera or first.id!r}) with a time jump, and the zoom changes by "
                f"only {delta:.2f} ({z_out:g} to {z_in:g}); at least {min_delta:g} is needed.",
                f"Cut to another camera, or punch in at least {min_delta:g} zoom on one side "
                "of the cut.",
                _join_refs(ctx, outgoing, incoming),
            )
        )
    return out


@rule(
    "timejump-close-to-wide",
    severity="warn",
    summary="At a time jump the cut should go from a tighter shot to a wider one.",
    params=("min_size_step", "continuity_tolerance_frames"),
)
def timejump_close_to_wide(ctx: LintContext) -> list[Finding]:
    step = int(ctx.params["min_size_step"])
    tolerance = int(ctx.params["continuity_tolerance_frames"])
    out: list[Finding] = []
    unknown: list[Ref] = []
    for outgoing, incoming in ctx.plan.joins():
        if not (ctx.usable(outgoing) and ctx.usable(incoming)):
            continue
        if not ctx.time_jump(outgoing, incoming, tolerance):
            continue
        first, second = ctx.source(outgoing), ctx.source(incoming)
        assert first is not None and second is not None
        if first.shot_size not in _SIZE_RANK or second.shot_size not in _SIZE_RANK:
            unknown.append(ctx.ref(incoming, "in"))
            continue
        gain = _SIZE_RANK[second.shot_size] - _SIZE_RANK[first.shot_size]
        if gain >= step:
            continue
        out.append(
            ctx.finding(
                "warn",
                f"Time jump between segments {outgoing} and {incoming} goes "
                f"{first.shot_size} to {second.shot_size}; it should go to a wider shot.",
                "Reorder the angles so the cut after the jump is wider (close to wide), "
                "or move the jump to a point where the angles already differ that way.",
                _join_refs(ctx, outgoing, incoming),
            )
        )
    if unknown:
        out.append(
            ctx.finding(
                "info",
                f"Could not check {len(unknown)} time jump(s): a source has shot_size unknown.",
                "Set shot_size to close, medium or wide on every source.",
                unknown,
            )
        )
    return out


@rule("no-dissolve", severity="error", summary="Video transitions are cuts.")
def no_dissolve(ctx: LintContext) -> list[Finding]:
    out: list[Finding] = []
    count = len(ctx.plan.segments)
    for transition in ctx.plan.transitions:
        if transition.video == "cut":
            continue
        refs = [ctx.ref(transition.after, "out")] if transition.after < count else []
        out.append(
            ctx.finding(
                "error",
                f"Video transition after segment {transition.after} is {transition.video!r} "
                f"({transition.video_frames} frames); only hard cuts are allowed.",
                'Set the transition\'s video to "cut" with video_frames 0.',
                refs,
            )
        )
    return out


@rule(
    "audio-crossfade",
    severity="error",
    summary="Audio joins use the crossfade type, short, with enough media on both sides.",
    params=(
        "audio_type",
        "max_frames",
        "handle_fraction",
        "require_on_cuts",
        "continuity_tolerance_frames",
    ),
)
def audio_crossfade(ctx: LintContext) -> list[Finding]:
    wanted = str(ctx.params["audio_type"])
    max_frames = int(ctx.params["max_frames"])
    fraction = float(ctx.params["handle_fraction"])
    required = bool(ctx.params["require_on_cuts"])
    tolerance = int(ctx.params["continuity_tolerance_frames"])
    by_after: dict[int, Transition] = {}
    for transition in ctx.plan.transitions:
        by_after.setdefault(transition.after, transition)
    out: list[Finding] = []
    unchecked: list[Ref] = []
    for outgoing, incoming in ctx.plan.joins():
        if not (ctx.usable(outgoing) and ctx.usable(incoming)):
            continue
        if ctx.is_continuous(outgoing, incoming, tolerance):
            continue  # the audio plays on unbroken, so there is no edit to smooth
        refs = _join_refs(ctx, outgoing, incoming)
        audio = by_after[outgoing].audio if outgoing in by_after else None
        if audio is None:
            if required:
                out.append(
                    ctx.finding(
                        "error",
                        f"The join between segments {outgoing} and {incoming} has no audio "
                        "transition, so it is a hard audio cut.",
                        f'Add a transition after segment {outgoing} with audio type "{wanted}" '
                        f"and at most {max_frames} frames.",
                        refs,
                    )
                )
            continue
        if audio.type != wanted:
            out.append(
                ctx.finding(
                    "error",
                    f"Audio transition after segment {outgoing} is {audio.type!r}, not {wanted!r}.",
                    f'Use audio type "{wanted}".',
                    refs,
                )
            )
        if audio.frames > max_frames:
            out.append(
                ctx.finding(
                    "error",
                    f"Audio crossfade after segment {outgoing} is {audio.frames} frames; "
                    f"the maximum is {max_frames}.",
                    f"Shorten it to {max_frames} frames or fewer.",
                    refs,
                )
            )
        need = math.ceil(audio.frames * fraction)
        first, second = ctx.source(outgoing), ctx.source(incoming)
        assert first is not None and second is not None
        have_in = ctx.plan.segments[incoming].src_in
        if have_in < need:
            out.append(
                ctx.finding(
                    "error",
                    f"Segment {incoming} has only {have_in} frames of media before its in "
                    f"point; a {audio.frames}-frame crossfade needs {need}.",
                    f"Start segment {incoming} at least {need - have_in} frames later, or "
                    "shorten the crossfade.",
                    refs,
                )
            )
        if first.duration is None:
            unchecked.append(ctx.ref(outgoing, "out"))
            continue
        have_out = first.duration - ctx.plan.segments[outgoing].src_out
        if have_out < need:
            out.append(
                ctx.finding(
                    "error",
                    f"Segment {outgoing} has only {have_out} frames of media after its out "
                    f"point; a {audio.frames}-frame crossfade needs {need}.",
                    f"End segment {outgoing} at least {need - have_out} frames earlier, or "
                    "shorten the crossfade.",
                    refs,
                )
            )
    if unchecked:
        out.append(
            ctx.finding(
                "info",
                f"Skipped the outgoing handle check on {len(unchecked)} join(s): "
                "the source's media duration is unknown.",
                "Set `duration` (frames) on the sources to enable the handle check.",
                unchecked,
            )
        )
    return out


@rule(
    "no-whip-pan",
    severity="error",
    summary="Reframe keyframes move and zoom slowly.",
    params=("max_pan_speed", "max_zoom_speed"),
)
def no_whip_pan(ctx: LintContext) -> list[Finding]:
    max_pan = float(ctx.params["max_pan_speed"])
    max_zoom = float(ctx.params["max_zoom_speed"])
    fps = ctx.plan.fps
    out: list[Finding] = []
    for index, seg in enumerate(ctx.plan.segments):
        for first, second in zip(seg.reframe, seg.reframe[1:], strict=False):
            seconds = fps.seconds(second.time - first.time)
            distance = math.hypot(second.x - first.x, second.y - first.y)
            zoom = abs(second.zoom - first.zoom)
            refs = [ctx.ref_at(index, first.time), ctx.ref_at(index, second.time)]
            pan_speed = math.inf if seconds == 0 and distance > 0 else 0.0
            zoom_speed = math.inf if seconds == 0 and zoom > 0 else 0.0
            if seconds > 0:
                pan_speed, zoom_speed = distance / seconds, zoom / seconds
            if pan_speed > max_pan:
                out.append(
                    ctx.finding(
                        "error",
                        f"Whip-pan in segment {index}: the frame centre moves "
                        f"{distance:.3f} frame widths in {seconds:.2f} s "
                        f"({pan_speed:.2f}/s); the limit is {max_pan:g}/s.",
                        "Hold a steady frame, or spread the move over more time.",
                        refs,
                    )
                )
            if zoom_speed > max_zoom:
                out.append(
                    ctx.finding(
                        "error",
                        f"Fast zoom in segment {index}: zoom changes {zoom:.3f} in "
                        f"{seconds:.2f} s ({zoom_speed:.2f}/s); the limit is {max_zoom:g}/s.",
                        "Hold the zoom, or spread the change over more time.",
                        refs,
                    )
                )
    return out
