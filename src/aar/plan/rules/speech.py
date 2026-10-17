"""Rules that need a transcript or a laugh map: dialogue splices and ending on the laugh."""

from __future__ import annotations

from aar.plan.inputs import Word
from aar.plan.lint import Finding, LintContext, Point, Ref, rule


def _cut_points(ctx: LintContext, tolerance: int) -> list[tuple[int, Point]]:
    """Segment ends that are real edits: not the unbroken continuation of a neighbour."""
    before = {incoming: outgoing for outgoing, incoming in ctx.plan.joins()}
    after = {outgoing: incoming for outgoing, incoming in ctx.plan.joins()}
    points: list[tuple[int, Point]] = []
    for index in range(len(ctx.plan.segments)):
        if not ctx.usable(index):
            continue
        prev, nxt = before.get(index), after.get(index)
        if prev is None or not (ctx.usable(prev) and ctx.is_continuous(prev, index, tolerance)):
            points.append((index, "in"))
        if nxt is None or not (ctx.usable(nxt) and ctx.is_continuous(index, nxt, tolerance)):
            points.append((index, "out"))
    return points


def _pauses(words: list[Word], minimum_s: float) -> list[tuple[float, float]]:
    """Gaps between words that last at least ``minimum_s`` seconds."""
    found: list[tuple[float, float]] = []
    latest_end = None
    for word in words:
        if latest_end is not None and word.start_s - latest_end >= minimum_s:
            found.append((latest_end, word.start_s))
        latest_end = word.end_s if latest_end is None else max(latest_end, word.end_s)
    return found


def _source_frame(ctx: LintContext, index: int, seconds: float) -> int:
    source = ctx.source(index)
    assert source is not None
    return ctx.plan.fps.frames(seconds) - source.recording_origin


@rule(
    "no-dialogue-splice",
    severity="error",
    summary="A cut never lands inside a word and should sit in a pause.",
    params=("min_pause_ms", "word_tolerance_ms", "continuity_tolerance_frames"),
    needs=("transcript",),
)
def no_dialogue_splice(ctx: LintContext) -> list[Finding]:
    transcript = ctx.transcript
    assert transcript is not None
    min_pause = float(ctx.params["min_pause_ms"]) / 1000
    tol = float(ctx.params["word_tolerance_ms"]) / 1000
    fps = ctx.plan.fps
    out: list[Finding] = []
    uncovered: list[Ref] = []
    for index, point in _cut_points(ctx, int(ctx.params["continuity_tolerance_frames"])):
        if not ctx.applies(transcript.recording, index):
            uncovered.append(ctx.ref(index, point))
            continue
        frames = ctx.recording_range(index)[0 if point == "in" else 1]
        at = fps.seconds(frames)
        ref = ctx.ref(index, point)
        label = f"The {point} point of segment {index} ({ref.record_tc} on the timeline)"
        inside = next(
            (w for w in transcript.words if w.start_s + tol < at < w.end_s - tol),
            None,
        )
        if inside is not None:
            before = _source_frame(ctx, index, inside.start_s)
            after = _source_frame(ctx, index, inside.end_s)
            out.append(
                ctx.finding(
                    "error",
                    f'{label} falls inside the word "{inside.text}" '
                    f"({inside.start_s:.2f}-{inside.end_s:.2f} s).",
                    f"Move it to source frame {before} (before the word) or {after} (after it), "
                    "ideally inside a pause.",
                    [ref],
                )
            )
            continue
        prev_ends = [w for w in transcript.words if w.end_s <= at + tol]
        next_starts = [w for w in transcript.words if w.start_s >= at - tol]
        if not prev_ends or not next_starts:
            continue  # speech starts or ends beyond this cut: nothing is spliced
        prev_word = max(prev_ends, key=lambda w: w.end_s)
        next_word = min(next_starts, key=lambda w: w.start_s)
        gap = next_word.start_s - prev_word.end_s
        if gap >= min_pause:
            continue
        hint = f"Move the cut into a pause of at least {min_pause * 1000:g} ms."
        pauses = _pauses(transcript.words, min_pause)
        if pauses:
            start, end = min(pauses, key=lambda p: min(abs(p[0] - at), abs(p[1] - at)))
            middle = _source_frame(ctx, index, (start + end) / 2)
            hint += f" The nearest is {start:.2f}-{end:.2f} s; source frame {middle} is its middle."
        out.append(
            ctx.finding(
                "warn",
                f'{label} sits in a {max(gap, 0) * 1000:.0f} ms gap between "{prev_word.text}" '
                f'and "{next_word.text}"; at least {min_pause * 1000:g} ms is wanted.',
                hint,
                [ref],
            )
        )
    if uncovered:
        out.append(
            ctx.finding(
                "info",
                f"The transcript is for recording {transcript.recording!r}; {len(uncovered)} "
                "cut point(s) in other recordings were not checked.",
                "Supply a transcript for each recording in the plan.",
                uncovered,
            )
        )
    return out


@rule(
    "end-on-laugh",
    severity="warn",
    summary="The final out point lands just after the end of a laugh.",
    params=("max_after_laugh_ms", "min_strength"),
    needs=("laughs",),
)
def end_on_laugh(ctx: LintContext) -> list[Finding]:
    laughs = ctx.laughs
    assert laughs is not None
    plan = ctx.plan
    usable = [i for i in range(len(plan.segments)) if ctx.usable(i)]
    if not usable:
        return []
    last = max(usable, key=lambda i: (ctx.ranges[i][1], i))
    ref = ctx.ref(last, "out")
    if not ctx.applies(laughs.recording, last):
        return [
            ctx.finding(
                "info",
                f"The laugh map is for recording {laughs.recording!r}, not the one the final "
                f"segment {last} comes from; the ending was not checked.",
                "Supply a laugh map for the recording of the final segment.",
                [ref],
            )
        ]
    max_after = float(ctx.params["max_after_laugh_ms"]) / 1000
    min_strength = float(ctx.params["min_strength"])
    half_frame = plan.fps.seconds(1) / 2
    at = plan.fps.seconds(ctx.recording_range(last)[1])
    strong = [laugh for laugh in laughs.laughs if laugh.strength >= min_strength]
    if any(-half_frame <= at - laugh.end_s <= max_after + half_frame for laugh in strong):
        return []
    if not strong:
        return [
            ctx.finding(
                "warn",
                "The laugh map has no laugh to end on.",
                "Check the laugh map, or end on the punchline by hand.",
                [ref],
            )
        ]
    nearest = min(strong, key=lambda laugh: abs(at - laugh.end_s))
    offset_ms = (at - nearest.end_s) * 1000
    if offset_ms < 0:
        how = f"{-offset_ms:.0f} ms before the nearest laugh ends ({nearest.end_s:.2f} s)"
    else:
        how = f"{offset_ms:.0f} ms after the nearest laugh ended ({nearest.end_s:.2f} s)"
    target = _source_frame(ctx, last, nearest.end_s)
    return [
        ctx.finding(
            "warn",
            f"The edit ends {how}; it should end within {max_after * 1000:g} ms after a laugh.",
            f"Set the final src_out to source frame {target} (the end of that laugh).",
            [ref],
        )
    ]
