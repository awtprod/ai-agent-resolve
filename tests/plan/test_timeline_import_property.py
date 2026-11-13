"""Timeline import (S20) property test: random edit scripts on V1 of a plan's layout, imported,
and the edited plan's own layout compared frame for frame with the edited timeline on V1."""

from __future__ import annotations

import random
from dataclasses import replace

import pytest
from export_support import permissive
from timeline_import_support import Row, build_timeline, content, rows_from_layout

from aar.plan.export.layout import build_layout
from aar.plan.model import EditPlan
from aar.plan.timeline_import import import_timeline

FILES = {"camA": "E:\\Show\\A.mov", "camB": "E:\\Show\\B.mov", "broll": "E:\\Show\\Crowd.mov"}
DURATION = 3000


def same_file(path: str) -> str:
    """The test's own reading of two Windows paths naming one file (not the product's
    ``media_key``, so a wrong ``media_key`` cannot pass here)."""
    return path.replace("/", "\\").lower()


def base_plan(rng: random.Random) -> EditPlan:
    segments = []
    for _ in range(rng.randint(1, 6)):
        source = rng.choice(["camA", "camB"])
        src_in = rng.randrange(0, DURATION - 200)
        segments.append(
            {"source": source, "src_in": src_in, "src_out": src_in + rng.randint(2, 200)}
        )
    data = {
        "version": 2,
        "project": "P",
        "timeline": "T",
        "kind": "longform",
        "fps": rng.choice([25, "30000/1001", 24]),
        "resolution": {"width": 1920, "height": 1080},
        "profile": "ccn",
        "sources": [
            {"id": sid, "media": path, "duration": DURATION, "start_timecode": 0}
            for sid, path in FILES.items()
        ]
        + [{"id": "out", "media": "E:\\Assets\\Outro.mov", "duration": 90, "start_timecode": 0}],
        "segments": segments,
        "outro": {"source": "out", "audio_track": "A1"},
        "audio_tracks": [{"name": "A1"}],
    }
    return EditPlan.model_validate(data)


def edit(rng: random.Random, clips: list[Row]) -> list[Row]:
    """One random edit of the V1 content clips (record positions are set afterwards)."""
    op = rng.choice(["head", "tail", "delete", "swap", "split", "add", "extend"])
    if not clips:
        op = "add"
    i = rng.randrange(len(clips)) if clips else 0
    out = list(clips)
    if op == "head" and clips[i].duration > 1:
        d = rng.randint(1, clips[i].duration - 1)
        out[i] = replace(clips[i], src_in=clips[i].src_in + d, duration=clips[i].duration - d)
    elif op == "tail" and clips[i].duration > 1:
        out[i] = replace(clips[i], duration=rng.randint(1, clips[i].duration - 1))
    elif op == "extend":
        assert clips[i].src_in is not None
        room = DURATION - (clips[i].src_in + clips[i].duration)
        out[i] = replace(clips[i], duration=clips[i].duration + rng.randint(0, min(room, 50)))
    elif op == "delete":
        del out[i]
    elif op == "swap" and len(clips) > 1:
        j = rng.randrange(len(clips))
        out[i], out[j] = out[j], out[i]
    elif op == "split" and clips[i].duration > 1:
        assert clips[i].src_in is not None
        cut = rng.randint(1, clips[i].duration - 1)
        gap = rng.randint(0, 20)
        first = replace(clips[i], duration=cut)
        second = replace(
            clips[i], src_in=clips[i].src_in + cut + gap, duration=clips[i].duration - cut - gap
        )
        out[i : i + 1] = [first] + ([second] if second.duration > 0 else [])
    elif op == "add":
        source = rng.choice(list(FILES))
        src_in = rng.randrange(0, DURATION - 100)
        new = Row("V1", "video", 0, rng.randint(1, 100), src_in, FILES[source], sound="A1")
        out.insert(rng.randint(0, len(out)), new)
    return out


def lay_out(rng: random.Random, rows: list[Row], clips: list[Row]) -> list[Row]:
    """``clips`` on V1 in order, each after a random gap (zero most of the time, as a ripple
    edit leaves it), then the outro where the content ends."""
    out: list[Row] = []
    cursor = 0
    for clip in clips:
        cursor += rng.choice([0, 0, 0, rng.randint(1, 40)])
        out.append(replace(clip, record_in=cursor, widen_in=0, widen_out=0))
        cursor += clip.duration
    out += [replace(r, record_in=cursor) for r in rows if r.role == "outro"]
    return out


@pytest.mark.parametrize("seed", range(150))
def test_random_edits_import_frame_exact(seed: int) -> None:
    rng = random.Random(seed)
    plan = base_plan(rng)
    rows = rows_from_layout(build_layout(plan, allow_unknown_timecode=True))
    clips = content(rows)
    for _ in range(rng.randint(1, 5)):
        clips = edit(rng, clips)
    edited = lay_out(rng, rows, clips)
    rate = plan.fps.num / plan.fps.den
    timeline, media = build_timeline(edited, rate=rate)
    result = import_timeline(plan, timeline, media, profile=permissive())

    # the edited plan, laid out by S18's own rules, is the edited V1, frame for frame
    layout = build_layout(result.plan, allow_unknown_timecode=True)
    paths = {ref.src: ref.windows_path for ref in layout.media.values()}
    planned = sorted(
        (c.record_in, c.record_out, same_file(paths[c.src]), c.src_in, c.src_out)
        for c in layout.clips
        if c.lane == 0
    )
    on_v1 = sorted(
        (r.record_in, r.record_out, same_file(r.path or ""), r.src_in, (r.src_in or 0) + r.duration)
        for r in edited
        if r.track == "V1"
    )
    assert planned == on_v1
    diff = result.diff
    assert diff["not_carried"] == [] and diff["foreign"] == []
    assert diff["lint_errors"] is False
    # every V1 item is accounted for once: as a piece of an original segment or as added
    pieces = sorted(n["index"] for row in diff["segments"] for n in row["new"])
    assert sorted(pieces + [a["index"] for a in diff["added"]]) == list(range(len(clips)))
    # the plan reloads from its own JSON
    assert EditPlan.model_validate_json(result.plan.model_dump_json(exclude_none=True)) == (
        result.plan
    )
