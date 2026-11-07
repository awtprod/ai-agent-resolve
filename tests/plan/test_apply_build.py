"""Phase A (``compile_build``): ordering, batching, the property that every segment is appended
exactly once in record order, and the typed refusals."""

from __future__ import annotations

import pytest
from apply_support import template_describe
from plan_support import FIXTURES

from aar.plan.apply import (
    DEFAULT_BATCH,
    MAX_BATCH,
    TEMPLATE_MAX_ITEMS,
    CompileError,
    compile_build,
)
from aar.plan.apply.compile import template_item_count
from aar.plan.model import EditPlan

LONGFORM = EditPlan.model_validate_json(
    (FIXTURES / "ccn_longform_v2.plan.json").read_text(encoding="utf-8")
)


def _plan(tracks: dict[int, int], *, length: int = 10) -> EditPlan:
    """A raw (not necessarily lint-clean) plan with ``tracks[t]`` segments on video track ``t``."""
    sources = [{"id": "s", "media": "pool-item-s"}]
    segments = []
    for track, count in tracks.items():
        for i in range(count):
            start = i * length
            segments.append(
                {"track": track, "source": "s", "src_in": start, "src_out": start + length}
            )
    return EditPlan.model_validate(
        {
            "version": 2,
            "project": "P",
            "timeline": "T",
            "template": "Tpl",
            "kind": "longform",
            "fps": {"num": 24, "den": 1},
            "resolution": {"width": 1920, "height": 1080},
            "profile": "ccn",
            "sources": sources,
            "segments": segments,
        }
    )


# -- ordering ----------------------------------------------------------------------------------


def test_phase_a_step_order_is_branch_import_tracks_then_appends() -> None:
    build = compile_build(LONGFORM)
    ops = [s.op for s in build.steps]
    assert ops[0] == "timeline.branch"
    assert ops[-1] == "timeline.append_ranges"
    first_append = ops.index("timeline.append_ranges")
    assert ops[:first_append] == [
        "timeline.branch",
        "media.import",
        "media.import",
        "media.import",
        "media.import",
        "track.ensure",
    ]
    assert all(step.phase == "A" for step in build.steps)


def test_every_step_name_is_unique() -> None:
    build = compile_build(LONGFORM)
    names = [s.name for s in build.steps]
    assert len(names) == len(set(names))


# -- the property ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tracks",
    [
        {1: 1},
        {1: 10},
        {1: 23},
        {1: 7, 2: 13},
        {1: 25, 2: 1, 3: 9},
    ],
)
@pytest.mark.parametrize("batch_size", [1, 3, 10])
def test_every_segment_appended_once_in_record_order_within_the_cap(
    tracks: dict[int, int], batch_size: int
) -> None:
    plan = _plan(tracks)
    build = compile_build(plan, batch_size=batch_size)

    # Every segment appears exactly once across the placements.
    placed = sorted(p.segment for p in build.placements)
    assert placed == list(range(len(plan.segments)))

    # And exactly once across the actual append-step ranges.
    appended_labels = []
    for step in build.steps:
        if step.op != "timeline.append_ranges":
            continue
        assert len(step.params["ranges"]) <= batch_size  # batches within the cap
        appended_labels.extend(
            r["label"] for r in step.params["ranges"] if r["label"].startswith("segment ")
        )
    assert sorted(appended_labels) == sorted(f"segment {i}" for i in range(len(plan.segments)))

    # Per track, record_in is strictly increasing (record order).
    per_track: dict[int, list[int]] = {}
    for placement in build.placements:
        per_track.setdefault(placement.track, []).append(placement.record_in)
    for record_ins in per_track.values():
        assert record_ins == sorted(record_ins)
        assert len(record_ins) == len(set(record_ins))


def test_append_steps_named_per_track_and_batch() -> None:
    build = compile_build(_plan({1: 12, 2: 3}), batch_size=10)
    append_names = [s.name for s in build.steps if s.op == "timeline.append_ranges"]
    assert append_names == ["append:v1:0", "append:v1:1", "append:v2:0"]


# -- typed refusals ----------------------------------------------------------------------------


@pytest.mark.parametrize("batch_size", [0, -1, MAX_BATCH + 1, 100])
def test_batch_size_out_of_range_refused(batch_size: int) -> None:
    with pytest.raises(CompileError) as exc:
        compile_build(LONGFORM, batch_size=batch_size)
    assert exc.value.code == "batch_size_invalid"


def test_default_batch_is_ten() -> None:
    assert DEFAULT_BATCH == 10 and MAX_BATCH == 10


def test_missing_template_refused() -> None:
    data = LONGFORM.model_dump(mode="json")
    data["template"] = None
    with pytest.raises(CompileError) as exc:
        compile_build(EditPlan.model_validate(data))
    assert exc.value.code == "missing_template"


def test_template_too_large_refused_above_thirty() -> None:
    with pytest.raises(CompileError) as exc:
        compile_build(LONGFORM, template=template_describe(TEMPLATE_MAX_ITEMS + 1))
    assert exc.value.code == "template_too_large"
    assert exc.value.details["items"] == TEMPLATE_MAX_ITEMS + 1


def test_template_at_the_limit_is_allowed() -> None:
    build = compile_build(LONGFORM, template=template_describe(TEMPLATE_MAX_ITEMS))
    assert build.template_items == TEMPLATE_MAX_ITEMS
    assert template_item_count(template_describe(5)) == 5


def test_unknown_source_refused() -> None:
    data = _plan({1: 1}).model_dump(mode="json")
    data["segments"][0]["source"] = "ghost"
    with pytest.raises(CompileError) as exc:
        compile_build(EditPlan.model_validate(data))
    assert exc.value.code == "unknown_source"


def test_template_describe_supplies_branch_source_unique_id() -> None:
    build = compile_build(LONGFORM, template=template_describe(5, unique_id="tl-abc"))
    branch = next(s for s in build.steps if s.op == "timeline.branch")
    assert branch.params["source"]["unique_id"] == "tl-abc"
    # Without a describe the branch names the template only, with a dry-run note.
    plain = compile_build(LONGFORM)
    branch2 = next(s for s in plain.steps if s.op == "timeline.branch")
    assert branch2.params["source"]["unique_id"] is None
    assert any("template size unchecked" in n for n in plain.notes)
