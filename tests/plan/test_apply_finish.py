"""Phase B (``compile_finish``): the finishing actions, item-id resolution by ``(track,
record_in)``, the grouped steps, and the unmatched / doubly-matched refusals."""

from __future__ import annotations

import pytest
from apply_support import synth_phase_a
from plan_support import FIXTURES

from aar.plan.apply import CompileError, compile_build, compile_finish
from aar.plan.model import EditPlan

LONGFORM = EditPlan.model_validate_json(
    (FIXTURES / "ccn_longform_v2.plan.json").read_text(encoding="utf-8")
)


def test_unresolved_intents_cover_every_finishing_action() -> None:
    finish = compile_finish(LONGFORM)
    assert not finish.resolved
    assert finish.steps == []
    kinds = {i.kind for i in finish.intents}
    assert {
        "colour",
        "transform",
        "audio_crossfade",
        "outro_transition",
        "head_fade",
        "end_fade",
        "audio_gain",
        "title",
        "sting",
        "marker",
    } <= kinds
    assert all(not i.resolved for i in finish.intents if i.kind not in ("title", "sting", "marker"))


def test_resolved_finish_emits_grouped_steps_and_all_intents_resolve() -> None:
    build = compile_build(LONGFORM)
    finish = compile_finish(LONGFORM, synth_phase_a(build))
    assert finish.resolved
    assert all(i.resolved for i in finish.intents)
    ops = [s.op for s in finish.steps]
    assert ops == [
        "transition.add",
        "item.set_color",
        "item.set_transform",
        "item.set_audio_gain",
        "item.set_fades",
        "title.place",
        "markers.add",
    ]
    assert all(step.phase == "B" for step in finish.steps)


def test_colour_target_carries_the_resolved_video_item_id() -> None:
    build = compile_build(LONGFORM)
    finish = compile_finish(LONGFORM, synth_phase_a(build))
    color_step = next(s for s in finish.steps if s.op == "item.set_color")
    item_ids = {t["item"]["item_id"] for t in color_step.params["targets"]}
    # Segments 0 and 2 carry colour, appended at record 0 and 144 on track 1.
    assert item_ids == {"V1:0", "V1:144"}
    for target in color_step.params["targets"]:
        assert target["item"]["track_type"] == "video"


def test_transition_targets_video_and_audio_items() -> None:
    build = compile_build(LONGFORM)
    finish = compile_finish(LONGFORM, synth_phase_a(build))
    tx = next(s for s in finish.steps if s.op == "transition.add")
    cats = {(t["category"], t["item"]["track_type"]) for t in tx.params["targets"]}
    assert ("audio", "audio") in cats  # the audio crossfade
    assert ("simple", "video") in cats  # the outro dissolve


def test_outro_transition_is_a_left_aligned_dissolve() -> None:
    build = compile_build(LONGFORM)
    finish = compile_finish(LONGFORM, synth_phase_a(build))
    tx = next(s for s in finish.steps if s.op == "transition.add")
    simple = [t for t in tx.params["targets"] if t["category"] == "simple"]
    assert len(simple) == 1  # the outro dissolve
    assert simple[0]["type"] == "Cross Dissolve"
    assert simple[0]["alignment"] == "left"
    assert simple[0]["frames"] == 15


def _raw_plan(**extra: object) -> EditPlan:
    base = {
        "version": 2,
        "project": "P",
        "timeline": "T",
        "template": "Tpl",
        "kind": "longform",
        "fps": {"num": 24, "den": 1},
        "resolution": {"width": 1920, "height": 1080},
        "profile": "ccn",
        "sources": [{"id": "a", "media": "pool-a"}],
        "segments": [
            {"track": 1, "source": "a", "src_in": 0, "src_out": 48},
            {"track": 1, "source": "a", "src_in": 48, "src_out": 96},
        ],
    }
    base.update(extra)
    return EditPlan.model_validate(base)


def test_content_dissolve_is_centre_aligned() -> None:
    # Content video dissolves are lint-forbidden, so this exercises compile_finish directly.
    plan = _raw_plan(transitions=[{"after": 0, "video": "dissolve", "video_frames": 12}])
    build = compile_build(plan)
    finish = compile_finish(plan, synth_phase_a(build))
    tx = next(s for s in finish.steps if s.op == "transition.add")
    target = next(t for t in tx.params["targets"] if t["category"] == "simple")
    assert target["type"] == "Cross Dissolve"
    assert target["alignment"] == "center"
    assert target["frames"] == 12


def test_transform_keeps_pan_and_tilt_on_their_own_axes() -> None:
    plan = _raw_plan(
        segments=[
            {
                "track": 1,
                "source": "a",
                "src_in": 0,
                "src_out": 48,
                "transform": {"zoom": 1.2, "pan": 40.0, "tilt": -15.0},
            },
        ],
    )
    build = compile_build(plan)
    finish = compile_finish(plan, synth_phase_a(build))
    tf = next(s for s in finish.steps if s.op == "item.set_transform")
    target = tf.params["targets"][0]
    assert target["pan"] == 40.0 and target["tilt"] == -15.0


def test_segment_unmatched_is_a_typed_error() -> None:
    build = compile_build(LONGFORM)
    results = synth_phase_a(build)
    # Drop the range that carries segment 0 (record 0 on track 1).
    kept = [r for r in results[0].ranges if r.planned_record_in != 0]
    results = [results[0].model_copy(update={"ranges": kept})]
    with pytest.raises(CompileError) as exc:
        compile_finish(LONGFORM, results)
    assert exc.value.code == "segment_unmatched"


def test_segment_double_matched_is_a_typed_error() -> None:
    build = compile_build(LONGFORM)
    results = synth_phase_a(build)
    # Duplicate the range for record 0: two items now claim the same slot.
    doubled = next(r for r in results[0].ranges if r.planned_record_in == 0)
    results = [results[0].model_copy(update={"ranges": [*results[0].ranges, doubled]})]
    with pytest.raises(CompileError) as exc:
        compile_finish(LONGFORM, results)
    assert exc.value.code == "segment_double_matched"


def test_empty_phase_a_result_refused() -> None:
    build = compile_build(LONGFORM)
    results = synth_phase_a(build)
    results = [results[0].model_copy(update={"ranges": []})]
    with pytest.raises(CompileError) as exc:
        compile_finish(LONGFORM, results)
    # No ranges at all -> no video item matches the first colour target.
    assert exc.value.code == "segment_unmatched"


def test_one_item_timeline_merges_head_and_end_fades_on_one_target() -> None:
    plan = EditPlan.model_validate(
        {
            "version": 2,
            "project": "P",
            "timeline": "T",
            "template": "Tpl",
            "kind": "short",
            "fps": {"num": 24, "den": 1},
            "resolution": {"width": 1080, "height": 1920},
            "profile": "ccn",
            "sources": [{"id": "a", "media": "pool-a"}],
            "segments": [{"track": 1, "source": "a", "src_in": 0, "src_out": 120}],
            "head": {"video": "cut", "audio_fade_frames": 6},
            "end_fade_frames": 1,
        }
    )
    build = compile_build(plan)
    finish = compile_finish(plan, synth_phase_a(build))
    fades = next(s for s in finish.steps if s.op == "item.set_fades")
    # The single audio item carries both fades, as one target (item.set_fades refuses a repeat).
    assert len(fades.params["targets"]) == 1
    target = fades.params["targets"][0]
    assert target["fade_in"] == 6 and target["fade_out"] == 1


def test_input_color_space_is_a_note_not_a_colour_set() -> None:
    data = LONGFORM.model_dump(mode="json")
    data["sources"][1]["color"] = {"input_color_space": "Rec.709 Gamma 2.4"}
    plan = EditPlan.model_validate(data)
    finish = compile_finish(plan)
    assert any("input_color_space is not compiled" in n for n in finish.notes)
    # camB has only input_color_space (no cdl/lut), so it adds no colour action.
    assert not any(i.kind == "colour" and "track 1 @72" in i.target for i in finish.intents)
