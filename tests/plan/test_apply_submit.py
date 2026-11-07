"""``build_submission``: the unresolved refusal (F1) and idempotency keys (F2)."""

from __future__ import annotations

import pytest
from apply_builders import longform
from apply_support import synth_assemble, synth_prepare

from aar.plan.apply import (
    CompileError,
    build_submission,
    compile_assemble,
    compile_finish,
    compile_prepare,
)

LF = longform()
PREP = synth_prepare(LF)
ASM = synth_assemble(LF, PREP)
FIN = compile_finish(LF, prepare=PREP, assemble=ASM)


def test_f1_unresolved_finish_refused() -> None:
    comp = compile_finish(LF)
    assert comp.unresolved > 0
    with pytest.raises(CompileError) as ei:
        build_submission(LF, "finish", comp.steps)
    assert ei.value.code == "unresolved_reference"


def test_f1_unresolved_prepare_and_assemble_refused() -> None:
    for comp, phase in ((compile_prepare(LF), "prepare"), (compile_assemble(LF), "assemble")):
        with pytest.raises(CompileError) as ei:
            build_submission(LF, phase, comp.steps)
        assert ei.value.code == "unresolved_reference"


def test_f2_keys_differ_by_phase_and_inputs() -> None:
    steps = FIN.steps
    base = build_submission(LF, "finish", steps, input_hashes=["a", "b"]).idempotency_key
    assert base == build_submission(LF, "finish", steps, input_hashes=["a", "b"]).idempotency_key
    assert base != build_submission(LF, "finish", steps, input_hashes=["a", "c"]).idempotency_key
    assert base != build_submission(LF, "assemble", steps, input_hashes=["a", "b"]).idempotency_key
    assert base != build_submission(LF, "finish", steps).idempotency_key
    assert base.startswith("plan.apply:")


def test_f2_different_assemble_hash_is_a_new_key() -> None:
    k1 = build_submission(LF, "finish", FIN.steps, input_hashes=["prep", "asm1"])
    k2 = build_submission(LF, "finish", FIN.steps, input_hashes=["prep", "asm2"])
    assert k1.idempotency_key != k2.idempotency_key


def test_resolved_submissions_validate() -> None:
    for phase, comp in (
        ("prepare", compile_prepare(LF, template=None)),
        ("assemble", compile_assemble(LF, prepare=PREP)),
        ("finish", FIN),
    ):
        if phase == "prepare":
            continue  # needs a template describe; covered in the CLI tests
        sub = build_submission(LF, phase, comp.steps, input_hashes=["x"])
        assert sub.kind == "ops.sequence"
        assert sub.phase == phase
        assert len(sub.params["steps"]) == len(comp.steps)


def test_submit_validates_each_step_against_its_op_model() -> None:
    # Defence in depth: a step whose params violate the op model is refused at submit, exactly as
    # 'aar hub job submit' would, not queued.
    import pytest
    from pydantic import ValidationError

    from aar.plan.apply.model import Step

    p = longform()
    bad = Step(
        phase="finish",
        op="item.set_audio_gain",
        name="bad",
        params={
            "timeline": {"name": "t", "unique_id": "u"},
            "targets": [
                {"item": {"track_type": "audio", "track": 1, "item_id": "i"}, "gain_db": 999.0}
            ],
        },
        summary="",
    )
    with pytest.raises(ValidationError):  # from the op's params model, inside _validate
        build_submission(p, "finish", [bad], input_hashes=["h"])
