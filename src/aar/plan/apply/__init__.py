"""``aar.plan.apply``: compile a linted EditPlan v2 into Resolve build-op steps, in three phases.

A pure, cold, hub-side compiler. It never talks to Resolve. Each phase compiles for submission
only from the real result files of the phases before it, so a submission never carries a
placeholder (:mod:`aar.plan.apply.refs`). Submission hands one phase to the hub job API as an
``ops.sequence``; the recipe runner (S13l) drives the three in order.

* :func:`~aar.plan.apply.compile.compile_prepare` -- branch, import, find title templates.
* :func:`~aar.plan.apply.compile.compile_assemble` -- tracks and appends, from the prepare result.
* :func:`~aar.plan.apply.compile.compile_finish` -- finishing ops, from prepare and assemble.
"""

from __future__ import annotations

from aar.plan.apply.compile import (
    DEFAULT_BATCH,
    MAX_BATCH,
    MAX_STEPS,
    TEMPLATE_MAX_ITEMS,
    Compilation,
    compile_assemble,
    compile_finish,
    compile_prepare,
    plan_hash,
    template_item_count,
    title_track,
    video_tracks,
)
from aar.plan.apply.errors import CompileError
from aar.plan.apply.inputs import AssembleResult, PrepareResult
from aar.plan.apply.model import FinishIntent, SegmentPlacement, Step
from aar.plan.apply.submit import Submission, build_submission

__all__ = [
    "DEFAULT_BATCH",
    "MAX_BATCH",
    "MAX_STEPS",
    "TEMPLATE_MAX_ITEMS",
    "AssembleResult",
    "Compilation",
    "CompileError",
    "FinishIntent",
    "PrepareResult",
    "SegmentPlacement",
    "Step",
    "Submission",
    "build_submission",
    "compile_assemble",
    "compile_finish",
    "compile_prepare",
    "plan_hash",
    "template_item_count",
    "title_track",
    "video_tracks",
]
