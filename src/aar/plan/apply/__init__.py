"""``aar.plan.apply``: compile a linted EditPlan v2 into Resolve build-op steps.

This is a pure, cold, hub-side compiler. It never talks to Resolve. From a plan (and the
describe results of the template and, on a re-apply, the agent branch) it produces the exact
ordered :class:`~aar.plan.apply.model.Step` list that builds the plan, plus a human-readable
dry-run diff. Submission (``--submit``) hands phase A to the existing hub job API as an
``ops.sequence`` job; running it against Resolve and the recipe wiring are S13l.

Two phases, because the build ops address timeline items by ``item_id``, which only exists
after a clip is appended:

* :func:`~aar.plan.apply.compile.compile_build` (phase A): branch the template, import the
  source media, ensure the tracks, append the ranges.
* :func:`~aar.plan.apply.compile.compile_finish` (phase B): transitions, colour, transforms,
  titles, audio gain, fades and markers, each addressing an item resolved from phase A's
  verified append results by ``(track, record_in)``.
"""

from __future__ import annotations

from aar.plan.apply.compile import (
    DEFAULT_BATCH,
    MAX_BATCH,
    TEMPLATE_MAX_ITEMS,
    BranchRef,
    BuildCompilation,
    FinishCompilation,
    compile_build,
    compile_finish,
    plan_hash,
    template_item_count,
)
from aar.plan.apply.errors import CompileError
from aar.plan.apply.model import FinishIntent, SegmentPlacement, Step
from aar.plan.apply.submit import Submission, build_submission

__all__ = [
    "DEFAULT_BATCH",
    "MAX_BATCH",
    "TEMPLATE_MAX_ITEMS",
    "BranchRef",
    "BuildCompilation",
    "CompileError",
    "FinishCompilation",
    "FinishIntent",
    "SegmentPlacement",
    "Step",
    "Submission",
    "build_submission",
    "compile_build",
    "compile_finish",
    "plan_hash",
    "template_item_count",
]
