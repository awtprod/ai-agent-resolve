"""``aar.recipes.ccn``: the ``ccn-longform`` recipe, the Linux-only CCN editing run (S19a).

Loading this module (``AAR_RECIPES=aar.recipes.ccn`` or ``--recipes aar.recipes.ccn``) registers
the recipe and its cold callables. The steps, on the S13g2 engine:

1. ``stub`` (cold, idempotent): the plan stub from the params' sources (``aar.plan.stub``).
2. ``sync`` (cold, idempotent): ``aar.analyze.sync`` of each other source against the first; a
   pair that is not ``ok`` fails the step. One source: nothing to sync.
3. ``transcript`` (cold, idempotent): the first source transcribed (Whisper, normalised audio by
   default), its readable text and report stored; reused through a ``transcript.<key>`` fact
   keyed on the file's sha256, the model and the settings.
4. ``laughs`` (cold, idempotent): ``aar.analyze.laughs`` only when the params ask (a paid API).
5. ``guide`` (cold, idempotent): the cut step's full guide as an artifact (exact commands with the
   hub's own paths, this run's artifact ids, the brief, the house rules, later rounds).
6. ``cut`` (agent): the agent reads the guide, writes the keep list, runs ``aar plan produce``
   and ``aar plan lint`` itself, and submits the plan; the plan gate refuses lint errors.
7. ``render`` (cold, idempotent): the plan checked against the stub, the review render on the
   hub and its QA gate.
8. ``export`` (cold, idempotent): FCPXML with the media map, stored and copied to the PC's share.
9. ``review`` (gate): a human reviews the render; a rejection goes back to ``cut``.
10. ``report``: the built-in ``recipe.report``.
"""

from __future__ import annotations

from aar.recipes import registry
from aar.recipes.ccn.instruction import CUT_INSTRUCTION
from aar.recipes.ccn.params import CcnLongformParams
from aar.recipes.ccn.steps import (
    export_step,
    guide_step,
    laughs_step,
    render_step,
    set_laughs_runner,
    set_transcriber,
    stub_step,
    sync_step,
    transcript_step,
)
from aar.recipes.models import PlanExpect, RecipeDef, StepDef

RECIPE_NAME = "ccn-longform"
HOUR = 3600.0

#: Callable ids, registered under the recipe's own prefix.
CALLABLES = {
    "ccn.stub": stub_step,
    "ccn.sync": sync_step,
    "ccn.transcript": transcript_step,
    "ccn.laughs": laughs_step,
    "ccn.render": render_step,
    "ccn.export": export_step,
    "ccn.guide": guide_step,
}

CCN_LONGFORM = RecipeDef(
    name=RECIPE_NAME,
    version=2,
    params_model=CcnLongformParams,
    steps=(
        # a packet scan of each proxy (seconds for a 90-minute show)
        StepDef(id="stub", kind="cold", callable_id="ccn.stub", idempotent=True, timeout_s=HOUR),
        StepDef(
            id="sync",
            kind="cold",
            callable_id="ccn.sync",
            idempotent=True,
            needs=("stub",),
            timeout_s=2 * HOUR,
        ),
        # 90 minutes of far-field audio took about 25 minutes on the hub (S19 pilot); the decode
        # cap is 4 h of audio
        StepDef(
            id="transcript",
            kind="cold",
            callable_id="ccn.transcript",
            idempotent=True,
            needs=("stub",),
            timeout_s=6 * HOUR,
        ),
        StepDef(
            id="laughs",
            kind="cold",
            callable_id="ccn.laughs",
            idempotent=True,
            needs=("stub",),
            timeout_s=2 * HOUR,
        ),
        StepDef(
            id="guide",
            kind="cold",
            callable_id="ccn.guide",
            idempotent=True,
            needs=("stub", "sync", "transcript", "laughs"),
            timeout_s=HOUR,
        ),
        StepDef(
            id="cut",
            kind="agent",
            needs=("stub", "sync", "transcript", "laughs", "guide"),
            instruction=CUT_INSTRUCTION,
            expects=PlanExpect(),
            timeout_s=24 * HOUR,
        ),
        # about 0.8x realtime of the cut (S19 pilot: 236 s in 189 s), plus QA; idempotent: a
        # re-claimed op starts from emptied scratch and stores content-addressed artifacts
        StepDef(
            id="render",
            kind="cold",
            callable_id="ccn.render",
            idempotent=True,
            needs=("stub", "cut", "transcript", "laughs"),
            timeout_s=4 * HOUR,
        ),
        StepDef(
            id="export",
            kind="cold",
            callable_id="ccn.export",
            idempotent=True,
            needs=("stub", "cut", "render", "transcript", "laughs"),
            timeout_s=HOUR,
        ),
        StepDef(
            id="review",
            kind="gate",
            subject="render",
            on_reject="cut",
            needs=("export",),
            timeout_s=24 * HOUR,
        ),
        StepDef(
            id="report",
            kind="cold",
            callable_id="recipe.report",
            idempotent=True,
            needs=(
                "stub",
                "sync",
                "transcript",
                "laughs",
                "guide",
                "cut",
                "render",
                "export",
                "review",
            ),
            timeout_s=HOUR,
        ),
    ),
)


def register_ccn() -> RecipeDef:
    """Register the recipe and its callables (idempotent; tests call it after a registry clear)."""
    for callable_id, fn in CALLABLES.items():
        registry.register_callable(callable_id, fn)
    return registry.register(CCN_LONGFORM)


register_ccn()

__all__ = [
    "CCN_LONGFORM",
    "RECIPE_NAME",
    "CcnLongformParams",
    "register_ccn",
    "set_laughs_runner",
    "set_transcriber",
]
