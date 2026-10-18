"""Whitelisted, verified edits of ``.drp``/``.drt`` files (S8, offline part).

::

    from aar.drp.modify import edit_file, parse_plan

    plan = parse_plan(open("plan.json").read())
    outcome = edit_file("in.drp", plan, "out.drp")   # never touches in.drp
    outcome.to_dict()                                # changes, refusals, "unverified" label

Allowed edits: rename (project, timeline, bin, media pool clip, timeline item), path mapping
between Windows and Mac paths, and markers (add, update, delete) on timelines and items. Each
is a splice of a span the reader recorded plus a re-encode of the blob around it; every other
byte is copied. A written file must pass the S2 gate and show exactly the intended changes, or
nothing is written. Outputs are labelled ``unverified`` until the Resolve sandbox round trip
(:mod:`aar.drp.modify.roundtrip` holds its allow-list) can run.
"""

from aar.drp.modify.engine import EditRefused, Expected, Refusal
from aar.drp.modify.plan import DrpEditPlan, PlanProblem, parse_plan
from aar.drp.modify.roundtrip import ROUND_TRIP_ALLOW_LIST, compare_round_trip
from aar.drp.modify.verify import verify_write
from aar.drp.modify.write import EditOutcome, edit_file

__all__ = [
    "ROUND_TRIP_ALLOW_LIST",
    "DrpEditPlan",
    "EditOutcome",
    "EditRefused",
    "Expected",
    "PlanProblem",
    "Refusal",
    "compare_round_trip",
    "edit_file",
    "parse_plan",
    "verify_write",
]
