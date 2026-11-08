"""Shared helpers for the S13g2 recipe-runner tests.

Defines the test recipe (registered only here, never in the production registry), which exercises
every step kind: a cold step, a hot step through the fake dispatcher, an agent plan step, a gate
whose ``on_reject`` goes back to the plan step, a cold QA-gate step that runs ``qa_gate`` on a
stored report fixture, and the built-in ``recipe.report``.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from pydantic import BaseModel, ConfigDict
from write_support import AGENT, brief_data

from aar.hub.artifacts import ArtifactStore
from aar.hub.events import Actor
from aar.hub.writes import create_brief, create_recipe
from aar.qa.report import QACheck, QAReport, build_report
from aar.recipes import registry
from aar.recipes.context import StepContext, StepFailure, StepOutput
from aar.recipes.gates import qa_gate
from aar.recipes.models import PlanExpect, RecipeDef, StepDef
from aar.recipes.ports import FakeHotDispatcher, HotJobView, HotResult

RECIPE_NAME = "rectest"


class RecTestParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    qa_report: str  # sha256 of a stored QAReport artifact
    max_seconds: int = 37


def _cold(ctx: StepContext) -> StepOutput:
    ctx.record_fact("qa.lufs.master", {"value": -14.5, "unit": "LUFS"})
    aid = ctx.put(b"cold output bytes", mime="text/plain")
    return StepOutput(
        outputs={"cold": True, "seconds": ctx.params.max_seconds}, artifact_ids=(aid,)
    )


def _qa(ctx: StepContext) -> StepOutput:
    raw = ctx.get(ctx.params.qa_report)
    report = QAReport.model_validate_json(raw)
    gate = qa_gate(report, required=("decode", "loudness"), allow_warn=frozenset({"clipping"}))
    if not gate.passed:
        raise StepFailure("qa_failed", "the QA gate did not pass", "Fix the render and re-run QA.")
    return StepOutput(outputs={"qa": gate.to_dict()})


def test_recipe_def() -> RecipeDef:
    return RecipeDef(
        name=RECIPE_NAME,
        version=1,
        params_model=RecTestParams,
        steps=(
            StepDef(id="c", kind="cold", callable_id="rectest.cold", idempotent=True),
            StepDef(id="h", kind="hot", hot_kind="rectest.hot"),
            StepDef(
                id="p",
                kind="agent",
                instruction="Submit an EditPlan for the short.",
                expects=PlanExpect(),
            ),
            StepDef(id="g", kind="gate", subject="p", on_reject="p", needs=("p",)),
            StepDef(id="q", kind="cold", callable_id="rectest.qa", needs=("g",)),
            StepDef(
                id="report",
                kind="cold",
                callable_id="recipe.report",
                idempotent=True,
                needs=("c", "h", "p", "g", "q"),
            ),
        ),
    )


def register_test_recipe() -> RecipeDef:
    """Register the test recipe and its callables; return the definition (see conftest)."""
    registry.clear()
    registry.register_callable("rectest.cold", _cold)
    registry.register_callable("rectest.qa", _qa)
    return registry.register(test_recipe_def())


def passing_qa_report() -> QAReport:
    """A QAReport whose required checks pass (decode, loudness)."""
    checks = [
        QACheck(id="decode", status="pass", message="decoded"),
        QACheck(id="loudness", status="pass", message="-14.5 LUFS"),
        QACheck(id="clipping", status="warn", message="one near-clip"),
    ]
    return build_report(file="short.mp4", kind="short", profile="ccn", media=None, checks=checks)


def store_qa_report(store: ArtifactStore, report: QAReport, actor: Actor) -> str:
    return store.put(
        report.model_dump_json().encode("utf-8"),
        actor=actor,
        mime="application/json",
    )


def hot_fake() -> FakeHotDispatcher:
    """A fake dispatcher whose ``rectest.hot`` kind completes immediately with outputs."""
    fake = FakeHotDispatcher()
    fake.set_behaviour(
        "rectest.hot",
        lambda params: HotJobView("done", result=HotResult(outputs={"hot": True})),
    )
    return fake


class FixedClock:
    """A controllable clock for lease/round tests."""

    def __init__(self, start: datetime | None = None) -> None:
        self.now = start or datetime(2026, 1, 1, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now

    def tick(self, seconds: float) -> None:
        self.now = self.now + timedelta(seconds=seconds)


def make_run(
    hub: Any,
    *,
    actor: Actor = AGENT,
    qa_report_sha: str | None = None,
    max_review_rounds: int | None = None,
) -> str:
    """Create a brief and a recipe for the test recipe; return the recipe node id."""
    if qa_report_sha is None:
        qa_report_sha = store_qa_report(hub.store, passing_qa_report(), actor)
    brief = create_brief(
        hub.log,
        brief_data(hub.project_id, recipe=RECIPE_NAME),
        actor,
        idempotency_key="brief-rectest",
    )
    brief_id = brief.body["node"]["id"]
    body: dict[str, Any] = {
        "recipe": RECIPE_NAME,
        "brief_id": brief_id,
        "params": {"qa_report": qa_report_sha},
    }
    if max_review_rounds is not None:
        body["max_review_rounds"] = max_review_rounds
    recipe = create_recipe(
        hub.log,
        hub.project_id,
        body,
        actor,
        idempotency_key="recipe-rectest",
        params_validator=registry.registry_params_validator,
    )
    return recipe.body["node"]["id"]


def submit_test_plan(hub: Any, recipe_node: str, actor: Actor, key: str) -> str:
    """Submit the clean short plan deriving from the recipe node; return the plan node id."""
    from write_support import plan_fixture

    from aar.hub.writes import submit_plan

    result = submit_plan(
        hub.log,
        hub.store,
        hub.project_id,
        plan_fixture("short_clean"),
        actor,
        derives_from=recipe_node,
        idempotency_key=key,
    )
    return result.body["node"]["id"]
