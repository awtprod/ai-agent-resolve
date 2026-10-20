"""REPORT.md is a pure function of the log: a golden test over a fixed-timestamp run."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict

from aar.hub.events import Actor, EventLog
from aar.hub.registry import Registry
from aar.hub.writes import RecipeData
from aar.recipes import registry
from aar.recipes.models import RecipeDef, StepDef
from aar.recipes.report import build_report_text

A = Actor("agent", "ed")

GOLDEN = """# Recipe report: gold v1

- recipe_node: nd_rec
- status: done
- review rounds: used 1 of 3

## Steps
### c (cold)
- round 1 attempt 1: done 1.500s

### report (cold)
- round 1 attempt 1: done 0.600s

## Gates
- (no decided gates)

## QA
- (no QA checks)

## Facts
- qa.lufs.master = {"unit":"LUFS","value":-14.5} (by agent:ed)

## Artifacts
- aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa (produced by c attempt 1)
- bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb (produced by report attempt 1)
"""


class _Params(BaseModel):
    model_config = ConfigDict(extra="forbid")


@pytest.fixture
def gold_recipe() -> Iterator[None]:
    registry.clear()
    registry.register_callable("rectest.noop", lambda ctx: None)
    registry.register(
        RecipeDef(
            name="gold",
            version=1,
            params_model=_Params,
            steps=(
                StepDef(id="c", kind="cold", callable_id="rectest.noop", idempotent=True),
                StepDef(
                    id="report",
                    kind="cold",
                    callable_id="recipe.report",
                    idempotent=True,
                    needs=("c",),
                ),
            ),
        )
    )
    yield
    registry.clear()


def _build(log: EventLog) -> None:
    Registry(log).create_project("acme", Actor("human", "op-1"), project_id="prj_acme")

    def ap(etype: str, payload: object, ts: str) -> None:
        log.append(etype, A, payload, project_id="prj_acme", ts=ts)

    rec = RecipeData(recipe="gold", brief_id="nd_b", params={}, max_review_rounds=3).dump()
    sha_a, sha_b = "a" * 64, "b" * 64
    ap(
        "graph.node_added",
        {"node_id": "nd_rec", "kind": "recipe", "status": "pending", "title": "r", "data": rec},
        "2026-01-01T00:00:00+00:00",
    )
    ap(
        "graph.node_status_changed",
        {"node_id": "nd_rec", "to": "active"},
        "2026-01-01T00:00:01+00:00",
    )
    _step(
        ap,
        "nd_c",
        "c",
        "2026-01-01T00:00:02+00:00",
        "2026-01-01T00:00:03.500000+00:00",
        {"cold": True},
        sha_a,
    )
    ap(
        "facts.recorded",
        {
            "key": "qa.lufs.master",
            "value": {"value": -14.5, "unit": "LUFS"},
            "source": {"recipe_node": "nd_rec"},
        },
        "2026-01-01T00:00:03.600000+00:00",
    )
    _step(
        ap,
        "nd_rp",
        "report",
        "2026-01-01T00:00:04+00:00",
        "2026-01-01T00:00:04.600000+00:00",
        {"report_sha256": sha_b},
        sha_b,
    )
    ap(
        "graph.node_status_changed",
        {"node_id": "nd_rec", "to": "done"},
        "2026-01-01T00:00:04.700000+00:00",
    )


def _step(ap, op_id, step_id, created, finished, outputs, sha) -> None:  # type: ignore[no-untyped-def]
    ap(
        "graph.node_added",
        {
            "node_id": op_id,
            "kind": "op",
            "status": "pending",
            "title": step_id,
            "data": {
                "schema": "aar.step/1",
                "recipe_node": "nd_rec",
                "step_id": step_id,
                "attempt": 1,
                "round": 1,
                "step_kind": "cold",
            },
        },
        created,
    )
    ap("graph.node_status_changed", {"node_id": op_id, "to": "active"}, created)
    ap("graph.node_status_changed", {"node_id": op_id, "to": "done"}, finished)
    ap(
        "recipe.result",
        {"op_node": op_id, "outcome": "done", "outputs": outputs, "artifact_ids": [sha]},
        finished,
    )


def test_report_is_a_deterministic_golden(tmp_path: Path, gold_recipe: None) -> None:
    log = EventLog(tmp_path / "g.db")
    try:
        _build(log)
        text = build_report_text(log, "nd_rec")
        assert text == GOLDEN
        # Deterministic: the same log renders the same bytes.
        assert build_report_text(log, "nd_rec") == text
    finally:
        log.close()


def test_report_lists_failed_attempts(tmp_path: Path, gold_recipe: None) -> None:
    """The report shows every attempt, including failed ones with their error code."""
    log = EventLog(tmp_path / "f.db")
    try:
        Registry(log).create_project("acme", Actor("human", "op-1"), project_id="prj_acme")

        def ap(etype: str, payload: object, ts: str) -> None:
            log.append(etype, A, payload, project_id="prj_acme", ts=ts)

        rec = RecipeData(recipe="gold", brief_id="nd_b", params={}, max_review_rounds=3).dump()
        ap(
            "graph.node_added",
            {"node_id": "nd_rec", "kind": "recipe", "status": "pending", "title": "r", "data": rec},
            "2026-01-01T00:00:00+00:00",
        )
        ap(
            "graph.node_status_changed",
            {"node_id": "nd_rec", "to": "active"},
            "2026-01-01T00:00:01+00:00",
        )
        ap(
            "graph.node_added",
            {
                "node_id": "nd_c",
                "kind": "op",
                "status": "pending",
                "title": "c",
                "data": {
                    "schema": "aar.step/1",
                    "recipe_node": "nd_rec",
                    "step_id": "c",
                    "attempt": 1,
                    "round": 1,
                    "step_kind": "cold",
                },
            },
            "2026-01-01T00:00:02+00:00",
        )
        ap(
            "graph.node_status_changed",
            {"node_id": "nd_c", "to": "active"},
            "2026-01-01T00:00:02+00:00",
        )
        ap(
            "graph.node_status_changed",
            {"node_id": "nd_c", "to": "failed"},
            "2026-01-01T00:00:03+00:00",
        )
        ap(
            "recipe.result",
            {
                "op_node": "nd_c",
                "outcome": "failed",
                "outputs": {},
                "artifact_ids": [],
                "error": {"code": "boom", "message": "it broke", "fix_hint": ""},
            },
            "2026-01-01T00:00:03+00:00",
        )
        text = build_report_text(log, "nd_rec")
        assert "round 1 attempt 1: failed [boom]" in text
    finally:
        log.close()
