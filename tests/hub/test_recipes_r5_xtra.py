"""Round-5 slice tests (adopted from the validator): R34 reachability, the 2 KiB instruction cap,
the per-(recipe, type) dedup with first_seen, and the log rate-limit table bound."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel
from write_support import AGENT, brief_data, make_hub

from aar.hub.writes import create_brief, create_recipe
from aar.recipes import registry, worker
from aar.recipes.engine import record_worker_error
from aar.recipes.models import JsonExpect, RecipeDef, RecipeDefError, StepDef
from aar.recipes.run import recipe_start
from aar.recipes.status import MAX_VIEW_BYTES, status_view


class P(BaseModel):
    pass


class J(BaseModel):
    v: int = 0


def _size(view: dict[str, Any]) -> int:
    return len(json.dumps(view, sort_keys=True, separators=(",", ":")).encode())


def _hub_with_long_instructions(root: Path, length: int) -> tuple[Any, str]:
    registry.clear()
    name = f"valwide{length}"
    registry.register(
        RecipeDef(
            name=name,
            version=1,
            params_model=P,
            steps=tuple(
                StepDef(
                    id=f"s{i:02d}", kind="agent", instruction="x" * length, expects=JsonExpect(J)
                )
                for i in range(64)
            ),
        )
    )
    root.mkdir(parents=True, exist_ok=True)
    hub = make_hub(root)
    brief = create_brief(
        hub.log, brief_data(hub.project_id, recipe=name), AGENT, idempotency_key="b"
    )
    rn = create_recipe(
        hub.log,
        hub.project_id,
        {"recipe": name, "brief_id": brief.body["node"]["id"], "params": {}},
        AGENT,
        idempotency_key="r",
        params_validator=registry.registry_params_validator,
    ).body["node"]["id"]
    recipe_start(hub.log, rn, AGENT, idempotency_key="s")
    return hub, rn


def test_r34_entry_drop_is_reachable_and_needed(tmp_path: Path) -> None:
    """Find an instruction length where the view WITHOUT worker-error entries fits 64 KiB but the
    entries push it over: then only the entry drop keeps the bound (the branch R34 removes)."""
    chosen = None
    for length in range(600, 1100, 10):
        hub, rn = _hub_with_long_instructions(tmp_path / f"probe{length}", length)
        s0 = _size(status_view(hub.log, rn))
        hub.log.close()
        if MAX_VIEW_BYTES - 5000 < s0 <= MAX_VIEW_BYTES - 300:
            chosen = (length, s0)
            break
    assert chosen is not None, "no length lands in the window"
    length, s0 = chosen
    hub, rn = _hub_with_long_instructions(tmp_path / "final", length)
    for i in range(5):
        record_worker_error(
            hub.log,
            project_id=hub.project_id,
            recipe_node=rn,
            exc_type=f"Err{i}",
            message=f"Err{i}: " + "m" * 4000,
        )
    view = status_view(hub.log, rn)
    size = _size(view)
    print(
        f"\nOBS R34: instruction length {length}; view without records {s0} B; with 5 records "
        f"{size} B; entries listed {len(view['diagnostics']['worker_errors'])}; "
        f"count {view['diagnostics']['worker_error_count']}"
    )
    assert size <= MAX_VIEW_BYTES
    registry.clear()


def test_long_instructions_break_the_view_bound(tmp_path: Path) -> None:
    """64 steps at the 2 KiB instruction cap still keep the view inside the bound (R5-2)."""
    hub, rn = _hub_with_long_instructions(tmp_path / "wide", 2048)
    size = _size(status_view(hub.log, rn))
    print(f"\nOBS 64 agent steps x 2000-char instructions: view {size} B (bound {MAX_VIEW_BYTES})")
    registry.clear()
    assert size <= MAX_VIEW_BYTES


def test_instruction_over_2kib_is_refused_at_registration() -> None:
    registry.clear()
    big = RecipeDef(
        name="toolong",
        version=1,
        params_model=P,
        steps=(StepDef(id="s", kind="agent", instruction="é" * 1025, expects=JsonExpect(J)),),
    )
    with pytest.raises(RecipeDefError, match="instruction is over 2048 bytes"):
        registry.register(big)
    ok = RecipeDef(
        name="justfits",
        version=1,
        params_model=P,
        steps=(StepDef(id="s", kind="agent", instruction="é" * 1024, expects=JsonExpect(J)),),
    )
    registry.register(ok)
    registry.clear()


def test_dedup_hides_a_second_failure_mode_of_the_same_type(tmp_path: Path) -> None:
    hub = make_hub(tmp_path)
    from recipe_support import make_run, register_test_recipe

    register_test_recipe()
    rn = make_run(hub)
    recipe_start(hub.log, rn, AGENT, idempotency_key="s")
    for msg in ("KeyError: 'qa_report' missing from params", "KeyError: 'brief' gone"):
        worker._note_worker_error(hub.log, hub.project_id, rn, KeyError(msg))
    view = status_view(hub.log, rn)
    entries = view["diagnostics"]["worker_errors"]
    print("\nOBS two KeyErrors with different causes:", json.dumps(view["diagnostics"]))
    assert len(entries) == 1 and "qa_report" in entries[0]["error"]
    # R5-3: the entry says when it first failed (UTC); the dedup rule is unchanged.
    assert datetime.fromisoformat(entries[0]["first_seen"]).utcoffset() == timedelta(0)
    assert view["diagnostics"]["worker_error_count"] == 1
    registry.clear()


def test_rate_limit_table_is_bounded(tmp_path: Path) -> None:
    hub = make_hub(tmp_path)
    worker._log_state.clear()
    for i in range(5000):
        worker._note_worker_error(hub.log, hub.project_id, f"nd_{i:016x}", RuntimeError("x"))
    n = len(worker._log_state)
    print("\nOBS rate-limit table after 5000 distinct recipes:", n, "cap", worker._LOG_STATE_MAX)
    assert n <= worker._LOG_STATE_MAX
