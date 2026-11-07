"""Round-2 validator probes: the 64 KiB bound (nested, unicode, failure paths) and the worker's
survival against bad dispatchers and bad step outputs.

Each test asserts the SAFE property; a failure is a finding. Run with -s for OBS lines.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections.abc import Iterator
from typing import Any

import pytest
from pydantic import BaseModel
from recipe_support import make_run
from val_support import worker_idle
from write_support import AGENT, brief_data

from aar.hub.writes import WriteError, create_brief, create_recipe
from aar.recipes import registry
from aar.recipes.context import StepOutput
from aar.recipes.engine import advance
from aar.recipes.models import JsonExpect, RecipeDef, StepDef
from aar.recipes.ports import FakeHotDispatcher, HotJobView, HotResult
from aar.recipes.run import recipe_start, step_complete
from aar.recipes.state import load_state
from aar.recipes.status import status_view
from aar.recipes.worker import run_worker


def _obs(label: str, value: Any) -> None:
    print(f"\nOBS {label}: {json.dumps(value, default=str, sort_keys=True)[:3000]}")


def _nested(depth: int) -> Any:
    v: Any = 1
    for _ in range(depth):
        v = [v]
    return v


class AnyJ(BaseModel):
    d: Any = None
    s: str = ""


class P(BaseModel):
    pass


COLD_OUT: dict[str, Any] = {}


def _cold(ctx: Any) -> StepOutput:
    return StepOutput(outputs=COLD_OUT["value"])


@pytest.fixture
def bound_recipe(hub: Any) -> Iterator[str]:
    registry.clear()
    registry.register_callable("val.bcold", _cold)
    registry.register(
        RecipeDef(
            name="valbound",
            version=1,
            params_model=P,
            steps=(
                StepDef(id="j", kind="agent", instruction="json", expects=JsonExpect(AnyJ)),
                StepDef(id="c", kind="cold", callable_id="val.bcold", needs=("j",), max_attempts=1),
            ),
        )
    )
    brief = create_brief(
        hub.log, brief_data(hub.project_id, recipe="valbound"), AGENT, idempotency_key="bb"
    )
    rn = create_recipe(
        hub.log,
        hub.project_id,
        {"recipe": "valbound", "brief_id": brief.body["node"]["id"], "params": {}},
        AGENT,
        idempotency_key="rb",
        params_validator=registry.registry_params_validator,
    ).body["node"]["id"]
    recipe_start(hub.log, rn, AGENT, idempotency_key="sb")
    yield rn
    registry.clear()


def _complete(hub: Any, rn: str, outputs: Any, key: str) -> str:
    try:
        step_complete(hub.log, hub.store, rn, "j", None, outputs, AGENT, idempotency_key=key)
        return "accepted"
    except WriteError as exc:
        return exc.code
    except Exception as exc:
        return f"raised {type(exc).__name__}: {str(exc)[:80]}"


def _stored_result_sizes(hub: Any) -> list[int]:
    con = sqlite3.connect(str(hub.root / "hub.db"))
    try:
        rows = con.execute(
            "SELECT length(CAST(payload AS BLOB)) FROM events WHERE type='recipe.result'"
        ).fetchall()
    finally:
        con.close()
    return [r[0] for r in rows]


# -- 64 KiB bound on the agent path ---------------------------------------------------------


def test_unicode_heavy_agent_outputs(hub: Any, bound_recipe: str) -> None:
    rn = bound_recipe
    over = _complete(hub, rn, {"json": {"s": "é" * (33 * 1024)}}, "u1")  # ~198 KiB escaped
    # R3-1 stored-bytes rule: a surrogate pair escapes to 12 bytes, so 16k emoji is ~192 KiB stored
    # and over the 64 KiB bound. A small ASCII output is accepted and stored under 64 KiB.
    emoji = "\U0001f600" * 16000
    big_emoji = _complete(hub, rn, {"json": {"s": emoji}}, "u2")
    ok = _complete(hub, rn, {"json": {"s": "a" * 1024}}, "u3")
    sizes = _stored_result_sizes(hub)
    view = status_view(hub.log, rn)
    _obs(
        "unicode",
        {
            "66KiB_e_acute": over,
            "16k_emoji": big_emoji,
            "small_ascii": ok,
            "stored recipe.result payload bytes": sizes,
            "status view bytes": len(json.dumps(view)),
        },
    )
    # Both non-ASCII heavy outputs exceed the bound measured on the stored (escaped) JSON.
    assert over == "outputs_too_large"
    assert big_emoji == "outputs_too_large"
    assert ok == "accepted"
    # Every stored recipe.result payload is within the 64 KiB stored-bytes bound.
    assert sizes and max(sizes) <= 64 * 1024, f"stored recipe.result sizes {sizes}"


@pytest.mark.parametrize("depth", [900, 5000, 50000])
def test_deeply_nested_agent_outputs(hub: Any, bound_recipe: str, depth: int) -> None:
    got = _complete(hub, bound_recipe, {"json": {"d": _nested(depth)}}, f"n{depth}")
    try:
        st = load_state(hub.log, bound_recipe)
        fold = st.latest_in_round("j", 1).status
    except BaseException as exc:
        fold = f"load_state raised {type(exc).__name__}"
    _obs(f"nested agent outputs depth {depth}", {"step_complete": got, "fold": fold})
    assert got in ("accepted", "outputs_too_large", "outputs_invalid", "expects_unmet"), got
    assert not fold.startswith("load_state raised"), fold


# -- the cold path: bad outputs must fail the step, never crash the worker -------------------


@pytest.mark.parametrize(
    "label,value",
    [
        ("200KiB", {"x": "y" * (200 * 1024)}),
        ("nested5000", {"d": _nested(5000)}),
        ("nested50000", {"d": _nested(50000)}),
        ("neg_inf_lufs", {"lufs": float("-inf")}),
        ("nan", {"x": float("nan")}),
        ("nonstr_keys", {1: "a", "b": 2}),
    ],
)
def test_cold_bad_outputs_fail_the_step_not_the_worker(
    hub: Any, bound_recipe: str, label: str, value: Any
) -> None:
    rn = bound_recipe
    _complete(hub, rn, {"json": {"s": "ok"}}, "ok")
    COLD_OUT["value"] = value
    try:
        n = run_worker(hub.log, hub.store, AGENT, once=True)
        worker = f"ran {n}"
    except BaseException as exc:
        worker = f"raised {type(exc).__name__}: {str(exc)[:120]}"
    st = load_state(hub.log, rn)
    c = st.latest_in_round("c", 1)
    _obs(
        f"cold outputs {label}",
        {
            "worker": worker,
            "c": [c.status, (c.result_error or {}).get("code")],
            "stored": _stored_result_sizes(hub),
        },
    )
    assert worker.startswith("ran"), f"the worker loop died: {worker}"
    if label != "nested5000":
        assert c.status == "failed"


# -- hot path: failure payloads and the status view ----------------------------------------


def _hot_run(hub: Any, registered: Any, view: HotJobView) -> tuple[str, FakeHotDispatcher]:
    fake = FakeHotDispatcher()
    fake.set_behaviour("rectest.hot", lambda p: view)
    rn = make_run(hub)
    recipe_start(hub.log, rn, AGENT, idempotency_key="s", dispatcher=fake)
    worker_idle(hub, fake)
    return rn, fake


def test_hot_failed_error_is_bounded(hub: Any, registered_recipe: Any) -> None:
    big = {
        "code": "c" * 300_000,
        "message": "m" * 300_000,
        "fix_hint": "f" * 300_000,
        "extra": {"k": "v" * 300_000},
    }
    rn, fake = _hot_run(hub, registered_recipe, HotJobView("failed", error=big))
    h = load_state(hub.log, rn).latest_in_round("h", 1)
    view = status_view(hub.log, rn, dispatcher=fake)
    hstep = next(s for s in view["steps"] if s["step_id"] == "h")
    _obs(
        "hot failed with a 1.2 MB error",
        {
            "h": h.status,
            "stored": _stored_result_sizes(hub),
            "view_bytes": len(json.dumps(view)),
            "h_error_bytes": len(json.dumps(hstep["error"])),
            "truncated": view["truncated"],
        },
    )
    assert max(_stored_result_sizes(hub)) <= 64 * 1024, "a hot failure stored an unbounded error"


def test_status_view_per_step_summary_bound_with_one_huge_key(
    hub: Any, registered_recipe: Any
) -> None:
    out = {"k" * 60_000: 1}
    rn, fake = _hot_run(hub, registered_recipe, HotJobView("done", result=HotResult(outputs=out)))
    view = status_view(hub.log, rn, dispatcher=fake)
    hstep = next(s for s in view["steps"] if s["step_id"] == "h")
    size = len(json.dumps(hstep["outputs"], ensure_ascii=False))
    _obs(
        "one 60 KB key",
        {
            "h": hstep["state"],
            "outputs_summary_bytes": size,
            "view_bytes": len(json.dumps(view)),
            "truncated": view["truncated"],
        },
    )
    assert size <= 1024, f"per-step outputs summary is {size} bytes (brief: at most 1 KiB)"


# -- dispatchers that raise, hang or return garbage ----------------------------------------


class _Garbage:
    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.submits = 0

    def submit(self, **kw: Any) -> Any:
        self.submits += 1
        if self.mode == "submit_raises":
            raise ConnectionError("node down")
        if self.mode == "submit_object_ref":
            return object()
        if self.mode == "submit_hangs":
            time.sleep(3)
        return "job-1"

    def poll(self, job_ref: Any) -> Any:
        if self.mode == "poll_raises":
            raise TimeoutError("poll")
        if self.mode == "poll_none":
            return None
        if self.mode == "poll_bad_outputs":
            return HotJobView("done", result=HotResult(outputs="not a dict"))  # type: ignore[arg-type]
        if self.mode == "poll_bad_state":
            return HotJobView("exploded")
        return HotJobView("running")


@pytest.mark.parametrize(
    "mode",
    [
        "submit_raises",
        "submit_object_ref",
        "poll_raises",
        "poll_none",
        "poll_bad_outputs",
        "poll_bad_state",
    ],
)
def test_worker_survives_a_bad_dispatcher(hub: Any, registered_recipe: Any, mode: str) -> None:
    disp = _Garbage(mode)
    rn = make_run(hub)
    errors = []
    try:
        recipe_start(hub.log, rn, AGENT, idempotency_key="s", dispatcher=disp)
        start = "ok"
    except BaseException as exc:
        start = f"raised {type(exc).__name__}"
    for _ in range(4):
        try:
            run_worker(hub.log, hub.store, AGENT, dispatcher=disp, once=True)
        except BaseException as exc:
            errors.append(type(exc).__name__)
    try:
        view = status_view(hub.log, rn, dispatcher=disp)
        hstate = next(s["state"] for s in view["steps"] if s["step_id"] == "h")
        cstate = next(s["state"] for s in view["steps"] if s["step_id"] == "c")
    except BaseException as exc:
        hstate, cstate = f"status_view raised {type(exc).__name__}", "?"
    _obs(
        f"dispatcher {mode}",
        {
            "recipe_start": start,
            "worker_errors": errors,
            "h": hstate,
            "c": cstate,
            "submit_calls": disp.submits,
        },
    )
    assert not errors, f"the worker loop raised: {errors}"
    assert cstate == "done", "a bad dispatcher stopped the cold step"
    assert not hstate.startswith("status_view raised"), hstate


def test_hanging_dispatcher_blocks_advance(hub: Any, registered_recipe: Any) -> None:
    """Not a brief requirement: records whether a hanging port is bounded at all."""
    disp = _Garbage("submit_hangs")
    rn = make_run(hub)
    recipe_start(hub.log, rn, AGENT, idempotency_key="s")  # no dispatcher yet
    t0 = time.monotonic()
    th = threading.Thread(
        target=advance, args=(hub.log, rn), kwargs={"actor": AGENT, "dispatcher": disp}
    )
    th.start()
    th.join(10)
    _obs("hanging submit (3 s)", {"advance_seconds": round(time.monotonic() - t0, 2)})
