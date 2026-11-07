"""Validator probes with real OS processes on one SQLite file."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
from write_support import AGENT, brief_data, make_hub

from aar.hub.events import EventLog
from aar.hub.graph import WorkGraph
from aar.hub.writes import create_brief, create_recipe
from aar.recipes import registry
from aar.recipes.state import load_state

PROCMOD = Path(__file__).parent / "procmod"
DRIVER = PROCMOD / "val_driver.py"


def _obs(label: str, value: Any) -> None:
    print(f"\nOBS {label}: {json.dumps(value, default=str, sort_keys=True)}")


def _load_recipes() -> None:
    sys.modules.pop("val_recipes", None)
    if str(PROCMOD) not in sys.path:
        sys.path.insert(0, str(PROCMOD))
    import val_recipes  # noqa: F401


def _make(hub: Any, name: str, *, activate: bool = True) -> str:
    _load_recipes()
    brief = create_brief(
        hub.log, brief_data(hub.project_id, recipe=name), AGENT, idempotency_key=f"b-{name}"
    )
    rn = create_recipe(
        hub.log,
        hub.project_id,
        {"recipe": name, "brief_id": brief.body["node"]["id"], "params": {}},
        AGENT,
        idempotency_key=f"r-{name}",
        params_validator=registry.registry_params_validator,
    ).body["node"]["id"]
    if activate:
        __import__("aar.recipes.run", fromlist=["recipe_start"]).recipe_start(
            hub.log, rn, AGENT, idempotency_key=f"a-{name}"
        )  # A1: start via recipe_start
    return rn


def _spawn(action: str, db: Path, arts: Path, actor: str, rn: str, **env: str) -> Any:
    full = dict(os.environ)
    full.update(env)
    return subprocess.Popen(
        [sys.executable, str(DRIVER), action, str(db), str(arts), actor, rn],
        env=full,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _wait(pred: Any, timeout: float = 30.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.05)
    return False


@pytest.mark.parametrize(
    "actors",
    [
        ("agent:agent-2", "agent:worker-1"),  # two non-creator actors
        ("agent:agent-1", "agent:worker-1"),  # the creator and the worker
        ("agent:agent-2", "human:operator-1"),
    ],
)
def test_two_actor_advance_race(tmp_path: Path, actors: tuple[str, str]) -> None:
    trials = 6
    summary = []
    for t in range(trials):
        root = tmp_path / f"t{t}"
        root.mkdir()
        hub = make_hub(root)
        try:
            rn = _make(hub, "valproc")
        finally:
            hub.log.close()
        disp = root / "disp"
        disp.mkdir()
        go = root / "go"
        procs = [
            _spawn(
                "advance",
                root / "hub.db",
                root / "artifacts",
                actor,
                rn,
                VAL_GO=str(go),
                VAL_DISPATCH=str(disp),
                VAL_PASSES="3",
                VAL_SUBMIT_DELAY="0.05",
            )
            for actor in actors
        ]
        time.sleep(0.3)
        go.write_text("go")
        errs = []
        for p in procs:
            _out, err = p.communicate(timeout=120)
            if p.returncode != 0:
                errs.append(err.decode()[-400:])
        log = EventLog(root / "hub.db", read_only=True)
        try:
            graph = WorkGraph(log, "prj_acme")
            per = Counter(
                (n.payload["step_id"], n.payload["round"], n.payload["attempt"])
                for n in graph.nodes()
                if n.kind == "op" and n.payload.get("recipe_node") == rn
            )
            intents = Counter(
                e.payload["op_node"] for e in log.iter_events(types=["recipe.hot_intent"])
            )
            submitted = Counter(
                e.payload["op_node"] for e in log.iter_events(types=["recipe.hot_submitted"])
            )
            creators = Counter(
                f"{graph.creator(n.id).kind}:{graph.creator(n.id).id}"
                for n in graph.nodes()
                if n.kind == "op"
            )
        finally:
            log.close()
        calls = [json.loads(line) for line in (disp / "calls.jsonl").read_text().splitlines()]
        per_req = Counter(c["request_id"] for c in calls)
        jobs = json.loads((disp / "jobs.json").read_text())
        summary.append(
            {
                "trial": t,
                "exit_errors": errs,
                "max_ops_per_attempt": max(per.values()),
                "n_ops": sum(per.values()),
                "submit_calls": len(calls),
                "max_calls_per_request_id": max(per_req.values()),
                "distinct_request_ids": len(per_req),
                "jobs": len(jobs),
                "max_intent_events": max(intents.values()),
                "max_submitted_events": max(submitted.values()),
                "op_creators": dict(creators),
            }
        )
    _obs(f"race {actors}", summary)
    assert all(not s["exit_errors"] for s in summary), summary
    assert all(s["max_ops_per_attempt"] == 1 and s["n_ops"] == 7 for s in summary)
    assert all(s["distinct_request_ids"] == 3 and s["jobs"] == 3 for s in summary)
    assert all(s["max_intent_events"] == 1 and s["max_submitted_events"] == 1 for s in summary)
    # Not asserted (reported): submit calls per request_id under the race.


@pytest.mark.parametrize("name,expect", [("valslow_idem", "reclaimed"), ("valslow_non", "lost")])
def test_sigkill_mid_step(tmp_path: Path, name: str, expect: str) -> None:
    hub = make_hub(tmp_path)
    try:
        rn = _make(hub, name)
    finally:
        hub.log.close()
    side = tmp_path / "side.log"
    db, arts = tmp_path / "hub.db", tmp_path / "artifacts"
    w1 = _spawn(
        "worker",
        db,
        arts,
        "agent:worker-1",
        rn,
        VAL_SIDEFILE=str(side),
        VAL_SLEEP="60",
        VAL_LEASE="2",
        VAL_HB="1",
        VAL_POLL="0.05",
    )
    assert _wait(lambda: side.exists() and "slow-start" in side.read_text(), 60)
    w1.send_signal(signal.SIGKILL)
    w1.wait(timeout=30)
    t_kill = time.monotonic()
    time.sleep(3.5)  # > lease_s (2) past the last heartbeat
    w2 = _spawn(
        "worker",
        db,
        arts,
        "agent:worker-2",
        rn,
        VAL_SIDEFILE=str(side),
        VAL_SLEEP="0",
        VAL_LEASE="2",
        VAL_HB="1",
        VAL_POLL="0.05",
    )

    def settled() -> bool:
        log = EventLog(db, read_only=True)
        try:
            op = load_state(log, rn).latest_in_round("slow", 1)
            return op is not None and op.status in ("done", "failed")
        finally:
            log.close()

    ok = _wait(settled, 60)
    w2.send_signal(signal.SIGTERM)
    w2.wait(timeout=30)
    log = EventLog(db, read_only=True)
    try:
        st = load_state(log, rn)
        op = st.latest_in_round("slow", 1)
        claims = [
            str(e.payload.get("worker")).split(":", 1)[-1]
            for e in log.iter_events(types=["recipe.claimed"])
        ]  # A5
        result_actor = [e.payload.get("by") for e in log.iter_events(types=["recipe.result"])]  # A5
    finally:
        log.close()
    lines = side.read_text().split("\n")
    _obs(
        f"sigkill {name}",
        {
            "settled": ok,
            "w1_rc": w1.returncode,
            "op": [op.status, (op.result_error or {}).get("code")],
            "claims": claims,
            "result_by": result_actor,
            "side": [x for x in lines if x],
            "secs_after_kill": round(time.monotonic() - t_kill, 1),
            "recipe": st.recipe_status,
        },
    )
    assert w1.returncode == -signal.SIGKILL
    if expect == "reclaimed":
        assert op.status == "done" and claims == ["worker-1", "worker-2"]
    else:
        assert op.status == "failed" and op.result_error["code"] == "lease_lost"
        assert claims == ["worker-1"]


def test_sigterm_finishes_current_step_and_claims_nothing(tmp_path: Path) -> None:
    hub = make_hub(tmp_path)
    try:
        rn = _make(hub, "valdrain")
    finally:
        hub.log.close()
    side = tmp_path / "side.log"
    w = _spawn(
        "worker",
        tmp_path / "hub.db",
        tmp_path / "artifacts",
        "agent:worker-1",
        rn,
        VAL_SIDEFILE=str(side),
        VAL_SLEEP="2",
        VAL_POLL="0.05",
    )
    assert _wait(lambda: side.exists() and "slow-start" in side.read_text(), 60)
    w.send_signal(signal.SIGTERM)
    rc = w.wait(timeout=60)
    log = EventLog(tmp_path / "hub.db", read_only=True)
    try:
        st = load_state(log, rn)
        s1, s2 = st.latest_in_round("s1", 1), st.latest_in_round("s2", 1)
    finally:
        log.close()
    lines = [x.split()[0] for x in side.read_text().splitlines()]
    _obs("sigterm", {"rc": rc, "s1": s1.status, "s2": s2.status, "side": lines})
    assert rc == 0 and s1.status == "done" and s2.status == "pending"
    assert lines == ["slow-start", "slow-end"]
