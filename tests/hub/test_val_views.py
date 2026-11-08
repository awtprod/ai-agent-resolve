"""Validator probes: status view, next_actions templates, REPORT.md, MCP surface."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import BaseModel
from recipe_support import hot_fake, make_run
from val_support import approve, complete_plan, mcp_run, reject, start_run, worker_idle
from write_support import AGENT, HUMAN, brief_data

from aar.hub.events import Actor, EventLog
from aar.hub.mcp_server import build_server
from aar.hub.registry import Registry
from aar.hub.writes import create_brief, create_recipe
from aar.recipes import registry
from aar.recipes.models import ArtifactExpect, RecipeDef, StepDef
from aar.recipes.report import build_report_text
from aar.recipes.run import recipe_start, step_complete
from aar.recipes.state import load_state
from aar.recipes.status import status_view


def _obs(label: str, value: Any) -> None:
    print(f"\nOBS {label}: {json.dumps(value, default=str, sort_keys=True)[:3000]}")


def _server(hub: Any, **kw: Any) -> Any:
    return build_server(
        hub.log,
        Registry(hub.log),
        hub.store,
        params_validator=registry.registry_params_validator,
        **kw,
    )


def test_next_actions_templates_called_literally(hub: Any, registered_recipe: Any) -> None:
    from aar.recipes.context import StepFailure

    def boom(ctx: Any) -> Any:
        raise StepFailure("boom", "fails")

    registry.register_callable("rectest.cold", boom)  # make c fail so a retry action appears
    recipe_node, fake = start_run(hub)
    worker_idle(hub, fake)
    view = status_view(hub.log, recipe_node, dispatcher=fake)
    actions = [a for a in view["next_actions"] if a.get("tool")]
    _obs("agent actions", actions)
    server = _server(hub, agent_writes=True)
    results: list[Any] = []

    async def body(client: Any) -> None:
        # A6: F14 redesign (brief): args + a `fill` list; fill ONLY the fill fields, then call.
        plan = __import__("recipe_support").submit_test_plan(hub, recipe_node, AGENT, key="pf")
        values = {"agent_id": "agent-1", "outputs": {"plan_node": plan}}
        for i, a in enumerate(actions):
            filled = {k: (values[k] if k != "idempotency_key" else f"fill-{i}") for k in a["fill"]}
            assert not (set(filled) & set(a["args"])), a
            r1 = await client.call_tool(a["tool"], {**a["args"], **filled})
            r2 = r1
            results.append(
                {
                    "tool": a["tool"],
                    "literal": [
                        r1.is_error,
                        (r1.structured_content or {}).get("error", {}).get("code")
                        if r1.structured_content
                        else str(r1.content)[:200],
                    ],
                    "literal+agent_id": [
                        r2.is_error,
                        (r2.structured_content or {}).get("error", {}).get("code")
                        if r2.structured_content
                        else str(r2.content)[:200],
                    ],
                }
            )

    mcp_run(server, body)
    _obs("literal calls", results)
    assert all(not r["literal"][0] for r in results), (
        "next_actions templates do not work as written: " + json.dumps(results)
    )


def test_status_states_and_claimed(hub: Any, registered_recipe: Any) -> None:
    from aar.recipes.engine import claim

    recipe_node, fake = start_run(hub)
    st = load_state(hub.log, recipe_node)
    c = st.latest_in_round("c", 1)
    claim(
        hub.log,
        project_id=hub.project_id,
        op=c,
        worker=AGENT,
        idempotent=True,
        clock=__import__("aar.hub.common", fromlist=["utcnow"]).utcnow,
    )
    v = status_view(hub.log, recipe_node, dispatcher=fake)
    _obs("states after a claim", {s["step_id"]: s["state"] for s in v["steps"]})
    _obs("view keys", sorted(v))


def test_status_view_size_bound(hub: Any, registered_recipe: Any) -> None:
    """Hot outputs and errors flow into the status view unbounded."""
    from aar.recipes.ports import FakeHotDispatcher, HotJobView, HotResult

    fake = FakeHotDispatcher()
    fake.set_behaviour(
        "rectest.hot",
        lambda p: HotJobView(
            "done", result=HotResult(outputs={f"k{i}": "v" * 5000 for i in range(200)})
        ),
    )
    recipe_node = make_run(hub)
    recipe_start(hub.log, recipe_node, AGENT, idempotency_key="s", dispatcher=fake)
    v = status_view(hub.log, recipe_node)
    ev = [len(json.dumps(e.payload)) for e in hub.log.iter_events(types=["recipe.result"])]
    server = _server(hub)
    out: dict[str, Any] = {}

    async def body(client: Any) -> None:
        r = await client.call_tool("recipe_status", {"recipe_node": recipe_node})
        out["mcp_bytes"] = len(json.dumps(r.structured_content))

    mcp_run(server, body)
    _obs(
        "hot outputs 1 MB",
        {"result_event_bytes": ev, "status_view_bytes": len(json.dumps(v)), **out},
    )
    assert max(ev) <= 64 * 1024, "hot result outputs are recorded unbounded"


def test_mcp_tool_lists_and_attribution(hub: Any, registered_recipe: Any) -> None:
    def names(server: Any) -> list[str]:
        async def body(client: Any) -> Any:
            return sorted(t.name for t in (await client.list_tools()).tools)

        return mcp_run(server, body)

    plain = names(_server(hub))
    plain_default_validator = names(build_server(hub.log, Registry(hub.log), hub.store))
    writes = names(_server(hub, agent_writes=True))
    human_words = ("decide", "approve", "grant", "revoke", "resume", "stop", "withdraw", "reject")
    _obs("read tools", [len(plain), plain])
    _obs("with --agent-writes", [len(writes), sorted(set(writes) - set(plain))])
    assert (
        len(plain) == 14 and plain == plain_default_validator
    )  # A7: +jobs_list from the dev release merge
    assert not [n for n in writes if any(w in n for w in human_words)]

    # Attribution: a HUMAN-created recipe; an agent completes the plan step through MCP.
    recipe_node = make_run(hub, actor=HUMAN)
    recipe_start(hub.log, recipe_node, HUMAN, idempotency_key="hs", dispatcher=hot_fake())
    worker_idle(hub, hot_fake())
    plan = __import__("recipe_support").submit_test_plan(
        hub, recipe_node, Actor("agent", "ed2"), key="pe"
    )
    before = max(e.seq for e in hub.log.iter_events())
    server = _server(hub, agent_writes=True)

    async def body(client: Any) -> Any:
        return await client.call_tool(
            "step_complete",
            {
                "agent_id": "ed2",
                "idempotency_key": "c",
                "recipe_node": recipe_node,
                "step_id": "p",
                "outputs": {"plan_node": plan},
            },
        )

    r = mcp_run(server, body)
    assert not r.is_error, r.structured_content
    written = [
        (e.type, f"{e.actor.kind}:{e.actor.id}") for e in hub.log.iter_events() if e.seq > before
    ]
    _obs("events written by ONE MCP step_complete call from agent ed2", written)
    st = load_state(hub.log, recipe_node)
    g = st.latest_in_round("g", 1)
    rv = __import__("aar.hub.reviews", fromlist=["get_review"]).get_review(
        hub.log, hub.project_id, g.review_id
    )
    _obs("gate review requested_by", rv.requested_by)
    # A7: brief F2 design: runner writes are RUNNER with the caller recorded; nothing as the human.
    non_agent = [
        w for w in written if not (w[1].startswith("agent:ed2") or w[1] == "runner:recipes")
    ]
    assert not non_agent, f"an agent's MCP call wrote events attributed to others: {non_agent}"
    callers = {
        e.payload.get("caller")
        for e in hub.log.iter_events(types=["hub.write"])
        if e.seq > before and e.actor.kind == "runner"
    }
    _obs("caller recorded on runner hub.write records", sorted(map(str, callers)))
    assert callers == {"agent:ed2"}, callers
    assert str(rv.requested_by) == "runner:recipes", rv.requested_by


def _replay(src: EventLog, path: Any) -> EventLog:
    dst = EventLog(path)
    for e in src.iter_events():
        dst.append(e.type, e.actor, e.payload, project_id=e.project_id, ts=e.ts)
    return dst


def test_report_determinism_and_content(hub: Any, registered_recipe: Any, tmp_path: Any) -> None:
    from aar.recipes.context import StepFailure

    calls = {"n": 0}
    real = registry.callable_for("rectest.qa")

    def flaky_qa(ctx: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == 1:
            raise StepFailure("qa_failed", "first QA run fails")
        return real(ctx)

    registry.register_callable("rectest.qa", flaky_qa)
    recipe_node, fake = start_run(hub)
    worker_idle(hub, fake)
    complete_plan(hub, recipe_node, fake, "r1")
    reject(hub, recipe_node, fake)
    complete_plan(hub, recipe_node, fake, "r2")
    approve(hub, recipe_node, fake)
    worker_idle(hub, fake)
    from aar.recipes.run import recipe_retry

    recipe_retry(hub.log, recipe_node, "q", AGENT, idempotency_key="rq", dispatcher=fake)
    worker_idle(hub, fake)
    st = load_state(hub.log, recipe_node)
    assert st.recipe_status == "done", st.recipe_status
    t1 = build_report_text(hub.log, recipe_node)
    t2 = build_report_text(hub.log, recipe_node)
    copy = _replay(hub.log, tmp_path / "copy.db")
    try:
        t3 = build_report_text(copy, recipe_node)
    finally:
        copy.close()
    rep = st.done_op("report")
    stored = hub.store.read_bytes(rep.result_artifacts[0]).decode()
    _obs("REPORT.md", t1)
    arts = [
        line.split()[1]
        for line in t1.split("## Artifacts")[1].splitlines()
        if line.startswith("- ")
    ]
    sha_ok = {a: hashlib.sha256(hub.store.read_bytes(a)).hexdigest() == a for a in arts}
    _obs("artifact ids are sha256 of their bytes", sha_ok)
    assert t1 == t2 == t3
    assert "attempt 1: failed [qa_failed]" in t1
    assert "by human:operator-1" in t1 and "reject" in t1 and "approve" in t1
    assert "## QA" in t1 and "decode" in t1
    assert "qa.lufs.master" in t1
    assert all(sha_ok.values())
    _obs("stored report == text at report time (status line differs?)", stored == t1)


class Pm(BaseModel):
    pass


def test_report_lists_agent_step_artifacts(hub: Any) -> None:
    registry.clear()
    registry.register(
        RecipeDef(
            name="valart",
            version=1,
            params_model=Pm,
            steps=(
                StepDef(
                    id="art", kind="agent", instruction="render", expects=ArtifactExpect("short")
                ),
                StepDef(
                    id="report",
                    kind="cold",
                    callable_id="recipe.report",
                    idempotent=True,
                    needs=("art",),
                ),
            ),
        )
    )
    try:
        brief = create_brief(
            hub.log, brief_data(hub.project_id, recipe="valart"), AGENT, idempotency_key="b"
        )
        rn = create_recipe(
            hub.log,
            hub.project_id,
            {"recipe": "valart", "brief_id": brief.body["node"]["id"], "params": {}},
            AGENT,
            idempotency_key="r",
            params_validator=registry.registry_params_validator,
        ).body["node"]["id"]
        recipe_start(hub.log, rn, AGENT, idempotency_key="s")
        aid = hub.store.put(
            b"the short", actor=AGENT, project_id=hub.project_id, meta={"kind": "short"}
        )
        step_complete(
            hub.log, hub.store, rn, "art", None, {"artifact_id": aid}, AGENT, idempotency_key="c"
        )
        worker_idle(hub, None)
        text = build_report_text(hub.log, rn)
        from aar.hub.graph import WorkGraph

        g = WorkGraph(hub.log, hub.project_id)
        art_op = load_state(hub.log, rn).done_op("art")
        produces = [e for e in g.edges() if e.type == "produces" and e.src == art_op.op_id]
        _obs("agent artifact in REPORT / produces edges", [aid in text, len(produces)])
        assert aid in text and produces, (
            "an agent step's artifact (ArtifactExpect) is neither linked with a produces edge "
            "nor listed in REPORT.md"
        )
    finally:
        registry.clear()
