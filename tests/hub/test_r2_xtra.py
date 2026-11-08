"""Extra probe (run explicitly): build_server's DEFAULT params validator refuses bad params (F12).

The slice's own F12 MCP pin passes registry_params_validator explicitly, so it does not test the
default (it passes on 29689da). This probe uses the default.
"""

from __future__ import annotations

from typing import Any

from val_support import mcp_run
from write_support import brief_data

from aar.hub.mcp_server import build_server
from aar.hub.registry import Registry


def test_build_server_default_validator_refuses_bad_params(
    hub: Any, registered_recipe: Any
) -> None:
    server = build_server(hub.log, Registry(hub.log), hub.store, agent_writes=True)
    pid = hub.project_id

    async def body(client: Any) -> Any:
        b = await client.call_tool(
            "brief_create",
            {"agent_id": "ed", "idempotency_key": "b", "brief": brief_data(pid, recipe="rectest")},
        )
        bid = b.structured_content["node"]["id"]
        out = {}
        for label, recipe in (
            ("bogus", {"recipe": "rectest", "brief_id": bid, "params": {"bogus": 1}}),
            ("ghost", {"recipe": "ghost", "brief_id": bid}),
        ):
            r = await client.call_tool(
                "recipe_create",
                {
                    "agent_id": "ed",
                    "idempotency_key": f"r-{label}",
                    "project_id": pid,
                    "recipe": recipe,
                },
            )
            out[label] = [r.is_error, (r.structured_content or {}).get("error", {}).get("code")]
        return out

    got = mcp_run(server, body)
    print("\nOBS build_server default validator:", got)
    assert got["bogus"] == [True, "invalid_params"]
    assert got["ghost"][0] and got["ghost"][1] in ("unknown_recipe", "recipe_mismatch")


def test_status_view_per_step_summary_capped_for_many_small_keys(
    hub: Any, registered_recipe: Any
) -> None:
    import json

    from recipe_support import make_run
    from val_support import worker_idle
    from write_support import AGENT

    from aar.recipes.ports import FakeHotDispatcher, HotJobView, HotResult
    from aar.recipes.run import recipe_start
    from aar.recipes.status import status_view

    fake = FakeHotDispatcher()
    fake.set_behaviour(
        "rectest.hot",
        lambda p: HotJobView(
            "done", result=HotResult(outputs={f"k{i}": "v" * 1000 for i in range(20)})
        ),
    )
    rn = make_run(hub)
    recipe_start(hub.log, rn, AGENT, idempotency_key="s", dispatcher=fake)
    worker_idle(hub, fake)
    view = status_view(hub.log, rn, dispatcher=fake)
    h = next(s for s in view["steps"] if s["step_id"] == "h")
    size = len(json.dumps(h["outputs"], ensure_ascii=False))
    print("\nOBS 20 x 1 KB outputs: summary bytes", size, "truncated", view["truncated"])
    assert size <= 1024


def test_library_round_guard_refuses_a_stale_round(hub: Any, registered_recipe: Any) -> None:
    from recipe_support import submit_test_plan
    from val_support import reject, start_run, worker_idle
    from write_support import AGENT

    from aar.hub.writes import WriteError
    from aar.recipes.run import step_complete

    rn, fake = start_run(hub)
    worker_idle(hub, fake)
    plan = submit_test_plan(hub, rn, AGENT, key="p1")
    step_complete(
        hub.log,
        hub.store,
        rn,
        "p",
        None,
        {"plan_node": plan},
        AGENT,
        idempotency_key="c1",
        dispatcher=fake,
    )
    reject(hub, rn, fake)
    try:
        step_complete(
            hub.log,
            hub.store,
            rn,
            "p",
            1,
            {"plan_node": plan},
            AGENT,
            round=1,
            idempotency_key="c2",
            dispatcher=fake,
        )
        got = "accepted"
    except WriteError as exc:
        got = exc.code
    print("\nOBS round=1 passed in round 2 (fresh key):", got)
    assert got == "step_not_open"
