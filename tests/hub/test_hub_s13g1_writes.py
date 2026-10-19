"""S13g1: the typed hub write surface (briefs, recipes, plans, own status).

Every test here exercises code that does not exist on the base commit (``aar.hub.writes``).
The old-log no-change pin lives in test_hub_s13g1_pins.py, which imports only base APIs.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from write_support import (
    AGENT,
    HUMAN,
    OTHER,
    Crash,
    Hub,
    brief_data,
    crash_after_appends,
    make_hub,
    plan_fixture,
)

from aar.hub import writes
from aar.hub.events import Actor, EventLog, canonical_json
from aar.hub.graph import (
    GRAPH_EVENT_TYPES,
    NODE_KINDS,
    WorkGraph,
)
from aar.hub.reviews import decide, request_review
from aar.hub.writes import WriteError
from aar.plan.lint import all_rules, lint
from aar.plan.model import EditPlan
from aar.plan.profile import load_profile


@pytest.fixture
def hub(tmp_path: Path) -> Iterator[Hub]:
    made = make_hub(tmp_path)
    yield made
    made.log.close()


def _refused(code: str, call: Any) -> WriteError:
    with pytest.raises(WriteError) as info:
        call()
    assert info.value.code == code, info.value.to_dict()
    return info.value


def _brief(hub: Hub, key: str = "brief-1", actor: Actor = AGENT, **changes: Any) -> str:
    result = writes.create_brief(
        hub.log, brief_data(hub.project_id, **changes), actor, idempotency_key=key
    )
    return result.body["node"]["id"]


def _recipe(hub: Hub, brief_id: str, key: str = "recipe-1", actor: Actor = AGENT) -> str:
    result = writes.create_recipe(
        hub.log,
        hub.project_id,
        {"recipe": "ccn-short", "brief_id": brief_id, "params": {"max_seconds": 37}},
        actor,
        idempotency_key=key,
    )
    return result.body["node"]["id"]


def _plan(
    hub: Hub, parent: str, name: str = "short_clean", key: str = "plan-1", **kw: Any
) -> writes.WriteResult:
    return writes.submit_plan(
        hub.log,
        hub.store,
        hub.project_id,
        plan_fixture(name),
        kw.pop("actor", AGENT),
        derives_from=parent,
        idempotency_key=key,
        **kw,
    )


# -- the recipe node kind and old logs ------------------------------------------------------


def test_recipe_is_a_node_kind_that_folds(hub: Hub) -> None:
    assert "recipe" in NODE_KINDS
    graph = WorkGraph(hub.log, hub.project_id)
    node = graph.add_node("recipe", AGENT, title="raw recipe")
    assert WorkGraph(hub.log, hub.project_id).node(node.id).kind == "recipe"
    assert graph.skipped() == ()


def test_add_node_in_and_add_edge_in_write_inside_the_callers_transaction(hub: Hub) -> None:
    brief = WorkGraph(hub.log, hub.project_id).add_node("brief", AGENT)
    # One WorkGraph per transaction (as with set_status_in): after a rollback its live
    # state is stale, so the next transaction builds a fresh one.
    with pytest.raises(RuntimeError, match="boom"), hub.log.transaction() as txn:
        graph = WorkGraph(hub.log, hub.project_id)
        node_event = graph.add_node_in(txn, "recipe", AGENT, idempotency_key="k#node")
        graph.add_edge_in(txn, "derives_from", node_event.payload["node_id"], brief.id, AGENT)
        raise RuntimeError("boom")
    fresh = WorkGraph(hub.log, hub.project_id)
    assert [n.kind for n in fresh.nodes()] == ["brief"]  # rolled back: nothing kept
    assert fresh.edges() == []
    with hub.log.transaction() as txn:
        graph = WorkGraph(hub.log, hub.project_id)
        node_event = graph.add_node_in(txn, "recipe", AGENT, idempotency_key="k#node")
        edge_event = graph.add_edge_in(
            txn, "derives_from", node_event.payload["node_id"], brief.id, AGENT
        )
    fresh = WorkGraph(hub.log, hub.project_id)
    assert fresh.node(node_event.payload["node_id"]).kind == "recipe"
    assert fresh.edges()[0].id == edge_event.payload["edge_id"]
    assert fresh.creator(node_event.payload["node_id"]) == AGENT


def test_add_node_in_and_add_edge_in_refuse_what_add_node_and_add_edge_refuse(
    hub: Hub, tmp_path: Path
) -> None:
    graph = WorkGraph(hub.log, hub.project_id)
    art = graph.add_node("artifact", AGENT)
    plan = graph.add_node("plan", AGENT)
    other = EventLog(tmp_path / "other.db")
    with other.transaction() as txn, pytest.raises(ValueError, match="different event log"):
        graph.add_node_in(txn, "brief", AGENT)
    other.close()
    with hub.log.transaction() as txn:
        with pytest.raises(ValueError, match="node kind"):
            graph.add_node_in(txn, "bogus", AGENT)
        with pytest.raises(ValueError, match="status"):
            graph.add_node_in(txn, "brief", AGENT, status="weird")
        with pytest.raises(ValueError, match="title"):
            graph.add_node_in(txn, "brief", AGENT, title="x" * 1001)
        with pytest.raises(ValueError, match="edge type"):
            graph.add_edge_in(txn, "bogus", plan.id, art.id, AGENT)
        with pytest.raises(KeyError):
            graph.add_edge_in(txn, "derives_from", "nd_missing", art.id, AGENT)
        with pytest.raises(KeyError):
            graph.add_edge_in(txn, "derives_from", plan.id, "nd_missing", AGENT)
        with pytest.raises(ValueError, match="must start at"):
            graph.add_edge_in(txn, "produces", plan.id, art.id, AGENT)
        with pytest.raises(ValueError, match="must end at"):
            graph.add_edge_in(
                txn,
                "produces",
                graph.add_node_in(txn, "op", AGENT).payload["node_id"],
                plan.id,
                AGENT,
            )


def test_graph_event_types_are_public_and_unchanged() -> None:
    assert GRAPH_EVENT_TYPES == (
        "graph.node_added",
        "graph.node_status_changed",
        "graph.edge_added",
    )


# -- the typed models -------------------------------------------------------------------------


def test_brief_round_trips_with_its_schema_literal(hub: Hub) -> None:
    model = writes.Brief.model_validate(brief_data(hub.project_id))
    dumped = model.dump()
    assert dumped["schema"] == "aar.brief/1"
    assert writes.Brief.model_validate(dumped) == model
    assert [s["kind"] for s in dumped["sources"]] == ["cloud_project", "timeline", "media"]


@pytest.mark.parametrize(
    ("changes", "loc"),
    [
        ({"show": 7}, "show"),  # strict: no coercion
        ({"show": ""}, "show"),
        ({"show": "   "}, "show"),  # empty after stripping
        ({"show": "x" * 201}, "show"),
        ({"show": "bad\x07bell"}, "show"),
        ({"title": "two\nlines"}, "title"),
        ({"hook": "h" * 301}, "hook"),
        ({"handle": "h" * 101}, "handle"),
        ({"notes": "n" * 8001}, "notes"),
        ({"kind": "trailer"}, "kind"),
        ({"recipe": "Bad Name"}, "recipe"),
        ({"rule_profile": "/etc/aar/profile.toml"}, "rule_profile"),
        ({"rule_profile": "ccn\n"}, "rule_profile"),
        ({"schema": "aar.brief/2"}, "schema"),
        ({"schema_": "aar.brief/1"}, "schema_"),
        ({"extra": 1}, "extra"),
        ({"deliverables": []}, "deliverables"),
        ({"deliverables": [{"name": f"d{i}", "kind": "short"} for i in range(33)]}, "deliverables"),
        ({"deliverables": [{"name": "a.b", "kind": "short"}]}, "deliverables[0].name"),
        ({"deliverables": [{"name": "a", "kind": "gif"}]}, "deliverables[0].kind"),
        ({"sources": [{"kind": "media", "path": "x"}] * 65}, "sources"),
        ({"sources": [{"kind": "rumour"}]}, "sources[0]"),
        (
            {"sources": [{"kind": "timeline", "unique_id": "has space"}]},
            "sources[0].timeline.unique_id",
        ),
        ({"sources": [{"kind": "media", "path": "m", "sha256": "ABC"}]}, "sources[0].media.sha256"),
        ({"sources": [{"kind": "media", "path": "p" * 1025}]}, "sources[0].media.path"),
    ],
)
def test_brief_validation_names_the_field(hub: Hub, changes: dict[str, Any], loc: str) -> None:
    error = _refused(
        "invalid_brief",
        lambda: writes.create_brief(
            hub.log, brief_data(hub.project_id, **changes), AGENT, idempotency_key="b"
        ),
    )
    locs = [item["loc"] for item in error.details["errors"]]
    assert loc in locs, locs
    assert hub.node_kinds()["brief"] == 0


def test_brief_deliverable_names_are_unique(hub: Hub) -> None:
    twice = [{"name": "short-1", "kind": "short"}, {"name": "short-1", "kind": "review"}]
    error = _refused(
        "invalid_brief",
        lambda: writes.create_brief(
            hub.log, brief_data(hub.project_id, deliverables=twice), AGENT, idempotency_key="b"
        ),
    )
    assert "used twice" in json.dumps(error.details)


def test_brief_text_is_stripped_and_notes_keep_newlines(hub: Hub) -> None:
    result = writes.create_brief(
        hub.log,
        brief_data(hub.project_id, show="  Acme  ", notes="line one\r\nline two"),
        AGENT,
        idempotency_key="b",
    )
    payload = result.body["node"]["payload"]
    assert payload["show"] == "Acme"
    assert payload["notes"] == "line one\nline two"


def test_brief_that_is_not_an_object_is_refused(hub: Hub) -> None:
    _refused("invalid_brief", lambda: writes.create_brief(hub.log, [1], AGENT, idempotency_key="b"))


def test_recipe_model_bounds() -> None:
    ok = writes.RecipeData.model_validate({"recipe": "r", "brief_id": "nd_1"})
    assert ok.max_review_rounds == 3
    assert ok.dump()["schema"] == "aar.recipe/1"
    for rounds in (1, 10):
        assert (
            writes.RecipeData.model_validate(
                {"recipe": "r", "brief_id": "nd_1", "max_review_rounds": rounds}
            ).max_review_rounds
            == rounds
        )
    for bad in (0, 11, True, 3.0, "3"):
        with pytest.raises(ValueError):
            writes.RecipeData.model_validate(
                {"recipe": "r", "brief_id": "nd_1", "max_review_rounds": bad}
            )


@pytest.mark.parametrize(
    "params",
    [
        {"x": float("nan")},
        {"x": float("inf")},
        {"x": 2**63},
        {"x": "s" * 100_001},
        {"deep": {"a": {"b": {"c": {"d": {"e": {"f": {"g": {"h": 1}}}}}}}}},
        {"big": ["x" * 1000] * 70},  # over 64 KiB of canonical JSON
        {"many": list(range(5001))},
        {"x": {1, 2}},
    ],
)
def test_recipe_params_must_be_bounded_json(params: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="params"):
        writes.RecipeData.model_validate({"recipe": "r", "brief_id": "nd_1", "params": params})


def test_json_problem_walks_without_recursion() -> None:
    value: Any = 1
    for _ in range(50_000):
        value = [value]
    problem = writes.json_problem(value, max_depth=8, max_items=10**6, max_bytes=10**9)
    assert problem is not None and "nested deeper" in problem[1]
    assert (
        writes.json_problem(
            {"a": [1, 2.5, None, True, "x"]}, max_depth=3, max_items=10, max_bytes=100
        )
        is None
    )
    assert writes.json_problem({1: "x"}, max_depth=3, max_items=10, max_bytes=100) == (
        "",
        "has a key that is not a string",
    )


# -- create_brief and idempotency ---------------------------------------------------------------


def test_create_brief_writes_one_typed_pending_node_and_a_write_record(hub: Hub) -> None:
    result = writes.create_brief(
        hub.log, brief_data(hub.project_id), AGENT, idempotency_key="brief-1"
    )
    body = result.to_dict()
    assert body["ok"] is True and body["op"] == "brief.create" and body["replayed"] is False
    node = body["node"]
    assert node["kind"] == "brief" and node["status"] == "pending"
    assert node["title"] == "Parking Rant"
    assert node["payload"] == writes.Brief.model_validate(brief_data(hub.project_id)).dump()
    graph = WorkGraph(hub.log, hub.project_id)
    assert graph.creator(node["id"]) == AGENT
    record = hub.log.get_by_idempotency_key(writes.write_key(AGENT, "brief-1"))
    assert record is not None and record.type == "hub.write"
    assert record.payload["op"] == "brief.create" and record.payload["node_id"] == node["id"]


def test_brief_title_defaults_to_show_and_kind(hub: Hub) -> None:
    data = brief_data(hub.project_id)
    del data["title"]
    result = writes.create_brief(hub.log, data, AGENT, idempotency_key="b")
    assert result.body["node"]["title"] == "Acme Comedy short"


def test_same_key_same_request_replays_the_first_result(hub: Hub) -> None:
    first = writes.create_brief(hub.log, brief_data(hub.project_id), AGENT, idempotency_key="b")
    # A later status move must not change what the replay returns.
    writes.set_own_status(
        hub.log, hub.project_id, first.body["node"]["id"], "active", AGENT, idempotency_key="s"
    )
    before = hub.counts()
    again = writes.create_brief(hub.log, brief_data(hub.project_id), AGENT, idempotency_key="b")
    assert again.replayed is True
    assert {**again.to_dict(), "replayed": False} == first.to_dict()
    assert hub.counts() == before
    # Normalisation: the same brief with text that strips to the same value is the same request.
    padded = brief_data(hub.project_id, show=" Acme Comedy ")
    assert writes.create_brief(hub.log, padded, AGENT, idempotency_key="b").replayed is True


def test_same_key_different_request_is_a_conflict(hub: Hub) -> None:
    writes.create_brief(hub.log, brief_data(hub.project_id), AGENT, idempotency_key="b")
    before = hub.counts()
    error = _refused(
        "idempotency_conflict",
        lambda: writes.create_brief(
            hub.log, brief_data(hub.project_id, show="Other"), AGENT, idempotency_key="b"
        ),
    )
    assert error.details["idempotency_key"] == "b"
    assert error.fix_hint
    # The same key in another project, or for another op, is a different request too.
    _refused(
        "idempotency_conflict",
        lambda: writes.create_brief(
            hub.log, brief_data(hub.other_project_id), AGENT, idempotency_key="b"
        ),
    )
    brief_id = WorkGraph(hub.log, hub.project_id).nodes()[0].id
    _refused(
        "idempotency_conflict",
        lambda: writes.set_own_status(
            hub.log, hub.project_id, brief_id, "active", AGENT, idempotency_key="b"
        ),
    )
    assert hub.counts() == before


def test_keys_are_namespaced_per_actor(hub: Hub) -> None:
    first = writes.create_brief(hub.log, brief_data(hub.project_id), AGENT, idempotency_key="b")
    second = writes.create_brief(hub.log, brief_data(hub.project_id), OTHER, idempotency_key="b")
    assert second.replayed is False
    assert second.body["node"]["id"] != first.body["node"]["id"]
    # An id or key with ':' or '/' cannot make two actors' keys collide.
    assert writes.write_key(Actor("human", "a:b"), "c") != writes.write_key(
        Actor("human", "a"), "b:c"
    )


@pytest.mark.parametrize("key", ["", " k", "k k", "k\n", "-k", "x" * 129, 7, None, "é"])
def test_bad_idempotency_keys_are_refused(hub: Hub, key: Any) -> None:
    error = _refused(
        "invalid_input",
        lambda: writes.create_brief(
            hub.log, brief_data(hub.project_id), AGENT, idempotency_key=key
        ),
    )
    assert error.details["field"] == "idempotency_key"


def test_agent_ids_follow_the_agent_id_rule(hub: Hub) -> None:
    for bad in ("bad id", "a:b", "x" * 129):
        error = _refused(
            "invalid_input",
            lambda bad=bad: writes.create_brief(
                hub.log, brief_data(hub.project_id), Actor("agent", bad), idempotency_key="b"
            ),
        )
        assert error.details["field"] == "agent_id"
    long_human = Actor("human", "h" * 129)
    _refused(
        "invalid_input",
        lambda: writes.create_brief(
            hub.log, brief_data(hub.project_id), long_human, idempotency_key="b"
        ),
    )
    with pytest.raises(TypeError):
        writes.create_brief(hub.log, brief_data(hub.project_id), "agent:x", idempotency_key="b")


def test_unknown_project_and_unknown_profile_are_refused_before_writing(hub: Hub) -> None:
    before = hub.counts()
    _refused(
        "unknown_project",
        lambda: writes.create_brief(hub.log, brief_data("prj_missing"), AGENT, idempotency_key="b"),
    )
    error = _refused(
        "unknown_profile",
        lambda: writes.create_brief(
            hub.log, brief_data(hub.project_id, rule_profile="nope"), AGENT, idempotency_key="b"
        ),
    )
    assert error.details["field"] == "rule_profile"
    assert hub.counts() == before


def test_forged_write_records_are_conflicts_never_replays(hub: Hub) -> None:
    key = writes.write_key(AGENT, "forged")
    hub.log.append(
        "hub.write", AGENT, {"op": "plan.submit"}, project_id=hub.project_id, idempotency_key=key
    )
    _refused(
        "idempotency_conflict",
        lambda: writes.create_brief(
            hub.log, brief_data(hub.project_id), AGENT, idempotency_key="forged"
        ),
    )
    # A record with the right op and fingerprint but no node event behind it.
    data = writes.Brief.model_validate(brief_data(hub.project_id)).dump()
    fingerprint = writes.request_sha256(
        "brief.create", {"project_id": hub.project_id, "brief": data}
    )
    hub.log.append(
        "hub.write",
        AGENT,
        {"op": "brief.create", "request_sha256": fingerprint},
        project_id=hub.project_id,
        idempotency_key=writes.write_key(AGENT, "hollow"),
    )
    error = _refused(
        "idempotency_conflict",
        lambda: writes.create_brief(
            hub.log, brief_data(hub.project_id), AGENT, idempotency_key="hollow"
        ),
    )
    assert "cannot find" in error.message
    # Another event type squatting on the key, and a squatter on a sub-key.
    hub.log.append("note", AGENT, {}, idempotency_key=writes.write_key(AGENT, "squat"))
    _refused(
        "idempotency_conflict",
        lambda: writes.create_brief(
            hub.log, brief_data(hub.project_id), AGENT, idempotency_key="squat"
        ),
    )
    hub.log.append("note", AGENT, {}, idempotency_key=writes.write_key(AGENT, "sub") + "#node")
    before = hub.counts()
    _refused(
        "idempotency_conflict",
        lambda: writes.create_brief(
            hub.log, brief_data(hub.project_id), AGENT, idempotency_key="sub"
        ),
    )
    assert hub.counts() == before  # rolled back whole


@pytest.mark.parametrize("crash_at", [1, 2])
def test_create_brief_survives_a_crash_after_each_event(
    hub: Hub, monkeypatch: pytest.MonkeyPatch, crash_at: int
) -> None:
    before = hub.counts()
    with monkeypatch.context() as patched:
        seen = crash_after_appends(patched, crash_at)
        with pytest.raises(Crash):
            writes.create_brief(hub.log, brief_data(hub.project_id), AGENT, idempotency_key="b")
    assert seen == ["graph.node_added", "hub.write"][:crash_at]
    assert hub.counts() == before  # one transaction: nothing survives the crash
    result = writes.create_brief(hub.log, brief_data(hub.project_id), AGENT, idempotency_key="b")
    assert result.replayed is False
    after = hub.counts()
    assert after["graph.node_added"] == before["graph.node_added"] + 1
    assert after["hub.write"] == before["hub.write"] + 1
    assert (
        writes.create_brief(
            hub.log, brief_data(hub.project_id), AGENT, idempotency_key="b"
        ).replayed
        is True
    )
    assert hub.counts() == after


# -- create_recipe ------------------------------------------------------------------------------


def test_create_recipe_derives_from_its_brief(hub: Hub) -> None:
    brief_id = _brief(hub)
    result = writes.create_recipe(
        hub.log,
        hub.project_id,
        {"recipe": "ccn-short", "brief_id": brief_id, "params": {"max_seconds": 37}},
        AGENT,
        idempotency_key="r",
    )
    node, edge = result.body["node"], result.body["edge"]
    assert node["kind"] == "recipe" and node["status"] == "pending"
    assert node["payload"] == {
        "schema": "aar.recipe/1",
        "recipe": "ccn-short",
        "brief_id": brief_id,
        "params": {"max_seconds": 37},
        "max_review_rounds": 3,
    }
    assert (edge["type"], edge["src"], edge["dst"]) == ("derives_from", node["id"], brief_id)
    again = writes.create_recipe(
        hub.log,
        hub.project_id,
        {"recipe": "ccn-short", "brief_id": brief_id, "params": {"max_seconds": 37}},
        AGENT,
        idempotency_key="r",
    )
    assert again.replayed is True and {**again.to_dict(), "replayed": False} == result.to_dict()
    _refused(
        "idempotency_conflict",
        lambda: writes.create_recipe(
            hub.log,
            hub.project_id,
            {"recipe": "ccn-short", "brief_id": brief_id, "params": {"max_seconds": 30}},
            AGENT,
            idempotency_key="r",
        ),
    )


def test_recipe_parent_must_be_a_typed_brief_asking_for_that_recipe(hub: Hub) -> None:
    brief_id = _brief(hub)
    graph = WorkGraph(hub.log, hub.project_id)
    plan_node = graph.add_node("plan", AGENT).id
    untyped = graph.add_node("brief", AGENT, data={"show": "raw"}).id
    before = hub.counts()

    def make(brief: str, recipe: str = "ccn-short") -> Any:
        return lambda: writes.create_recipe(
            hub.log,
            hub.project_id,
            {"recipe": recipe, "brief_id": brief},
            AGENT,
            idempotency_key=f"r-{brief}-{recipe}",
        )

    assert _refused("unknown_parent", make("nd_missing")).details["brief_id"] == "nd_missing"
    assert _refused("wrong_parent_kind", make(plan_node)).details["kind"] == "plan"
    _refused("untyped_parent", make(untyped))
    error = _refused("recipe_mismatch", make(brief_id, "ccn-longform"))
    assert error.details["brief_recipe"] == "ccn-short"
    _refused(
        "unknown_project",
        lambda: writes.create_recipe(
            hub.log,
            "prj_missing",
            {"recipe": "ccn-short", "brief_id": brief_id},
            AGENT,
            idempotency_key="r",
        ),
    )
    # A brief in another project is not this project's node.
    other_brief = writes.create_brief(
        hub.log, brief_data(hub.other_project_id), AGENT, idempotency_key="ob"
    ).body["node"]["id"]
    before = hub.counts()
    _refused("unknown_parent", make(other_brief))
    _refused(
        "invalid_recipe",
        lambda: writes.create_recipe(
            hub.log, hub.project_id, {"recipe": "ccn-short"}, AGENT, idempotency_key="r"
        ),
    )
    _refused(
        "invalid_recipe",
        lambda: writes.create_recipe(hub.log, hub.project_id, "x", AGENT, idempotency_key="r"),
    )
    assert hub.counts() == before


def test_recipe_params_validator_hook(hub: Hub) -> None:
    brief_id = _brief(hub)
    calls: list[tuple[str, dict[str, Any], str]] = []

    def strict(recipe: str, params: Any, brief: writes.Brief) -> list[tuple[str, str]]:
        calls.append((recipe, dict(params), brief.show))
        return [("params.fps", "unknown parameter")] if "fps" in params else []

    before = hub.counts()
    error = _refused(
        "invalid_params",
        lambda: writes.create_recipe(
            hub.log,
            hub.project_id,
            {"recipe": "ccn-short", "brief_id": brief_id, "params": {"fps": 25}},
            AGENT,
            idempotency_key="r",
            params_validator=strict,
        ),
    )
    assert error.details["problems"] == [{"loc": "params.fps", "msg": "unknown parameter"}]
    assert hub.counts() == before
    ok = writes.create_recipe(
        hub.log,
        hub.project_id,
        {"recipe": "ccn-short", "brief_id": brief_id, "params": {"max_seconds": 37}},
        AGENT,
        idempotency_key="r2",
        params_validator=strict,
    )
    assert ok.replayed is False
    assert calls == [
        ("ccn-short", {"fps": 25}, "Acme Comedy"),
        ("ccn-short", {"max_seconds": 37}, "Acme Comedy"),
    ]
    # The default hook accepts any bounded object.
    assert list(writes.accept_any_params("x", {"anything": [1, {"a": None}]}, None)) == []
    many = [(f"params.p{i}", "bad") for i in range(60)]
    error = _refused(
        "invalid_params",
        lambda: writes.create_recipe(
            hub.log,
            hub.project_id,
            {"recipe": "ccn-short", "brief_id": brief_id},
            AGENT,
            idempotency_key="r3",
            params_validator=lambda *a: many,
        ),
    )
    assert len(error.details["problems"]) == 50 and error.details["omitted"] == 10


@pytest.mark.parametrize("crash_at", [1, 2, 3])
def test_create_recipe_survives_a_crash_after_each_event(
    hub: Hub, monkeypatch: pytest.MonkeyPatch, crash_at: int
) -> None:
    brief_id = _brief(hub)
    before = hub.counts()
    request = {"recipe": "ccn-short", "brief_id": brief_id}
    with monkeypatch.context() as patched:
        seen = crash_after_appends(patched, crash_at)
        with pytest.raises(Crash):
            writes.create_recipe(hub.log, hub.project_id, request, AGENT, idempotency_key="r")
    assert seen == ["graph.node_added", "graph.edge_added", "hub.write"][:crash_at]
    assert hub.counts() == before
    result = writes.create_recipe(hub.log, hub.project_id, request, AGENT, idempotency_key="r")
    after = hub.counts()
    assert after - before == {"graph.node_added": 1, "graph.edge_added": 1, "hub.write": 1}
    replay = writes.create_recipe(hub.log, hub.project_id, request, AGENT, idempotency_key="r")
    assert replay.replayed and replay.body == result.body
    assert hub.counts() == after


# -- submit_plan --------------------------------------------------------------------------------


def _nested(depth: int) -> Any:
    value: Any = 1
    for _ in range(depth):
        value = [value]
    return value


def _lint_of(name: str) -> Any:
    plan = EditPlan.model_validate(plan_fixture(name))
    return plan, lint(plan, load_profile(plan.profile))


def test_clean_plan_is_stored_pending_with_its_artifact_and_summary(hub: Hub) -> None:
    parsed, report = _lint_of("short_clean")
    assert report.ok  # precondition: this fixture has no lint errors
    brief_id = _brief(hub)
    recipe_id = _recipe(hub, brief_id)
    result = _plan(hub, recipe_id)
    node, edge = result.body["node"], result.body["edge"]
    assert node["kind"] == "plan" and node["status"] == "pending"
    assert (edge["type"], edge["src"], edge["dst"]) == ("derives_from", node["id"], recipe_id)
    data = node["payload"]
    content = hub.store.read_bytes(data["plan_sha256"])
    assert hashlib.sha256(content).hexdigest() == data["plan_sha256"] == result.body["artifact_id"]
    assert content == canonical_json(parsed.model_dump(mode="json")).encode()
    assert EditPlan.model_validate_json(content) == parsed
    meta = hub.store.get_meta(data["plan_sha256"])
    assert meta.mime == "application/json" and meta.project_id == hub.project_id
    assert data["schema"] == "aar.plan-node/1"
    assert data["plan_version"] == parsed.version
    assert data["rule_profile"] == "ccn"
    assert data["derives_from"] == recipe_id
    assert data["brief_id"] == brief_id
    summary = data["lint"]
    assert summary["ok"] is True
    assert (summary["errors"], summary["warnings"], summary["infos"]) == (
        report.counts["error"],
        report.counts["warn"],
        report.counts["info"],
    )
    profile = load_profile("ccn")
    rules = all_rules()
    assert summary["skipped_rules"] == sorted(
        rid for rid, cfg in profile.rules.items() if cfg.enabled and rules[rid].needs
    )
    assert summary["skipped_rules"]  # the hub lints without a transcript or laugh map
    assert writes.PlanNodeData.model_validate(data).lint.ok is True
    assert result.body["node"]["title"] == f"plan v{parsed.version}: {parsed.timeline}"


def test_a_plan_without_a_version_gets_the_current_one_explicitly(hub: Hub) -> None:
    """Version-agnostic: whatever version aar.plan defaults to is what the node records."""
    from aar.plan.model import PLAN_VERSION

    plan = plan_fixture("short_clean")
    del plan["version"]
    result = writes.submit_plan(
        hub.log,
        hub.store,
        hub.project_id,
        plan,
        AGENT,
        derives_from=_brief(hub),
        idempotency_key="nov",
    )
    data = result.body["node"]["payload"]
    assert data["plan_version"] == PLAN_VERSION
    stored = json.loads(hub.store.read_bytes(data["plan_sha256"]))
    assert stored["version"] == PLAN_VERSION  # the canonical form spells it out
    assert hub.store.get_meta(data["plan_sha256"]).meta == {
        "schema": "aar.edit-plan",
        "version": PLAN_VERSION,
    }


def test_plan_with_lint_errors_is_stored_blocked_errors_first(hub: Hub) -> None:
    _, report = _lint_of("short_broken")
    assert report.counts["error"] > 0
    result = _plan(hub, _brief(hub), "short_broken")
    node = result.body["node"]
    assert node["status"] == "blocked"
    summary = node["payload"]["lint"]
    assert summary["ok"] is False and summary["errors"] == report.counts["error"]
    severities = [f["severity"] for f in summary["findings"]]
    assert severities == sorted(severities, key=["error", "warn", "info"].index)
    assert len(summary["findings"]) == min(20, len(report.findings))
    assert summary["omitted"] == len(report.findings) - len(summary["findings"])
    first_error = next(f for f in report.findings if f.severity == "error")
    assert summary["findings"][0]["rule"] == first_error.rule
    assert all(len(f["refs"]) <= 4 for f in summary["findings"])


def test_lint_summary_caps_findings_and_text() -> None:
    from aar.plan.lint import Finding, LintReport, Ref

    findings = [
        Finding(rule="r", severity="info", message="i", fix_hint="f"),
        *[
            Finding(
                rule="r",
                severity="error",
                message="m" * 900,
                fix_hint="f" * 900,
                refs=[Ref(segment=i) for i in range(9)],
            )
            for i in range(30)
        ],
    ]
    report = LintReport(
        profile="ccn",
        ok=False,
        counts={"error": 30, "warn": 0, "info": 1},
        duration_frames=0,
        duration_s=0,
        findings=findings,
    )
    summary = writes.lint_summary(report, load_profile("ccn"))
    assert len(summary.findings) == 20 and summary.omitted == 11
    assert all(f.severity == "error" for f in summary.findings)
    assert all(len(f.message) == 500 and f.message.endswith("…") for f in summary.findings)
    assert all(len(f.refs) == 4 for f in summary.findings)
    assert summary.errors == 30 and summary.infos == 1 and summary.ok is False


def test_invalid_plan_is_refused_naming_each_location_and_stores_nothing(hub: Hub) -> None:
    brief_id = _brief(hub)
    before = hub.counts()
    plan = plan_fixture("short_clean")
    plan["kind"] = "trailer"
    plan["segments"][1] = {"track": 1}
    plan["surprise"] = True
    error = _refused(
        "invalid_plan",
        lambda: writes.submit_plan(
            hub.log,
            hub.store,
            hub.project_id,
            plan,
            AGENT,
            derives_from=brief_id,
            idempotency_key="p",
        ),
    )
    locs = {item["loc"] for item in error.details["errors"]}
    assert {"kind", "segments[1].source", "segments[1].src_in", "surprise"} <= locs
    assert all({"loc", "msg", "type"} == set(item) for item in error.details["errors"])
    assert hub.counts() == before
    assert not any(p.is_file() for p in (hub.root / "artifacts").rglob("*"))


@pytest.mark.parametrize(
    ("mutate", "loc"),
    [
        (lambda p: p.update(fps={"num": float("nan"), "den": 1}), "fps.num"),
        (lambda p: p.update(markers=[{"record": 0, "name": "x" * 100_000}] * 11), "plan"),
        (lambda p: p.update(deep=_nested(40)), "deep"),
    ],
)
def test_unbounded_plans_are_refused_before_parsing(hub: Hub, mutate: Any, loc: str) -> None:
    brief_id = _brief(hub)
    plan = plan_fixture("short_clean")
    mutate(plan)
    error = _refused(
        "invalid_plan",
        lambda: writes.submit_plan(
            hub.log,
            hub.store,
            hub.project_id,
            plan,
            AGENT,
            derives_from=brief_id,
            idempotency_key="p",
        ),
    )
    assert error.details["errors"][0]["loc"].startswith(loc.split(".")[0])
    assert error.details["errors"][0]["type"] == "bounds"


def test_plan_that_is_not_an_object_is_refused(hub: Hub) -> None:
    brief_id = _brief(hub)
    _refused(
        "invalid_plan",
        lambda: writes.submit_plan(
            hub.log,
            hub.store,
            hub.project_id,
            ["x"],
            AGENT,
            derives_from=brief_id,
            idempotency_key="p",
        ),
    )


def test_plan_profile_must_be_a_built_in_name_never_a_path(
    hub: Hub, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    brief_id = _brief(hub)
    real = tmp_path / "lax.toml"
    real.write_text(
        (Path(writes.__file__).parents[1] / "plan" / "profiles" / "ccn.toml").read_text(
            encoding="utf-8"
        ),
        encoding="utf-8",
    )
    opened: list[str] = []
    real_load = writes.load_profile

    def spy(name: Any) -> Any:
        opened.append(str(name))
        return real_load(name)

    monkeypatch.setattr(writes, "load_profile", spy)
    for profile in (str(real), "../ccn", "CCN", "nope"):
        plan = plan_fixture("short_clean")
        plan["profile"] = profile
        error = _refused(
            "unknown_profile",
            lambda plan=plan: writes.submit_plan(
                hub.log,
                hub.store,
                hub.project_id,
                plan,
                AGENT,
                derives_from=brief_id,
                idempotency_key="p",
            ),
        )
        assert error.details["field"] == "plan.profile"
    assert str(real) not in opened  # a path is never handed to load_profile


def test_plan_profile_must_match_its_briefs_rule_profile(
    hub: Hub, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A second built-in profile named "lax" (the ccn rules under another name) lets the
    # mismatch be seen; profile loading is the dependency here, not the check under test.
    real_load = writes.load_profile

    def with_lax(name: str) -> Any:
        if name == "lax":
            return real_load("ccn").model_copy(update={"name": "lax"})
        return real_load(name)

    monkeypatch.setattr(writes, "load_profile", with_lax)
    brief_id = _brief(hub)
    recipe_id = _recipe(hub, brief_id)
    first_plan = _plan(hub, recipe_id).body["node"]["id"]
    lax = plan_fixture("short_clean")
    lax["profile"] = "lax"
    before = hub.counts()
    for parent in (brief_id, recipe_id, first_plan):
        error = _refused(
            "profile_mismatch",
            lambda parent=parent: writes.submit_plan(
                hub.log,
                hub.store,
                hub.project_id,
                lax,
                AGENT,
                derives_from=parent,
                idempotency_key=f"lax-{parent}",
            ),
        )
        assert error.details == {
            "brief_id": brief_id,
            "plan_profile": "lax",
            "brief_profile": "ccn",
        }
    assert hub.counts() == before
    # An untyped ancestry has no brief to hold the plan to.
    raw_brief = WorkGraph(hub.log, hub.project_id).add_node("brief", HUMAN).id
    stored = writes.submit_plan(
        hub.log,
        hub.store,
        hub.project_id,
        lax,
        AGENT,
        derives_from=raw_brief,
        idempotency_key="lax-raw",
    )
    assert stored.body["node"]["payload"]["brief_id"] is None
    assert stored.body["node"]["payload"]["rule_profile"] == "lax"


def test_plan_parents_brief_recipe_plan_review_only(hub: Hub) -> None:
    brief_id = _brief(hub)
    graph = WorkGraph(hub.log, hub.project_id)
    op_id = graph.add_node("op", AGENT).id
    art_id = graph.add_node("artifact", AGENT).id
    before = hub.counts()
    for parent, kind in ((op_id, "op"), (art_id, "artifact")):
        error = _refused("wrong_parent_kind", lambda parent=parent: _plan(hub, parent, key=parent))
        assert error.details["kind"] == kind
        assert error.details["allowed"] == ["brief", "plan", "recipe", "review"]
    _refused("unknown_parent", lambda: _plan(hub, "nd_missing", key="m"))
    _refused(
        "unknown_project",
        lambda: writes.submit_plan(
            hub.log,
            hub.store,
            "prj_missing",
            plan_fixture("short_clean"),
            AGENT,
            derives_from=brief_id,
            idempotency_key="u",
        ),
    )
    assert hub.counts() == before  # refused before the artifact is stored


def test_plan_derived_from_a_review_resolves_the_brief_through_its_subject(hub: Hub) -> None:
    brief_id = _brief(hub)
    first = _plan(hub, brief_id).body["node"]["id"]
    render = hub.store.put(b"render", actor=AGENT, mime="video/mp4", project_id=hub.project_id)
    review = request_review(
        hub.log, hub.store, hub.project_id, first, [render], AGENT, title="cut v1"
    )
    decide(hub.log, hub.project_id, review.id, "reject", "tighten the end", HUMAN)
    revised = _plan(hub, review.id, key="plan-2", title="cut v2")
    data = revised.body["node"]["payload"]
    assert data["derives_from"] == review.id and data["brief_id"] == brief_id
    assert revised.body["node"]["title"] == "cut v2"
    assert revised.body["edge"]["dst"] == review.id


def test_brief_behind_handles_untyped_and_odd_ancestry(hub: Hub) -> None:
    graph = WorkGraph(hub.log, hub.project_id)
    raw_recipe = graph.add_node("recipe", AGENT, data={"recipe": "x"})
    bad_plan = graph.add_node("plan", AGENT, data={"brief_id": "nd_x"})
    lonely_review = graph.add_node("review", AGENT, data={"subject": "nd_nowhere"})
    review_of_review = graph.add_node("review", AGENT, data={"subject": lonely_review.id})
    op = graph.add_node("op", AGENT)
    pointing = graph.add_node(
        "recipe", AGENT, data={"schema": "aar.recipe/1", "recipe": "r", "brief_id": op.id}
    )
    graph = WorkGraph(hub.log, hub.project_id)
    for node in (raw_recipe, bad_plan, lonely_review, review_of_review, op, pointing):
        assert writes.brief_behind(graph, graph.node(node.id)) is None


def test_plan_title_must_be_one_line(hub: Hub) -> None:
    brief_id = _brief(hub)
    error = _refused("invalid_input", lambda: _plan(hub, brief_id, title="a\nb"))
    assert error.details["field"] == "title"
    # A title derived from the plan's free-text timeline name is made one line.
    plan = plan_fixture("short_clean")
    plan["timeline"] = "Short 03\nsecond line\x07"
    stored = writes.submit_plan(
        hub.log,
        hub.store,
        hub.project_id,
        plan,
        AGENT,
        derives_from=brief_id,
        idempotency_key="derived",
    )
    version = EditPlan.model_validate(plan).version
    assert stored.body["node"]["title"] == f"plan v{version}: Short 03 second line "


def test_plan_replay_and_conflict(hub: Hub) -> None:
    brief_id = _brief(hub)
    first = _plan(hub, brief_id)
    before = hub.counts()
    again = _plan(hub, brief_id)
    assert again.replayed and {**again.to_dict(), "replayed": False} == first.to_dict()
    assert hub.counts() == before
    # The same plan with its fps spelled as a string is the same normalised request.
    spelled = plan_fixture("short_clean")
    spelled["fps"] = "24000/1001"
    assert writes.submit_plan(
        hub.log,
        hub.store,
        hub.project_id,
        spelled,
        AGENT,
        derives_from=brief_id,
        idempotency_key="plan-1",
    ).replayed
    _refused("idempotency_conflict", lambda: _plan(hub, brief_id, "short_broken"))
    _refused("idempotency_conflict", lambda: _plan(hub, brief_id, title="renamed"))
    assert hub.counts() == before


@pytest.mark.parametrize("crash_at", [1, 2, 3, 4])
def test_submit_plan_survives_a_crash_after_each_event(
    hub: Hub, monkeypatch: pytest.MonkeyPatch, crash_at: int
) -> None:
    brief_id = _brief(hub)
    before = hub.counts()
    with monkeypatch.context() as patched:
        seen = crash_after_appends(patched, crash_at)
        with pytest.raises(Crash):
            _plan(hub, brief_id)
    order = ["artifact.added", "graph.node_added", "graph.edge_added", "hub.write"]
    assert seen == order[:crash_at]
    survived = hub.counts() - before
    # The artifact is recorded outside the transaction; the graph writes are all-or-nothing.
    assert survived == ({"artifact.added": 1} if crash_at >= 1 else {})
    result = _plan(hub, brief_id)
    assert result.replayed is False
    after = hub.counts()
    assert after - before == {
        "artifact.added": 1,
        "graph.node_added": 1,
        "graph.edge_added": 1,
        "hub.write": 1,
    }
    assert _plan(hub, brief_id).replayed
    assert hub.counts() == after


# -- set_own_status -------------------------------------------------------------------------------


def test_creator_moves_its_own_node_and_a_retry_replays(hub: Hub) -> None:
    brief_id = _brief(hub)
    result = writes.set_own_status(
        hub.log, hub.project_id, brief_id, "active", AGENT, idempotency_key="s"
    )
    assert result.to_dict() == {
        "ok": True,
        "op": "node.set_status",
        "replayed": False,
        "project_id": hub.project_id,
        "node_id": brief_id,
        "kind": "brief",
        "from": "pending",
        "to": "active",
    }
    graph = WorkGraph(hub.log, hub.project_id)
    assert graph.node(brief_id).status == "active"
    assert graph.last_status_change(brief_id)[1] == AGENT
    before = hub.counts()
    again = writes.set_own_status(
        hub.log, hub.project_id, brief_id, "active", AGENT, idempotency_key="s"
    )
    assert again.replayed and again.body == result.body
    assert hub.counts() == before
    _refused(
        "idempotency_conflict",
        lambda: writes.set_own_status(
            hub.log, hub.project_id, brief_id, "done", AGENT, idempotency_key="s"
        ),
    )


def test_agents_move_only_their_own_brief_recipe_plan_and_op_nodes(hub: Hub) -> None:
    brief_id = _brief(hub)
    recipe_id = _recipe(hub, brief_id)
    plan_id = _plan(hub, recipe_id).body["node"]["id"]
    graph = WorkGraph(hub.log, hub.project_id)
    op_id = graph.add_node("op", AGENT).id
    art_id = graph.add_node("artifact", AGENT).id
    for index, node_id in enumerate((brief_id, recipe_id, plan_id, op_id)):
        before = hub.counts()
        error = _refused(
            "not_creator",
            lambda node_id=node_id, index=index: writes.set_own_status(
                hub.log, hub.project_id, node_id, "active", OTHER, idempotency_key=f"o{index}"
            ),
        )
        assert error.details["creator"] == {"kind": "agent", "id": "agent-1"}
        assert hub.counts() == before
        writes.set_own_status(
            hub.log, hub.project_id, node_id, "active", AGENT, idempotency_key=f"a{index}"
        )
    error = _refused(
        "kind_not_allowed",
        lambda: writes.set_own_status(
            hub.log, hub.project_id, art_id, "active", AGENT, idempotency_key="art"
        ),
    )
    assert error.details["kind"] == "artifact"
    # A node actor is not a human either.
    node_actor = Actor("node", "node-1")
    node_op = graph.add_node("op", node_actor).id
    writes.set_own_status(
        hub.log, hub.project_id, node_op, "active", node_actor, idempotency_key="n1"
    )
    _refused(
        "not_creator",
        lambda: writes.set_own_status(
            hub.log, hub.project_id, op_id, "done", node_actor, idempotency_key="n2"
        ),
    )


def test_a_human_moves_any_non_review_node(hub: Hub) -> None:
    brief_id = _brief(hub)
    art_id = WorkGraph(hub.log, hub.project_id).add_node("artifact", AGENT).id
    for index, node_id in enumerate((brief_id, art_id)):
        result = writes.set_own_status(
            hub.log, hub.project_id, node_id, "cancelled", HUMAN, idempotency_key=f"h{index}"
        )
        assert result.body["to"] == "cancelled"


def test_reviews_are_never_moved_here(hub: Hub) -> None:
    brief_id = _brief(hub)
    render = hub.store.put(b"render", actor=AGENT, project_id=hub.project_id)
    review = request_review(
        hub.log, hub.store, hub.project_id, brief_id, [render], AGENT, title="look"
    )
    for index, actor in enumerate((AGENT, HUMAN)):
        for to in ("done", "cancelled"):
            _refused(
                "protected_review",
                lambda actor=actor, to=to, index=index: writes.set_own_status(
                    hub.log,
                    hub.project_id,
                    review.id,
                    to,
                    actor,
                    idempotency_key=f"rv{index}{to}",
                ),
            )
    assert WorkGraph(hub.log, hub.project_id).node(review.id).status == "active"


def test_transitions_follow_the_graph_table(hub: Hub) -> None:
    brief_id = _brief(hub)
    error = _refused(
        "illegal_transition",
        lambda: writes.set_own_status(
            hub.log, hub.project_id, brief_id, "done", AGENT, idempotency_key="t1"
        ),
    )
    assert error.details["from"] == "pending" and error.details["to"] == "done"
    assert error.details["allowed"] == ["active", "blocked", "cancelled"]
    writes.set_own_status(
        hub.log, hub.project_id, brief_id, "cancelled", AGENT, idempotency_key="t2"
    )
    error = _refused(
        "illegal_transition",
        lambda: writes.set_own_status(
            hub.log, hub.project_id, brief_id, "active", AGENT, idempotency_key="t3"
        ),
    )
    assert error.details["allowed"] == [] and "final" in error.fix_hint
    blocked_plan = _plan(hub, _brief(hub, key="b2"), "short_broken").body["node"]["id"]
    assert (
        writes.set_own_status(
            hub.log, hub.project_id, blocked_plan, "active", AGENT, idempotency_key="t4"
        ).body["from"]
        == "blocked"
    )


def test_status_refusals_for_unknown_things(hub: Hub) -> None:
    brief_id = _brief(hub)
    _refused(
        "unknown_node",
        lambda: writes.set_own_status(
            hub.log, hub.project_id, "nd_missing", "active", AGENT, idempotency_key="u1"
        ),
    )
    _refused(
        "unknown_project",
        lambda: writes.set_own_status(
            hub.log, "prj_missing", brief_id, "active", AGENT, idempotency_key="u2"
        ),
    )
    for bad in ("finished", None, 3):
        _refused(
            "invalid_input",
            lambda bad=bad: writes.set_own_status(
                hub.log, hub.project_id, brief_id, bad, AGENT, idempotency_key="u3"
            ),
        )
    for bad in ("", "x" * 129, None):
        _refused(
            "invalid_input",
            lambda bad=bad: writes.set_own_status(
                hub.log, hub.project_id, bad, "active", AGENT, idempotency_key="u4"
            ),
        )


@pytest.mark.parametrize("crash_at", [1, 2])
def test_set_own_status_survives_a_crash_after_each_event(
    hub: Hub, monkeypatch: pytest.MonkeyPatch, crash_at: int
) -> None:
    brief_id = _brief(hub)
    before = hub.counts()
    with monkeypatch.context() as patched:
        seen = crash_after_appends(patched, crash_at)
        with pytest.raises(Crash):
            writes.set_own_status(
                hub.log, hub.project_id, brief_id, "active", AGENT, idempotency_key="s"
            )
    assert seen == ["graph.node_status_changed", "hub.write"][:crash_at]
    assert hub.counts() == before
    writes.set_own_status(hub.log, hub.project_id, brief_id, "active", AGENT, idempotency_key="s")
    assert hub.counts() - before == {"graph.node_status_changed": 1, "hub.write": 1}
    assert WorkGraph(hub.log, hub.project_id).node(brief_id).status == "active"


# -- typed reads ------------------------------------------------------------------------


def test_brief_view_and_plan_view(hub: Hub) -> None:
    brief_id = _brief(hub)
    recipe_id = _recipe(hub, brief_id)
    plan = _plan(hub, recipe_id)
    plan_id = plan.body["node"]["id"]
    view = writes.brief_view(hub.log, hub.project_id, brief_id)
    assert view["typed"] is True and view["brief"]["show"] == "Acme Comedy"
    assert view["creator"] == {"kind": "agent", "id": "agent-1"}
    assert view["derived"] == [{"node_id": recipe_id, "kind": "recipe", "status": "pending"}]
    assert writes.brief_view(hub.log, hub.project_id, plan_id) is None
    assert writes.brief_view(hub.log, hub.project_id, "nd_missing") is None
    raw = WorkGraph(hub.log, hub.project_id).add_node("brief", HUMAN, data={"show": "x"}).id
    assert writes.brief_view(hub.log, hub.project_id, raw)["typed"] is False

    pview = writes.plan_view(hub.log, hub.store, hub.project_id, plan_id)
    assert pview["typed"] is True and pview["plan_error"] is None
    assert EditPlan.model_validate(pview["plan"]) == EditPlan.model_validate(
        plan_fixture("short_clean")
    )
    assert (
        writes.plan_view(hub.log, hub.store, hub.project_id, plan_id, include_plan=False)["plan"]
        is None
    )
    assert writes.plan_view(hub.log, hub.store, hub.project_id, brief_id) is None

    sha = plan.body["artifact_id"]
    path = hub.root / "artifacts" / sha[:2] / sha[2:4] / sha
    path.write_bytes(b"{}")
    corrupt = writes.plan_view(hub.log, hub.store, hub.project_id, plan_id)
    assert corrupt["plan"] is None and "does not match" in corrupt["plan_error"]
    path.unlink()
    missing = writes.plan_view(hub.log, hub.store, hub.project_id, plan_id)
    assert missing["plan"] is None and "not in the artifact store" in missing["plan_error"]


def test_write_error_codes_are_closed() -> None:
    with pytest.raises(ValueError):
        WriteError("made_up", "m", "h")
    error = WriteError("unknown_node", "m", "h", node_id="nd_1")
    assert error.to_dict() == {
        "code": "unknown_node",
        "message": "m",
        "fix_hint": "h",
        "details": {"node_id": "nd_1"},
    }
