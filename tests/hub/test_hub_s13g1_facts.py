"""S13g1: the per-project facts store (aar.hub.facts). None of this exists on the base."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from write_support import AGENT, HUMAN, OTHER, Crash, Hub, crash_after_appends, make_hub

from aar.hub import facts, writes
from aar.hub.events import Actor
from aar.hub.graph import WorkGraph
from aar.hub.writes import WriteError

OFFSET = {"frames": 12, "fps": {"num": 24000, "den": 1001}, "relative_to": "close camera"}
SHA = "ab" * 32

VALID: list[tuple[str, dict[str, Any]]] = [
    ("sync.offset.wide", OFFSET),
    ("sync.offset.B-cam_2", {"frames": -3, "fps": {"num": 25}, "relative_to": "A1 board mix"}),
    ("audio.gain.A1", {"db": -3.5}),
    ("audio.gain.A5", {"db": 24}),
    ("audio.gain.A6", {"db": -96.0}),
    ("template.trap.outro-gap", {"text": "The outro starts 2 frames late.\nNudge it."}),
    ("template.id.lower-third", {"unique_id": "8b3c1e2f-1d2e-4f5a"}),
    ("path.map.edit-pc", {"from": "E:\\CCN\\Media", "to": "/srv/media/ccn"}),
    ("render.hash.master", {"sha256": SHA, "bytes": 123_456_789}),
    ("qa.lufs.master", {"value": -22.1, "unit": "LUFS"}),
    ("qa.true-peak.short-1", {"value": -1.2, "unit": "dBTP"}),
    ("qa.longest_silence.short-1", {"value": 0.48, "unit": "s"}),
]


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


def _record(
    hub: Hub,
    key: str,
    value: dict[str, Any],
    *,
    ikey: str | None = None,
    actor: Actor = AGENT,
    **kw: Any,
) -> writes.WriteResult:
    return facts.record_fact(
        hub.log, hub.project_id, key, value, actor, idempotency_key=ikey or f"f-{key}", **kw
    )


@pytest.mark.parametrize(("key", "value"), VALID)
def test_every_namespace_records_and_reads_back(hub: Hub, key: str, value: dict[str, Any]) -> None:
    result = _record(hub, key, value)
    body = result.to_dict()
    assert body["ok"] is True and body["op"] == "facts.record" and body["replayed"] is False
    fact = body["fact"]
    assert fact["key"] == key and fact["namespace"] == facts.namespace_of(key).name
    view = facts.facts_view(hub.log, hub.project_id)
    current = view.get(key)
    assert current is not None and current.current.value == fact["value"]
    model = facts.namespace_of(key).model
    assert model.model_validate(fact["value"]) == model.model_validate(value)
    assert view.skipped == ()


def test_namespaces_are_a_closed_set() -> None:
    assert [ns.shape for ns in facts.NAMESPACES] == [
        "sync.offset.<angle>",
        "audio.gain.<track>",
        "template.trap.<id>",
        "template.id.<name>",
        "path.map.<name>",
        "render.hash.<deliverable>",
        "qa.<metric>.<deliverable>",
    ]


@pytest.mark.parametrize(
    "key", ["notes.free", "sync.drift.wide", "audio.level.A1", "qa", "sync", "", "x" * 201, 7]
)
def test_unknown_namespaces_and_unusable_keys_are_refused(hub: Hub, key: Any) -> None:
    with pytest.raises(WriteError) as info:
        _record(hub, key, {"db": 0}, ikey="k")
    assert info.value.code in {"unknown_namespace", "invalid_fact"}
    if (
        isinstance(key, str)
        and 0 < len(key) <= 200
        and "." in key
        and key.split(".")[0] not in ("sync", "audio", "qa")
    ):
        assert info.value.code == "unknown_namespace"
    assert facts.facts_view(hub.log, hub.project_id).facts == {}


def test_unknown_namespace_lists_the_known_ones(hub: Hub) -> None:
    error = _refused("unknown_namespace", lambda: _record(hub, "notes.free", {}, ikey="k"))
    assert error.details["namespaces"] == [ns.shape for ns in facts.NAMESPACES]


@pytest.mark.parametrize(
    "key",
    [
        "sync.offset.",
        "sync.offset.a.b",
        "sync.offset.has space",
        "sync.offset.-lead",
        "sync.offset." + "a" * 65,
        "audio.gain.A1\n",
        "qa.lufs",
        "qa.lufs.master.extra",
        "qa..master",
        "render.hash.mästër",
    ],
)
def test_bad_key_parts_are_invalid_facts(hub: Hub, key: str) -> None:
    error = _refused("invalid_fact", lambda: _record(hub, key, {"db": 0}, ikey="k"))
    assert error.details["field"] == "key"


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("sync.offset.wide", {**OFFSET, "frames": 1.5}),
        ("sync.offset.wide", {**OFFSET, "frames": True}),
        ("sync.offset.wide", {**OFFSET, "frames": 10_000_001}),
        ("sync.offset.wide", {**OFFSET, "fps": {"num": 0}}),
        ("sync.offset.wide", {**OFFSET, "fps": "24000/1001"}),
        ("sync.offset.wide", {**OFFSET, "relative_to": ""}),
        ("sync.offset.wide", {"frames": 1, "fps": {"num": 25}}),
        ("audio.gain.A1", {"db": 24.5}),
        ("audio.gain.A1", {"db": -96.1}),
        ("audio.gain.A1", {"db": float("nan")}),
        ("audio.gain.A1", {"db": float("inf")}),
        ("audio.gain.A1", {"db": "-3"}),
        ("audio.gain.A1", {"db": -3, "note": "x"}),
        ("template.trap.x", {"text": "t" * 2001}),
        ("template.trap.x", {"text": "bell\x07"}),
        ("template.id.x", {"unique_id": "has space"}),
        ("path.map.x", {"from_": "a", "to": "b"}),
        ("path.map.x", {"from": "a"}),
        ("path.map.x", {"from": "a\nb", "to": "b"}),
        ("render.hash.master", {"sha256": SHA.upper(), "bytes": 1}),
        ("render.hash.master", {"sha256": SHA, "bytes": -1}),
        ("render.hash.master", {"sha256": SHA, "bytes": 2**53 + 1}),
        ("qa.lufs.master", {"value": float("nan"), "unit": "LUFS"}),
        ("qa.lufs.master", {"value": 1e16, "unit": "LUFS"}),
        ("qa.lufs.master", {"value": -22, "unit": "two words"}),
        ("qa.lufs.master", {"value": -22, "unit": ""}),
        ("qa.lufs.master", {"value": -22}),
    ],
)
def test_values_that_do_not_fit_are_refused(hub: Hub, key: str, value: dict[str, Any]) -> None:
    error = _refused("invalid_fact", lambda: _record(hub, key, value, ikey="k"))
    assert error.details["errors"], error.to_dict()
    assert facts.facts_view(hub.log, hub.project_id).facts == {}


def test_value_that_is_not_an_object_is_refused(hub: Hub) -> None:
    error = _refused("invalid_fact", lambda: _record(hub, "audio.gain.A1", [1], ikey="k"))
    assert error.details["field"] == "value"


def test_last_record_wins_and_history_is_bounded(hub: Hub) -> None:
    for frames in range(15):
        _record(
            hub,
            "sync.offset.wide",
            {**OFFSET, "frames": frames},
            ikey=f"o{frames}",
            actor=AGENT if frames % 2 else HUMAN,
        )
    _record(hub, "audio.gain.A1", {"db": -1})
    view = facts.facts_view(hub.log, hub.project_id)
    fact = view.get("sync.offset.wide")
    assert fact.current.value["frames"] == 14
    assert fact.current.by == HUMAN
    assert len(fact.history) == facts.MAX_FACT_HISTORY == 10
    assert [change.value["frames"] for change in fact.history] == list(range(5, 15))
    shown = view.to_dict()
    assert [row["key"] for row in shown["facts"]] == ["audio.gain.A1", "sync.offset.wide"]
    assert shown["facts"][1]["set_by"] == {"kind": "human", "id": "operator-1"}
    assert shown["facts"][1]["set_at"] == fact.current.at


def test_retract_removes_the_current_value_and_keeps_history(hub: Hub) -> None:
    _record(hub, "path.map.edit-pc", {"from": "E:\\", "to": "/srv"})
    result = facts.retract_fact(
        hub.log, hub.project_id, "path.map.edit-pc", " drive moved ", HUMAN, idempotency_key="r"
    )
    assert result.to_dict()["retracted"]["reason"] == "drive moved"
    view = facts.facts_view(hub.log, hub.project_id)
    assert view.get("path.map.edit-pc") is None
    assert [f.key for f in view.retracted()] == ["path.map.edit-pc"]
    shown = view.to_dict()
    assert shown["facts"] == []
    assert shown["retracted"][0]["reason"] == "drive moved"
    assert [c["action"] for c in shown["retracted"][0]["history"]] == ["recorded", "retracted"]
    # Recording again makes it current again.
    _record(hub, "path.map.edit-pc", {"from": "F:\\", "to": "/srv"}, ikey="again")
    assert facts.facts_view(hub.log, hub.project_id).get("path.map.edit-pc") is not None
    _refused(
        "unknown_fact",
        lambda: facts.retract_fact(
            hub.log, hub.project_id, "audio.gain.A1", "never set", AGENT, idempotency_key="r2"
        ),
    )
    for bad in ("", "x" * 1001, 5):
        _refused(
            "invalid_input",
            lambda bad=bad: facts.retract_fact(
                hub.log, hub.project_id, "path.map.edit-pc", bad, AGENT, idempotency_key="r3"
            ),
        )
    _refused(
        "unknown_namespace",
        lambda: facts.retract_fact(
            hub.log, hub.project_id, "notes.x", "why", AGENT, idempotency_key="r4"
        ),
    )


def test_any_actor_kind_may_record(hub: Hub) -> None:
    for index, actor in enumerate((AGENT, HUMAN, Actor("node", "node-1"))):
        _record(hub, "audio.gain.A1", {"db": -float(index)}, ikey=f"a{index}", actor=actor)
    assert facts.facts_view(hub.log, hub.project_id).get("audio.gain.A1").current.by == Actor(
        "node", "node-1"
    )


def test_facts_are_scoped_to_their_project(hub: Hub) -> None:
    _record(hub, "audio.gain.A1", {"db": -1})
    assert facts.facts_view(hub.log, hub.other_project_id).facts == {}
    _refused(
        "unknown_project",
        lambda: facts.record_fact(
            hub.log, "prj_missing", "audio.gain.A1", {"db": 0}, AGENT, idempotency_key="k"
        ),
    )
    _refused(
        "unknown_project",
        lambda: facts.retract_fact(
            hub.log, "prj_missing", "audio.gain.A1", "x", AGENT, idempotency_key="k2"
        ),
    )


def test_idempotent_record_and_retract(hub: Hub) -> None:
    first = _record(hub, "audio.gain.A1", {"db": -3.0}, ikey="g")
    before = hub.counts()
    again = _record(hub, "audio.gain.A1", {"db": -3}, ikey="g")  # -3 normalises to -3.0
    assert again.replayed and {**again.to_dict(), "replayed": False} == first.to_dict()
    assert hub.counts() == before
    _refused("idempotency_conflict", lambda: _record(hub, "audio.gain.A1", {"db": -4}, ikey="g"))
    _refused("idempotency_conflict", lambda: _record(hub, "audio.gain.A2", {"db": -3}, ikey="g"))
    _refused(
        "idempotency_conflict",
        lambda: facts.record_fact(
            hub.log,
            hub.other_project_id,
            "audio.gain.A1",
            {"db": -3.0},
            AGENT,
            idempotency_key="g",
        ),
    )
    _refused(
        "idempotency_conflict",
        lambda: facts.retract_fact(
            hub.log, hub.project_id, "audio.gain.A1", "why", AGENT, idempotency_key="g"
        ),
    )
    # Another actor's key space is separate.
    assert _record(hub, "audio.gain.A1", {"db": -4}, ikey="g", actor=OTHER).replayed is False
    gone = facts.retract_fact(
        hub.log, hub.project_id, "audio.gain.A1", "wrong track", AGENT, idempotency_key="r"
    )
    replay = facts.retract_fact(
        hub.log, hub.project_id, "audio.gain.A1", "wrong track", AGENT, idempotency_key="r"
    )
    assert replay.replayed and replay.body == gone.body
    # A key used for a graph write cannot be reused for a fact, nor the other way round.
    writes.create_brief(
        hub.log,
        {
            "project_id": hub.project_id,
            "show": "s",
            "kind": "short",
            "recipe": "r",
            "rule_profile": "ccn",
            "deliverables": [{"name": "d", "kind": "short"}],
        },
        AGENT,
        idempotency_key="shared",
    )
    _refused(
        "idempotency_conflict", lambda: _record(hub, "audio.gain.A3", {"db": 0}, ikey="shared")
    )
    _record(hub, "audio.gain.A4", {"db": 0}, ikey="fact-first")
    _refused(
        "idempotency_conflict",
        lambda: writes.set_own_status(
            hub.log, hub.project_id, "nd_x", "active", AGENT, idempotency_key="fact-first"
        ),
    )


@pytest.mark.parametrize("op", ["record", "retract"])
def test_a_crash_after_the_event_leaves_nothing_and_a_retry_completes(
    hub: Hub, monkeypatch: pytest.MonkeyPatch, op: str
) -> None:
    if op == "retract":
        _record(hub, "audio.gain.A1", {"db": -1})

    def call() -> writes.WriteResult:
        if op == "record":
            return _record(hub, "audio.gain.A1", {"db": -2}, ikey="c")
        return facts.retract_fact(
            hub.log, hub.project_id, "audio.gain.A1", "why", AGENT, idempotency_key="c"
        )

    before = hub.counts()
    with monkeypatch.context() as patched:
        crash_after_appends(patched, 1)
        with pytest.raises(Crash):
            call()
    assert hub.counts() == before
    assert call().replayed is False
    assert call().replayed is True
    assert sum((hub.counts() - before).values()) == 1


def test_sources_must_exist(hub: Hub) -> None:
    graph = WorkGraph(hub.log, hub.project_id)
    recipe = graph.add_node("recipe", AGENT).id
    op = graph.add_node("op", AGENT).id
    stored = hub.store.put(b"qa report", actor=AGENT, project_id=hub.project_id)
    ok = _record(
        hub,
        "qa.lufs.master",
        {"value": -22.0, "unit": "LUFS"},
        source={"recipe_node": recipe, "artifact": stored},
    )
    assert ok.body["fact"]["source"] == {"recipe_node": recipe, "artifact": stored}
    _record(hub, "qa.lufs.short-1", {"value": -14.5, "unit": "LUFS"}, source={"artifact": stored})
    for index, source in enumerate(
        ({"recipe_node": op}, {"recipe_node": "nd_missing"}, {"artifact": "cd" * 32})
    ):
        _refused(
            "unknown_source",
            lambda source=source, index=index: _record(
                hub,
                "qa.lufs.master",
                {"value": -22.0, "unit": "LUFS"},
                ikey=f"s{index}",
                source=source,
            ),
        )
    for index, source in enumerate(({}, {"artifact": "short"}, {"recipe_node": ""}, {"x": 1}, [1])):
        _refused(
            "invalid_fact",
            lambda source=source, index=index: _record(
                hub,
                "qa.lufs.master",
                {"value": -22.0, "unit": "LUFS"},
                ikey=f"b{index}",
                source=source,
            ),
        )


def test_a_project_holds_at_most_max_keys(hub: Hub, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(facts, "MAX_FACT_KEYS", 3)
    for index in range(3):
        _record(hub, f"audio.gain.A{index}", {"db": 0})
    facts.retract_fact(hub.log, hub.project_id, "audio.gain.A0", "x", AGENT, idempotency_key="r")
    error = _refused("too_many_facts", lambda: _record(hub, "audio.gain.A9", {"db": 0}))
    assert error.details["limit"] == 3
    # An existing key (set or retracted) can still change.
    _record(hub, "audio.gain.A1", {"db": -1}, ikey="again")
    _record(hub, "audio.gain.A0", {"db": -1}, ikey="back")


def test_malformed_and_forged_events_are_skipped_never_raised(hub: Hub) -> None:
    _record(hub, "audio.gain.A1", {"db": -1})
    pid = hub.project_id
    forged = [
        ("facts.recorded", ["not", "an", "object"]),
        ("facts.recorded", {"key": "notes.x", "value": {}}),
        ("facts.recorded", {"key": "audio.gain.A1", "value": {"db": 99}}),
        ("facts.recorded", {"key": "audio.gain.A1", "value": {"db": "loud"}}),
        ("facts.recorded", {"key": "audio.gain.A1", "value": {"db": -1}, "source": {"x": 1}}),
        ("facts.recorded", {"key": ["list"], "value": {"db": -1}}),
        ("facts.retracted", {"key": "audio.gain.A2", "reason": "never set"}),
        ("facts.retracted", {"key": "audio.gain.A1"}),
        ("facts.retracted", {"key": "audio.gain.A1", "reason": "r" * 1001}),
        ("facts.retracted", {"key": "qa.only", "reason": "bad key"}),
    ]
    for event_type, payload in forged:
        hub.log.append(event_type, AGENT, payload, project_id=pid)
    view = facts.facts_view(hub.log, pid)
    assert view.get("audio.gain.A1").current.value == {"db": -1.0}
    assert len(view.skipped) == len(forged)
    assert all(skip.reason for skip in view.skipped)
    assert view.to_dict()["skipped"]["count"] == len(forged)
    # A valid retraction after the junk still applies.
    facts.retract_fact(hub.log, pid, "audio.gain.A1", "gone", AGENT, idempotency_key="r")
    assert facts.facts_view(hub.log, pid).get("audio.gain.A1") is None


def test_a_record_in_log_order_wins_even_with_out_of_order_timestamps(hub: Hub) -> None:
    pid = hub.project_id
    hub.log.append(
        "facts.recorded",
        AGENT,
        {"key": "audio.gain.A1", "value": {"db": -1.0}, "source": None},
        project_id=pid,
        ts="2030-01-01T00:00:00+00:00",
    )
    hub.log.append(
        "facts.recorded",
        HUMAN,
        {"key": "audio.gain.A1", "value": {"db": -2.0}, "source": None},
        project_id=pid,
        ts="2020-01-01T00:00:00+00:00",
    )
    assert facts.facts_view(hub.log, pid).get("audio.gain.A1").current.value == {"db": -2.0}
