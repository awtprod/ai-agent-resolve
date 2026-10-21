"""S13g1: ``aar hub write ...`` and ``aar hub facts show``, through the real CLI."""

from __future__ import annotations

import io
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from write_support import Hub, brief_data, make_hub, plan_fixture

from aar.cli import main
from aar.hub.graph import WorkGraph


@pytest.fixture
def hub(tmp_path: Path) -> Iterator[Hub]:
    made = make_hub(tmp_path)
    yield made
    made.log.close()


def _stdin(monkeypatch: pytest.MonkeyPatch, data: bytes) -> None:
    monkeypatch.setattr("sys.stdin", io.TextIOWrapper(io.BytesIO(data)))


def _write(
    hub: Hub,
    capsys: pytest.CaptureFixture[str],
    command: str,
    request: Any,
    *,
    key: str = "k1",
    actor: str = "agent:agent-1",
    path: Path | None = None,
    extra: tuple[str, ...] = (),
) -> tuple[int, Any, Any]:
    """Run ``aar hub write <command>`` with ``request`` in a file; return (exit, out, err)."""
    path = path or hub.root / f"{command}-{key}.json"
    path.write_text(request if isinstance(request, str) else json.dumps(request), "utf-8")
    args = [
        "hub",
        "write",
        command,
        "--db",
        str(hub.root / "hub.db"),
        "--actor",
        actor,
        "--idempotency-key",
        key,
    ]
    if command == "plan":
        args += ["--artifacts", str(hub.root / "artifacts")]
    code = main([*args, *extra, str(path)])
    captured = capsys.readouterr()
    out = json.loads(captured.out) if captured.out.strip() else None
    err = json.loads(captured.err.strip().splitlines()[-1]) if captured.err.strip() else None
    return code, out, err


def _brief(hub: Hub, capsys: pytest.CaptureFixture[str]) -> str:
    code, out, _ = _write(hub, capsys, "brief", {"brief": brief_data(hub.project_id)}, key="b")
    assert code == 0
    return out["node"]["id"]


def test_write_brief_from_a_file_and_replay(hub: Hub, capsys: pytest.CaptureFixture[str]) -> None:
    code, out, err = _write(hub, capsys, "brief", {"brief": brief_data(hub.project_id)})
    assert (code, err) == (0, None)
    assert out["ok"] is True and out["op"] == "brief.create" and out["replayed"] is False
    code, again, _ = _write(hub, capsys, "brief", {"brief": brief_data(hub.project_id)})
    assert code == 0 and again["replayed"] is True and again["node"] == out["node"]
    graph = WorkGraph(hub.log, hub.project_id)
    assert [n.kind for n in graph.nodes()] == ["brief"]


def test_write_reads_stdin_by_default(
    hub: Hub, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _stdin(monkeypatch, json.dumps({"brief": brief_data(hub.project_id)}).encode())
    code = main(
        [
            "hub",
            "write",
            "brief",
            "--db",
            str(hub.root / "hub.db"),
            "--actor",
            "agent:agent-1",
            "--idempotency-key",
            "s",
        ]
    )
    out = json.loads(capsys.readouterr().out)
    assert code == 0 and out["node"]["kind"] == "brief"
    _stdin(monkeypatch, json.dumps({"brief": brief_data(hub.project_id)}).encode())
    code = main(
        [
            "hub",
            "write",
            "brief",
            "--db",
            str(hub.root / "hub.db"),
            "--actor",
            "agent:agent-1",
            "--idempotency-key",
            "s",
            "-",
        ]
    )
    assert code == 0 and json.loads(capsys.readouterr().out)["replayed"] is True


def test_every_write_command(hub: Hub, capsys: pytest.CaptureFixture[str]) -> None:
    pid = hub.project_id
    brief_id = _brief(hub, capsys)
    code, recipe, _ = _write(
        hub,
        capsys,
        "recipe",
        {"project_id": pid, "recipe": {"recipe": "ccn-short", "brief_id": brief_id}},
    )
    assert code == 0 and recipe["node"]["kind"] == "recipe"
    code, plan, _ = _write(
        hub,
        capsys,
        "plan",
        {
            "project_id": pid,
            "derives_from": recipe["node"]["id"],
            "plan": plan_fixture("short_clean"),
            "title": "first cut",
        },
        key="p1",
    )
    assert code == 0 and plan["node"]["status"] == "pending" and plan["artifact_id"]
    code, blocked, _ = _write(
        hub,
        capsys,
        "plan",
        {"project_id": pid, "derives_from": brief_id, "plan": plan_fixture("short_broken")},
        key="k2",
    )
    assert code == 1 and blocked["node"]["status"] == "blocked"  # stored, but lint errors
    code, replay, _ = _write(
        hub,
        capsys,
        "plan",
        {"project_id": pid, "derives_from": brief_id, "plan": plan_fixture("short_broken")},
        key="k2",
    )
    assert code == 1 and replay["replayed"] is True
    code, moved, _ = _write(
        hub,
        capsys,
        "status",
        {"project_id": pid, "node_id": brief_id, "status": "active"},
        key="s1",
    )
    assert code == 0 and moved["to"] == "active"
    code, fact, _ = _write(
        hub,
        capsys,
        "fact",
        {
            "project_id": pid,
            "key": "audio.gain.A1",
            "value": {"db": -3.0},
            "source": {"recipe_node": recipe["node"]["id"]},
        },
        key="f1",
    )
    assert code == 0 and fact["fact"]["value"] == {"db": -3.0}
    code, gone, _ = _write(
        hub,
        capsys,
        "retract",
        {"project_id": pid, "key": "audio.gain.A1", "reason": "wrong track"},
        key="r1",
    )
    assert code == 0 and gone["retracted"]["reason"] == "wrong track"


def test_a_refused_write_prints_a_structured_error_and_exits_2(
    hub: Hub, capsys: pytest.CaptureFixture[str]
) -> None:
    brief_id = _brief(hub, capsys)
    # illegal_transition carries details whose keys must never leak into the exit status.
    code, out, err = _write(
        hub, capsys, "status", {"project_id": hub.project_id, "node_id": brief_id, "status": "done"}
    )
    assert (code, out) == (2, None)
    assert err["error"]["code"] == "illegal_transition"
    assert set(err["error"]) == {"code", "message", "fix_hint", "details"}
    assert err["error"]["details"]["from"] == "pending"
    code, _, err = _write(
        hub,
        capsys,
        "status",
        {"project_id": hub.project_id, "node_id": brief_id, "status": "active"},
        actor="agent:agent-2",
        key="k2",
    )
    assert code == 2 and err["error"]["code"] == "not_creator"
    code, _, err = _write(
        hub, capsys, "brief", {"brief": brief_data(hub.project_id, kind="gif")}, key="k3"
    )
    assert code == 2 and err["error"]["code"] == "invalid_brief"
    assert err["error"]["details"]["errors"][0]["loc"] == "kind"
    code, _, err = _write(
        hub,
        capsys,
        "plan",
        {"project_id": hub.project_id, "derives_from": brief_id, "plan": {"kind": "short"}},
        key="k4",
    )
    assert code == 2 and err["error"]["code"] == "invalid_plan"
    code, _, err = _write(
        hub, capsys, "fact", {"project_id": hub.project_id, "key": "mood.x", "value": {}}, key="k5"
    )
    assert code == 2 and err["error"]["code"] == "unknown_namespace"
    code, _, err = _write(
        hub, capsys, "brief", {"brief": brief_data(hub.project_id, show="Other")}, key="b"
    )
    assert code == 2 and err["error"]["code"] == "idempotency_conflict"


@pytest.mark.parametrize(
    ("request_text", "fragment"),
    [
        ("not json", "not JSON"),
        ('{"brief": NaN}', "NaN is not JSON"),
        ("[1, 2]", "must be a JSON object"),
        ("{}", "missing ['brief']"),
        ('{"brief": {}, "extra": 1}', "unknown ['extra']"),
        ("[" * 100_000 + "]" * 100_000, "nested too deeply"),
    ],
)
def test_unusable_requests_are_invalid_input(
    hub: Hub, capsys: pytest.CaptureFixture[str], request_text: str, fragment: str
) -> None:
    code, out, err = _write(hub, capsys, "brief", request_text)
    assert code == 2 and out is None
    assert err["error"]["code"] == "invalid_input"
    assert fragment in err["error"]["message"]


def test_requests_that_are_too_big_or_not_utf8_are_invalid_input(
    hub: Hub, capsys: pytest.CaptureFixture[str]
) -> None:
    big = hub.root / "big.json"
    big.write_bytes(b'{"brief": "' + b"x" * (2 * 1024 * 1024) + b'"}')
    code, _, err = _write(hub, capsys, "brief", big.read_text(), path=big)
    assert code == 2 and "over" in err["error"]["message"]
    bad = hub.root / "bad.json"
    bad.write_bytes(b'{"brief": "\xff"}')
    code = main(
        [
            "hub",
            "write",
            "brief",
            "--db",
            str(hub.root / "hub.db"),
            "--actor",
            "agent:a",
            "--idempotency-key",
            "k",
            str(bad),
        ]
    )
    err = json.loads(capsys.readouterr().err.strip().splitlines()[-1])
    assert code == 2 and "UTF-8" in err["error"]["message"]
    code = main(
        [
            "hub",
            "write",
            "brief",
            "--db",
            str(hub.root / "hub.db"),
            "--actor",
            "agent:a",
            "--idempotency-key",
            "k",
            str(hub.root / "missing.json"),
        ]
    )
    assert code == 2


@pytest.mark.parametrize("actor", ["agent", "robot:x", "agent:", "agent:bad id"])
def test_bad_actors_are_invalid_input(
    hub: Hub, capsys: pytest.CaptureFixture[str], actor: str
) -> None:
    code, _, err = _write(hub, capsys, "brief", {"brief": brief_data(hub.project_id)}, actor=actor)
    assert code == 2 and err["error"]["code"] == "invalid_input"
    assert hub.counts()["hub.write"] == 0


def test_missing_database_and_store_are_reported_not_created(
    hub: Hub, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    request = tmp_path / "req.json"
    request.write_text(json.dumps({"brief": brief_data(hub.project_id)}))
    code = main(
        [
            "hub",
            "write",
            "brief",
            "--db",
            str(tmp_path / "nope.db"),
            "--actor",
            "agent:a",
            "--idempotency-key",
            "k",
            str(request),
        ]
    )
    err = json.loads(capsys.readouterr().err.strip().splitlines()[-1])
    assert code == 2 and err["error"]["code"] == "database_unavailable"
    assert not (tmp_path / "nope.db").exists()
    brief_id = _brief(hub, capsys)
    plan_request = tmp_path / "plan.json"
    plan_request.write_text(
        json.dumps(
            {
                "project_id": hub.project_id,
                "derives_from": brief_id,
                "plan": plan_fixture("short_clean"),
            }
        )
    )
    code = main(
        [
            "hub",
            "write",
            "plan",
            "--db",
            str(hub.root / "hub.db"),
            "--artifacts",
            str(tmp_path / "no-store"),
            "--actor",
            "agent:a",
            "--idempotency-key",
            "p",
            str(plan_request),
        ]
    )
    err = json.loads(capsys.readouterr().err.strip().splitlines()[-1])
    assert code == 2 and err["error"]["code"] == "artifact_store_unavailable"
    assert not (tmp_path / "no-store").exists()


def test_facts_show(hub: Hub, capsys: pytest.CaptureFixture[str]) -> None:
    for key, value in (
        ("audio.gain.A1", {"db": -1.0}),
        ("sync.offset.wide", {"frames": 3, "fps": {"num": 25}, "relative_to": "A cam"}),
        ("path.map.pc", {"from": "E:\\", "to": "/srv"}),
    ):
        assert (
            _write(
                hub,
                capsys,
                "fact",
                {"project_id": hub.project_id, "key": key, "value": value},
                key=key,
            )[0]
            == 0
        )
    _write(
        hub,
        capsys,
        "retract",
        {"project_id": hub.project_id, "key": "path.map.pc", "reason": "moved"},
        key="r",
    )
    db = str(hub.root / "hub.db")
    assert main(["hub", "facts", "show", "--db", db, "--project", hub.project_id]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert [f["key"] for f in shown["facts"]] == ["audio.gain.A1", "sync.offset.wide"]
    assert [f["key"] for f in shown["retracted"]] == ["path.map.pc"]
    assert (
        main(["hub", "facts", "show", "--db", db, "--project", hub.project_id, "--prefix", "sync."])
        == 0
    )
    shown = json.loads(capsys.readouterr().out)
    assert [f["key"] for f in shown["facts"]] == ["sync.offset.wide"] and shown["retracted"] == []
    assert main(["hub", "facts", "show", "--db", db, "--project", "prj_missing"]) == 1
    err = json.loads(capsys.readouterr().err.strip())
    assert err["error"]["code"] == "unknown_project"
    assert (
        main(
            ["hub", "facts", "show", "--db", str(hub.root / "nope.db"), "--project", hub.project_id]
        )
        == 2
    )
