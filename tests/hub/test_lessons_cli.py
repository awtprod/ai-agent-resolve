"""``aar hub lessons list`` and ``show`` (S23a), through the real CLI: read-only, JSON out."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from write_support import AGENT, HUMAN, Hub

from aar.cli import main
from aar.hub import lessons


def _run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, Any, Any]:
    """Run ``aar <argv>``; return the exit code and the JSON on stdout and on stderr."""
    capsys.readouterr()
    code = main(list(argv))
    captured = capsys.readouterr()
    out = json.loads(captured.out) if captured.out.strip() else None
    err = json.loads(captured.err) if captured.err.strip() else None
    return code, out, err


@pytest.fixture
def seeded(hub: Hub) -> dict[str, lessons.Lesson]:
    """One lesson in each state and scope kind, plus a forged event the fold skips."""
    acme = f"project:{hub.project_id}"
    made = {
        "proposed": lessons.propose(hub.log, "Proposed.", "client:ccn", AGENT, why="w"),
        "accepted": lessons.add(hub.log, "Accepted, global.", "global", HUMAN),
        "project": lessons.add(hub.log, "Accepted, acme.", acme, HUMAN),
    }
    dropped = lessons.propose(hub.log, "Dropped.", acme, AGENT)
    made["dropped"] = lessons.drop(hub.log, dropped.id, HUMAN, reason="No.")
    retired = lessons.add(hub.log, "Retired.", "client:ccn", HUMAN)
    made["retired"] = lessons.retire(hub.log, retired.id, HUMAN)
    hub.log.append(lessons.EVENT_ACCEPTED, AGENT, {"lesson_id": made["proposed"].id})
    return made


def _db(hub: Hub) -> str:
    return str(hub.root / "hub.db")


def _ids(out: dict[str, Any]) -> list[str]:
    return [row["id"] for row in out["lessons"]]


def test_list_with_its_filters(
    hub: Hub, seeded: dict[str, lessons.Lesson], capsys: pytest.CaptureFixture[str]
) -> None:
    db = _db(hub)
    code, out, err = _run(capsys, "hub", "lessons", "list", "--db", db)
    assert code == 0 and err is None
    every = [seeded[k].id for k in ("proposed", "accepted", "project", "dropped", "retired")]
    assert _ids(out) == every  # oldest first, every state by default
    assert out["lessons"][0] == seeded["proposed"].to_dict()
    assert out["skipped"]["count"] == 1  # the agent's forged accept
    assert "not a human" in out["skipped"]["events"][0]["reason"]

    for state in lessons.STATES:
        code, out, _ = _run(capsys, "hub", "lessons", "list", "--db", db, "--state", state)
        expected = [lesson.id for lesson in seeded.values() if lesson.state == state]
        assert code == 0 and _ids(out) == expected, state
    for scope, keys in (
        ("global", ["accepted"]),
        ("client:ccn", ["proposed", "retired"]),
        (f"project:{hub.project_id}", ["project", "dropped"]),
        (f"project:{hub.other_project_id}", []),
    ):
        code, out, _ = _run(capsys, "hub", "lessons", "list", "--db", db, "--scope", scope)
        assert code == 0 and _ids(out) == [seeded[k].id for k in keys], scope
    argv = ("--state", "accepted", "--scope", f"project:{hub.project_id}")
    code, out, _ = _run(capsys, "hub", "lessons", "list", "--db", db, *argv)
    assert _ids(out) == [seeded["project"].id]


def test_show_and_its_errors(
    hub: Hub, seeded: dict[str, lessons.Lesson], capsys: pytest.CaptureFixture[str]
) -> None:
    db = _db(hub)
    for lesson in seeded.values():
        code, out, err = _run(capsys, "hub", "lessons", "show", "--db", db, lesson.id)
        assert code == 0 and err is None and out == lesson.to_dict()
    code, out, err = _run(capsys, "hub", "lessons", "show", "--db", db, "les_0000000000000000")
    assert code == 1 and out is None
    assert err["error"]["code"] == "unknown_lesson" and err["error"]["fix_hint"]
    assert err["error"]["details"] == {"lesson_id": "les_0000000000000000"}
    code, out, err = _run(capsys, "hub", "lessons", "show", "--db", db, "../hub.db")
    assert code == 2 and out is None and err["error"]["code"] == "invalid_input"
    code, out, err = _run(capsys, "hub", "lessons", "list", "--db", db, "--scope", "client:../x")
    assert code == 2 and out is None and err["error"]["code"] == "invalid_input"
    assert err["error"]["details"] == {"field": "scope"}


@pytest.mark.parametrize("command", [["list"], ["show", "les_0000000000000000"]])
def test_the_read_only_open_never_creates_a_database(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], command: list[str]
) -> None:
    missing = tmp_path / "nope.db"
    code, out, err = _run(capsys, "hub", "lessons", command[0], "--db", str(missing), *command[1:])
    assert code == 2 and out is None
    assert err["error"]["code"] == "database_unavailable"
    assert list(tmp_path.iterdir()) == []


def test_the_open_is_read_only(
    hub: Hub,
    seeded: dict[str, lessons.Lesson],
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aar.hub import cli as hub_cli

    opened: list[bool] = []
    real = hub_cli._open_existing

    def spy(db: str, *, read_only: bool = False) -> Any:
        opened.append(read_only)
        return real(db, read_only=read_only)

    monkeypatch.setattr(hub_cli, "_open_existing", spy)
    before = hub.log.latest_seq()
    _run(capsys, "hub", "lessons", "list", "--db", _db(hub))
    _run(capsys, "hub", "lessons", "show", "--db", _db(hub), seeded["accepted"].id)
    assert opened == [True, True]
    assert hub.log.latest_seq() == before


@pytest.mark.parametrize("verb", ["accept", "drop", "retire", "add"])
def test_there_is_no_cli_write_path(
    hub: Hub, capsys: pytest.CaptureFixture[str], verb: str
) -> None:
    """Human-only writes stay in the console: the CLI's actors are self-declared."""
    before = hub.log.latest_seq()
    capsys.readouterr()
    assert main(["hub", "lessons", verb, "--db", _db(hub), "--actor", "human:operator-1"]) == 2
    assert f"invalid choice: '{verb}'" in capsys.readouterr().err
    assert hub.log.latest_seq() == before
