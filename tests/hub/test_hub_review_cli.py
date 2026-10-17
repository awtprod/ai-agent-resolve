"""Tests for ``aar hub review ...`` and ``aar hub console`` argument handling, via the real CLI."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from aar.cli import main
from aar.hub.artifacts import ArtifactStore
from aar.hub.events import Actor, EventLog
from aar.hub.graph import WorkGraph
from aar.hub.registry import Registry
from aar.hub.reviews import decide, get_review, list_reviews

HUMAN = Actor("human", "operator-1")
AGENT = Actor("agent", "agent-1")
UNSTORED_RENDER = "ab" * 32  # well-formed artifact id with no content behind it


@pytest.fixture(autouse=True)
def _never_serve(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every console call here is an error path: fail fast if one ever starts serving.

    Without this, a regressed guard would bind a real port and hang the suite.
    """

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("aar hub console started serving on an error path")

    monkeypatch.setattr("uvicorn.run", refuse)


@dataclass
class Hub:
    """A hub with one project, one work-graph node to review and one stored render."""

    log: EventLog
    db: Path
    artifacts: Path
    project_id: str
    subject_id: str
    render_id: str


@pytest.fixture
def hub(tmp_path: Path) -> Iterator[Hub]:
    db = tmp_path / "hub.db"
    log = EventLog(db)
    project = Registry(log).create_project("demo-show", HUMAN)
    subject = WorkGraph(log, project.id).add_node("op", AGENT, title="assemble first cut")
    artifacts = tmp_path / "artifacts"
    render_id = ArtifactStore(artifacts, log).put(
        b"render-bytes", actor=AGENT, mime="video/mp4", project_id=project.id
    )
    yield Hub(log, db, artifacts, project.id, subject.id, render_id)
    log.close()


def _run(
    capsys: pytest.CaptureFixture[str], *argv: str
) -> tuple[int, dict[str, Any] | None, dict[str, Any] | None]:
    """Run ``aar <argv>``; return the exit code and the JSON written to stdout and stderr."""
    capsys.readouterr()
    code = main(list(argv))
    captured = capsys.readouterr()
    out = json.loads(captured.out) if captured.out.strip() else None
    err = json.loads(captured.err) if captured.err.strip() else None
    return code, out, err


def _request_argv(hub: Hub, *extra: str, fps: str | None = "24") -> list[str]:
    argv = [
        "hub",
        "review",
        "request",
        "--db",
        str(hub.db),
        "--artifacts",
        str(hub.artifacts),
        "--project",
        hub.project_id,
        "--subject",
        hub.subject_id,
        "--render",
        hub.render_id,
        "--title",
        "first cut",
        "--actor",
        "agent:agent-1",
    ]
    if fps is not None:
        argv += ["--fps", fps]
    return [*argv, *extra]


def _request(hub: Hub, capsys: pytest.CaptureFixture[str], *, fps: str | None = "24") -> str:
    code, out, err = _run(capsys, *_request_argv(hub, fps=fps))
    assert (code, err) == (0, None)
    assert out is not None
    return str(out["id"])


def _note_argv(hub: Hub, review_id: str, *extra: str) -> list[str]:
    return [
        "hub",
        "review",
        "note",
        "--db",
        str(hub.db),
        "--project",
        hub.project_id,
        "--review",
        review_id,
        "--text",
        "trim",
        "--actor",
        "agent:agent-1",
        *extra,
    ]


# -- request -----------------------------------------------------------------


def test_request_opens_an_active_review_of_the_subject(
    hub: Hub, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, err = _run(capsys, *_request_argv(hub, "--timeline", "main"))

    assert code == 0
    assert err is None
    assert out is not None
    assert out["status"] == "active"
    assert out["pending"] is True
    assert out["subject"] == hub.subject_id
    assert out["renders"] == [hub.render_id]
    assert out["title"] == "first cut"
    assert out["timeline"] == "main"
    assert out["fps"] == 24.0
    assert out["requested_by"] == "agent:agent-1"
    assert out["project_id"] == hub.project_id
    assert out["notes"] == []
    assert out["decision"] is None
    # It was written to the shared log, not just printed.
    stored = get_review(hub.log, hub.project_id, out["id"])
    assert stored is not None
    assert stored.status == "active"


def test_request_retry_with_the_same_key_returns_the_same_review(
    hub: Hub, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = _request_argv(hub, "--idempotency-key", "req-1")

    first = _run(capsys, *argv)
    again = _run(capsys, *argv)

    assert first[0] == again[0] == 0
    assert first[1] is not None
    assert again[1] is not None
    assert again[1]["id"] == first[1]["id"]
    assert len(list_reviews(hub.log)) == 1


def test_request_key_reused_for_a_different_title_is_an_idempotency_conflict(
    hub: Hub, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run(capsys, *_request_argv(hub, "--idempotency-key", "req-1"))[0] == 0
    changed = _request_argv(hub, "--idempotency-key", "req-1")
    changed[changed.index("--title") + 1] = "a different cut"

    code, out, err = _run(capsys, *changed)

    assert code == 2
    assert out is None
    assert err is not None
    assert err["error"]["code"] == "idempotency_conflict"
    assert len(list_reviews(hub.log)) == 1


# -- note --------------------------------------------------------------------


def test_note_derives_the_timecode_from_the_frame_and_the_review_fps(
    hub: Hub, capsys: pytest.CaptureFixture[str]
) -> None:
    review_id = _request(hub, capsys)

    code, out, err = _run(capsys, *_note_argv(hub, review_id, "--frame", "48"))

    assert (code, err) == (0, None)
    assert out is not None
    assert out["timecode"] == "00:00:02:00"  # frame 48 at 24 fps
    assert out["frame"] == 48
    assert out["text"] == "trim"
    assert out["review_id"] == review_id
    assert out["author"] == {"kind": "agent", "id": "agent-1"}


def test_note_timecode_flag_overrides_the_derived_label(
    hub: Hub, capsys: pytest.CaptureFixture[str]
) -> None:
    review_id = _request(hub, capsys)

    code, out, _ = _run(
        capsys, *_note_argv(hub, review_id, "--frame", "48", "--timecode", "01:00:02:00")
    )

    assert code == 0
    assert out is not None
    assert out["timecode"] == "01:00:02:00"
    assert out["frame"] == 48


def test_note_with_neither_a_timecode_nor_a_review_fps_is_invalid_input(
    hub: Hub, capsys: pytest.CaptureFixture[str]
) -> None:
    review_id = _request(hub, capsys, fps=None)

    code, out, err = _run(capsys, *_note_argv(hub, review_id, "--frame", "48"))

    assert code == 2
    assert out is None
    assert err is not None
    assert err["error"]["code"] == "invalid_input"
    assert "timecode" in err["error"]["message"]
    stored = get_review(hub.log, hub.project_id, review_id)
    assert stored is not None
    assert stored.notes == ()


def test_note_with_a_malformed_timecode_is_invalid_input(
    hub: Hub, capsys: pytest.CaptureFixture[str]
) -> None:
    review_id = _request(hub, capsys)

    code, _, err = _run(capsys, *_note_argv(hub, review_id, "--frame", "48", "--timecode", "2s"))

    assert code == 2
    assert err is not None
    assert err["error"]["code"] == "invalid_input"


def test_note_on_an_unknown_review_reports_review_not_found(
    hub: Hub, capsys: pytest.CaptureFixture[str]
) -> None:
    code, _, err = _run(
        capsys, *_note_argv(hub, "nd_missing", "--frame", "1", "--timecode", "00:00:00:01")
    )

    assert code == 2
    assert err is not None
    assert err["error"]["code"] == "ReviewNotFound"


# -- list and show -----------------------------------------------------------


def test_list_and_show_follow_a_review_through_note_and_decision(
    hub: Hub, capsys: pytest.CaptureFixture[str]
) -> None:
    review_id = _request(hub, capsys)
    assert _run(capsys, *_note_argv(hub, review_id, "--frame", "48"))[0] == 0
    db = str(hub.db)

    code, listing, _ = _run(capsys, "hub", "review", "list", "--db", db)
    assert code == 0
    assert listing is not None
    assert [r["id"] for r in listing["reviews"]] == [review_id]
    assert listing["reviews"][0]["status"] == "active"

    code, shown, _ = _run(
        capsys, "hub", "review", "show", "--db", db, "--project", hub.project_id, review_id
    )
    assert code == 0
    assert shown is not None
    assert [(n["frame"], n["text"], n["timecode"]) for n in shown["notes"]] == [
        (48, "trim", "00:00:02:00")
    ]
    assert shown["decision"] is None

    decide(hub.log, hub.project_id, review_id, "approve", "ok", HUMAN)

    # Open reviews only by default: the decided one is gone, but --all keeps it.
    _, open_only, _ = _run(capsys, "hub", "review", "list", "--db", db)
    assert open_only == {"reviews": []}
    _, everything, _ = _run(capsys, "hub", "review", "list", "--db", db, "--all")
    assert everything is not None
    assert [(r["id"], r["status"]) for r in everything["reviews"]] == [(review_id, "done")]
    decision = everything["reviews"][0]["decision"]
    assert decision["decision"] == "approve"
    assert decision["reason"] == "ok"
    assert decision["by"] == {"kind": "human", "id": "operator-1"}

    _, decided, _ = _run(
        capsys, "hub", "review", "show", "--db", db, "--project", hub.project_id, review_id
    )
    assert decided is not None
    assert decided["decision"]["decision"] == "approve"
    assert decided["pending"] is False


def test_list_can_be_limited_to_one_project(hub: Hub, capsys: pytest.CaptureFixture[str]) -> None:
    review_id = _request(hub, capsys)
    db = str(hub.db)

    _, mine, _ = _run(capsys, "hub", "review", "list", "--db", db, "--project", hub.project_id)
    _, other, _ = _run(capsys, "hub", "review", "list", "--db", db, "--project", "prj_other")

    assert mine is not None
    assert [r["id"] for r in mine["reviews"]] == [review_id]
    assert other == {"reviews": []}


def test_show_of_a_missing_review_exits_1_with_review_not_found(
    hub: Hub, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, err = _run(
        capsys,
        "hub",
        "review",
        "show",
        "--db",
        str(hub.db),
        "--project",
        hub.project_id,
        "nd_missing",
    )

    assert code == 1
    assert out is None
    assert err is not None
    assert err["error"]["code"] == "review_not_found"


# -- errors are JSON on stderr with exit 2 -----------------------------------


def test_request_with_an_unknown_render_id_reports_unknown_artifact(
    hub: Hub, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = _request_argv(hub)
    argv[argv.index("--render") + 1] = UNSTORED_RENDER

    code, out, err = _run(capsys, *argv)

    assert code == 2
    assert out is None
    assert err is not None
    assert err["error"]["code"] == "UnknownArtifact"
    assert UNSTORED_RENDER in err["error"]["message"]
    assert list_reviews(hub.log) == []  # no review was created


def test_request_with_a_malformed_render_id_is_invalid_input(
    hub: Hub, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = _request_argv(hub)
    argv[argv.index("--render") + 1] = "not-an-artifact-id"

    code, _, err = _run(capsys, *argv)

    assert code == 2
    assert err is not None
    assert err["error"]["code"] == "invalid_input"


@pytest.mark.parametrize(
    ("flag", "value", "code_name"),
    [
        ("--project", "prj_missing", "ProjectNotFound"),
        ("--subject", "nd_missing", "SubjectNotFound"),
    ],
)
def test_request_for_an_unknown_project_or_subject_is_reported_by_name(
    hub: Hub, capsys: pytest.CaptureFixture[str], flag: str, value: str, code_name: str
) -> None:
    argv = _request_argv(hub)
    argv[argv.index(flag) + 1] = value

    code, _, err = _run(capsys, *argv)

    assert code == 2
    assert err is not None
    assert err["error"]["code"] == code_name
    assert list_reviews(hub.log) == []


def _command_argv(command: str, artifacts: Path) -> list[str]:
    """Valid arguments for a review subcommand, apart from ``--db``."""
    if command == "request":
        return [
            "--artifacts",
            str(artifacts),
            "--project",
            "prj_demo",
            "--subject",
            "nd_demo",
            "--render",
            UNSTORED_RENDER,
            "--title",
            "first cut",
            "--actor",
            "agent:agent-1",
        ]
    if command == "note":
        return [
            "--project",
            "prj_demo",
            "--review",
            "nd_demo",
            "--frame",
            "1",
            "--timecode",
            "00:00:00:01",
            "--text",
            "trim",
            "--actor",
            "agent:agent-1",
        ]
    if command == "show":
        return ["--project", "prj_demo", "nd_demo"]
    return []


@pytest.mark.parametrize("command", ["request", "note", "list", "show"])
def test_a_missing_database_is_reported_and_no_file_is_created(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], command: str
) -> None:
    db = tmp_path / "missing.db"
    argv = _command_argv(command, tmp_path / "artifacts")

    code, out, err = _run(capsys, "hub", "review", command, "--db", str(db), *argv)

    assert code == 2
    assert out is None
    assert err is not None
    assert err["error"]["code"] == "database_unavailable"
    assert str(db) in err["error"]["message"]
    assert list(tmp_path.iterdir()) == []  # not the database, a -wal/-shm, or the artifact dir


def test_a_database_that_is_not_an_event_log_is_reported_as_unavailable(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db = tmp_path / "other.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE unrelated (x INTEGER)")
    conn.commit()
    conn.close()

    code, _, err = _run(capsys, "hub", "review", "list", "--db", str(db))

    assert code == 2
    assert err is not None
    assert err["error"]["code"] == "database_unavailable"


@pytest.mark.parametrize("actor", ["agent", "robot:r2", ":x", "agent:"])
def test_a_bad_actor_is_invalid_input_and_writes_nothing(
    hub: Hub, capsys: pytest.CaptureFixture[str], actor: str
) -> None:
    argv = _request_argv(hub)
    argv[argv.index("--actor") + 1] = actor
    seq = hub.log.latest_seq()

    code, out, err = _run(capsys, *argv)

    assert code == 2
    assert out is None
    assert err is not None
    assert err["error"]["code"] == "invalid_input"
    assert hub.log.latest_seq() == seq


def test_a_bad_actor_without_a_colon_says_what_the_format_is(
    hub: Hub, capsys: pytest.CaptureFixture[str]
) -> None:
    review_id = _request(hub, capsys)
    argv = _note_argv(hub, review_id, "--frame", "48")
    argv[argv.index("--actor") + 1] = "agent"

    code, _, err = _run(capsys, *argv)

    assert code == 2
    assert err is not None
    assert err["error"]["code"] == "invalid_input"
    assert "KIND:ID" in err["error"]["message"]


# -- console -----------------------------------------------------------------


def test_console_refuses_a_non_loopback_host_without_allow_remote(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db = tmp_path / "hub.db"

    code, out, err = _run(
        capsys,
        "hub",
        "console",
        "--db",
        str(db),
        "--artifacts",
        str(tmp_path / "artifacts"),
        "--host",
        "0.0.0.0",
    )

    assert code == 2
    assert out is None
    assert err is not None
    assert err["error"]["code"] == "remote_bind_refused"
    assert "--allow-remote" in err["error"]["message"]
    # The guard fires before the database or the artifact store is touched.
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "extra", [[], ["--host", "0.0.0.0", "--allow-remote"]], ids=["loopback", "remote"]
)
def test_console_with_a_missing_database_exits_before_serving(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], extra: list[str]
) -> None:
    db = tmp_path / "hub.db"

    code, _, err = _run(
        capsys,
        "hub",
        "console",
        "--db",
        str(db),
        "--artifacts",
        str(tmp_path / "artifacts"),
        *extra,
    )

    assert code == 2
    assert err is not None
    assert err["error"]["code"] == "database_unavailable"
    assert list(tmp_path.iterdir()) == []


def test_console_rejects_a_malformed_operator_before_opening_the_database(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, _, err = _run(
        capsys,
        "hub",
        "console",
        "--db",
        str(tmp_path / "hub.db"),
        "--artifacts",
        str(tmp_path / "artifacts"),
        "--operator",
        "not a valid operator!",
    )

    assert code == 2
    assert err is not None
    assert err["error"]["code"] == "invalid_input"
    assert list(tmp_path.iterdir()) == []


# -- unusable --db and --artifacts paths --------------------------------------------


def _unusable_db(tmp_path: Path, kind: str) -> str:
    """Make a ``--db`` value that is not a hub database, of the given kind."""
    if kind == "directory":
        path = tmp_path / "hub.db"
        path.mkdir()
        return str(path)
    if kind == "not-sqlite":
        path = tmp_path / "hub.db"
        path.write_text("this is not a sqlite database")
        return str(path)
    if kind == "memory":
        return ":memory:"
    raise AssertionError(kind)


@pytest.mark.parametrize("kind", ["directory", "not-sqlite", "memory"])
@pytest.mark.parametrize("command", ["request", "note", "list", "show", "console"])
def test_an_unusable_database_path_is_a_json_error_not_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], command: str, kind: str
) -> None:
    db = _unusable_db(tmp_path, kind)
    before = sorted(p.name for p in tmp_path.iterdir())
    group = ["hub", "console"] if command == "console" else ["hub", "review", command]
    argv = (
        ["--artifacts", str(tmp_path / "artifacts")]
        if command == "console"
        else _command_argv(command, tmp_path / "artifacts")
    )

    code, out, err = _run(capsys, *group, "--db", db, *argv)

    assert code == 2
    assert out is None
    assert err is not None
    assert err["error"]["code"] == "database_unavailable"
    # Nothing was created beside it: no -wal/-shm and no artifact store directory.
    assert sorted(p.name for p in tmp_path.iterdir()) == before


def test_a_write_command_does_not_switch_a_foreign_sqlite_file_to_wal(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db = tmp_path / "other.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE unrelated (x INTEGER)")
    conn.commit()
    conn.close()
    argv = _command_argv("request", tmp_path / "artifacts")

    code, _, err = _run(capsys, "hub", "review", "request", "--db", str(db), *argv)

    assert code == 2
    assert err is not None and err["error"]["code"] == "database_unavailable"
    conn = sqlite3.connect(db)
    try:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
    finally:
        conn.close()


@pytest.mark.parametrize("command", ["request", "console"])
def test_a_missing_artifact_store_is_reported_and_not_created(
    hub: Hub, tmp_path: Path, capsys: pytest.CaptureFixture[str], command: str
) -> None:
    missing = tmp_path / "no-such-store"
    if command == "console":
        argv = ["hub", "console", "--db", str(hub.db), "--artifacts", str(missing)]
    else:
        argv = _request_argv(hub)
        argv[argv.index("--artifacts") + 1] = str(missing)
    seq = hub.log.latest_seq()

    code, out, err = _run(capsys, *argv)

    assert code == 2
    assert out is None
    assert err is not None
    assert err["error"]["code"] == "artifact_store_unavailable"
    assert str(missing) in err["error"]["message"]
    assert not missing.exists()
    assert hub.log.latest_seq() == seq


def test_withdraw_by_the_requester_cancels_the_review_and_others_are_refused(
    hub: Hub, capsys: pytest.CaptureFixture[str]
) -> None:
    review_id = _request(hub, capsys)
    base = ["hub", "review", "withdraw", "--db", str(hub.db), "--project", hub.project_id]
    base += ["--review", review_id, "--reason", "superseded by v2"]

    code, _, err = _run(capsys, *base, "--actor", "agent:agent-2")
    assert code == 2
    assert err is not None and err["error"]["code"] == "NotRequester"

    code, out, err = _run(capsys, *base, "--actor", "agent:agent-1")
    assert (code, err) == (0, None)
    assert out is not None and out["reason"] == "superseded by v2"
    review = get_review(hub.log, hub.project_id, review_id)
    assert review is not None and review.status == "cancelled" and not review.pending
