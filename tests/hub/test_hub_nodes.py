"""Node enrolment and bearer-token authentication (S15a, brief item 2)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aar.cli import main
from aar.hub import nodes
from aar.hub.events import Actor, EventLog

HUMAN = Actor("human", "andrew")


def _log(tmp_path: Path) -> EventLog:
    return EventLog(str(tmp_path / "hub.db"))


# -- the nodes module ---------------------------------------------------------------------


def test_enroll_prints_a_token_and_stores_only_its_hash(tmp_path: Path) -> None:
    log = _log(tmp_path)
    node_id, token = nodes.enroll(log, "node-1", HUMAN)
    assert node_id == "node-1"
    assert token and isinstance(token, str)
    # The authenticated node comes from the token.
    assert nodes.authenticate(log, token) == "node-1"
    # The token itself appears nowhere in the log; only its sha256 does.
    raw = json.dumps([event.payload for event in log.iter_events(types=nodes._NODE_TYPES)])
    assert token not in raw
    assert nodes.token_sha256(token) in raw


def test_enroll_refuses_a_non_human_actor(tmp_path: Path) -> None:
    log = _log(tmp_path)
    for actor in (Actor("agent", "a1"), Actor("node", "node-1")):
        with pytest.raises(nodes.NodeError) as exc:
            nodes.enroll(log, "node-1", actor)
        assert exc.value.code == "actor_not_human"


def test_enroll_validates_the_node_id(tmp_path: Path) -> None:
    log = _log(tmp_path)
    for bad in ("", "  spaced  ", "bad\x00id", "x" * 200):
        with pytest.raises(nodes.NodeError) as exc:
            nodes.enroll(log, bad, HUMAN)
        assert exc.value.code == "invalid_node_id"


def test_enroll_twice_is_refused_unless_rotate(tmp_path: Path) -> None:
    log = _log(tmp_path)
    _, first = nodes.enroll(log, "node-1", HUMAN)
    with pytest.raises(nodes.NodeError) as exc:
        nodes.enroll(log, "node-1", HUMAN)
    assert exc.value.code == "already_enrolled"
    # --rotate replaces the token: the old one stops working on the very next request.
    _, second = nodes.enroll(log, "node-1", HUMAN, rotate=True)
    assert second != first
    assert nodes.authenticate(log, first) is None
    assert nodes.authenticate(log, second) == "node-1"


def test_a_revoked_token_stops_working_at_once(tmp_path: Path) -> None:
    log = _log(tmp_path)
    _, token = nodes.enroll(log, "node-1", HUMAN)
    assert nodes.authenticate(log, token) == "node-1"
    nodes.revoke(log, "node-1", HUMAN)
    # No cache outlives the revocation: the next lookup folds the log fresh and refuses.
    assert nodes.authenticate(log, token) is None


def test_revoke_requires_an_active_enrolment(tmp_path: Path) -> None:
    log = _log(tmp_path)
    with pytest.raises(nodes.NodeError) as exc:
        nodes.revoke(log, "node-1", HUMAN)
    assert exc.value.code == "not_enrolled"
    nodes.enroll(log, "node-1", HUMAN)
    nodes.revoke(log, "node-1", HUMAN)
    with pytest.raises(nodes.NodeError) as exc:
        nodes.revoke(log, "node-1", HUMAN)  # already revoked
    assert exc.value.code == "not_enrolled"


def test_authenticate_rejects_unknown_and_empty_tokens(tmp_path: Path) -> None:
    log = _log(tmp_path)
    nodes.enroll(log, "node-1", HUMAN)
    assert nodes.authenticate(log, "not-a-real-token") is None
    assert nodes.authenticate(log, "") is None
    assert nodes.authenticate(log, None) is None


def test_two_nodes_each_authenticate_to_themselves(tmp_path: Path) -> None:
    log = _log(tmp_path)
    _, token_a = nodes.enroll(log, "node-a", HUMAN)
    _, token_b = nodes.enroll(log, "node-b", HUMAN)
    assert nodes.authenticate(log, token_a) == "node-a"
    assert nodes.authenticate(log, token_b) == "node-b"
    nodes.revoke(log, "node-a", HUMAN)
    assert nodes.authenticate(log, token_a) is None
    assert nodes.authenticate(log, token_b) == "node-b"  # unaffected


def test_list_shows_times_and_hash_prefix_only(tmp_path: Path) -> None:
    log = _log(tmp_path)
    _, token = nodes.enroll(log, "node-1", HUMAN)
    nodes.enroll(log, "node-2", HUMAN)
    nodes.revoke(log, "node-2", HUMAN)
    records = {r.node_id: r for r in nodes.list_nodes(log)}
    assert set(records) == {"node-1", "node-2"}
    assert records["node-1"].active and records["node-1"].revoked_at is None
    assert not records["node-2"].active and records["node-2"].revoked_at is not None
    # Only a prefix of the hash is exposed, never the token or the full hash.
    full = nodes.token_sha256(token)
    assert records["node-1"].hash_prefix == full[: nodes.HASH_PREFIX_LEN]
    assert records["node-1"].hash_prefix != full


# -- the CLI ------------------------------------------------------------------------------


def _db(tmp_path: Path) -> str:
    EventLog(str(tmp_path / "hub.db")).close()
    return str(tmp_path / "hub.db")


def test_cli_enroll_revoke_list(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    db = _db(tmp_path)
    assert main(["hub", "node", "enroll", "--db", db, "node-1", "--actor", "human:andrew"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["node_id"] == "node-1"
    token = out["token"]
    assert token
    # The token authenticates against the same db.
    assert nodes.authenticate(EventLog(db, create=False, read_only=True), token) == "node-1"

    # A non-human actor is refused with exit 2.
    assert main(["hub", "node", "enroll", "--db", db, "node-2", "--actor", "agent:a1"]) == 2
    err = json.loads(capsys.readouterr().err)
    assert err["error"]["code"] == "actor_not_human"

    # list shows the node with a hash prefix, never the token.
    assert main(["hub", "node", "list", "--db", db]) == 0
    listing = json.loads(capsys.readouterr().out)
    (record,) = listing["nodes"]
    assert record["node_id"] == "node-1"
    assert token not in json.dumps(listing)
    assert record["hash_prefix"] == nodes.token_sha256(token)[: nodes.HASH_PREFIX_LEN]

    # revoke, then the token no longer authenticates.
    assert main(["hub", "node", "revoke", "--db", db, "node-1", "--actor", "human:andrew"]) == 0
    capsys.readouterr()
    assert nodes.authenticate(EventLog(db, create=False, read_only=True), token) is None


def test_cli_rotate_prints_a_fresh_token(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db = _db(tmp_path)
    main(["hub", "node", "enroll", "--db", db, "node-1", "--actor", "human:andrew"])
    first = json.loads(capsys.readouterr().out)["token"]
    # Without --rotate a second enrol is refused.
    assert main(["hub", "node", "enroll", "--db", db, "node-1", "--actor", "human:andrew"]) == 2
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "already_enrolled"
    # With --rotate it succeeds and the old token dies.
    code = main(
        ["hub", "node", "enroll", "--db", db, "node-1", "--actor", "human:andrew", "--rotate"]
    )
    assert code == 0
    second = json.loads(capsys.readouterr().out)["token"]
    assert second != first
    read = EventLog(db, create=False, read_only=True)
    assert nodes.authenticate(read, first) is None
    assert nodes.authenticate(read, second) == "node-1"
