"""``aar hub artifact get`` (S19a): read a stored artifact's bytes from the CLI, verified."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from write_support import AGENT, Hub, make_hub

from aar.cli import main


@pytest.fixture
def hub(tmp_path: Path) -> Iterator[Hub]:
    (tmp_path / "hub").mkdir()
    made = make_hub(tmp_path / "hub")
    yield made
    made.log.close()


def _get(hub: Hub, tmp_path: Path, artifact_id: str, *extra: str) -> list[str]:
    return [
        "hub",
        "artifact",
        "get",
        artifact_id,
        "--db",
        str(tmp_path / "hub" / "hub.db"),
        "--artifacts",
        str(tmp_path / "hub" / "artifacts"),
        *extra,
    ]


def _error(capsys: pytest.CaptureFixture[str]) -> str:
    return json.loads(capsys.readouterr().err.strip().splitlines()[-1])["error"]["code"]


def test_get_writes_the_bytes_and_never_overwrites(
    hub: Hub, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data = b"[0001.0-0002.0] So I went to the store\n"
    aid = hub.store.put(data, actor=AGENT, mime="text/plain", project_id=hub.project_id)
    out = tmp_path / "text.txt"
    assert main(_get(hub, tmp_path, aid, "--out", str(out), "--project", hub.project_id)) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["ok"] and printed["artifact_id"] == aid and printed["size"] == len(data)
    assert out.read_bytes() == data
    out.write_bytes(b"mine")
    assert main(_get(hub, tmp_path, aid, "--out", str(out))) == 2
    assert _error(capsys) == "output_exists"
    assert out.read_bytes() == b"mine"
    assert main(_get(hub, tmp_path, aid, "--out", str(out), "--force")) == 0
    capsys.readouterr()
    assert out.read_bytes() == data
    assert not [p for p in tmp_path.iterdir() if ".aar-get-" in p.name]


def test_get_to_stdout(
    hub: Hub, tmp_path: Path, capsysbinary: pytest.CaptureFixture[bytes]
) -> None:
    data = bytes(range(256)) * 10
    aid = hub.store.put(data, actor=AGENT, project_id=hub.project_id)
    assert main(_get(hub, tmp_path, aid)) == 0
    assert capsysbinary.readouterr().out == data


def test_refusals(hub: Hub, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    aid = hub.store.put(b"plan bytes", actor=AGENT, project_id=hub.project_id)
    out = str(tmp_path / "x")
    assert main(_get(hub, tmp_path, aid, "--out", out, "--project", "prj_other")) == 2
    assert _error(capsys) == "wrong_project"
    assert main(_get(hub, tmp_path, "f" * 64, "--out", out)) == 2
    assert _error(capsys) == "unknown_artifact"
    assert main(_get(hub, tmp_path, "../../etc/passwd", "--out", out)) == 2
    assert _error(capsys) == "invalid_input"
    stored = tmp_path / "hub" / "artifacts" / aid[:2] / aid[2:4] / aid
    stored.chmod(0o644)
    stored.write_bytes(b"tampered")
    assert main(_get(hub, tmp_path, aid, "--out", out)) == 2
    assert _error(capsys) == "artifact_corrupt"
    assert not Path(out).exists()
    stored.unlink()
    assert main(_get(hub, tmp_path, aid, "--out", out)) == 2
    assert _error(capsys) == "artifact_missing"
    bad_db = ["hub", "artifact", "get", aid, "--db", str(tmp_path / "none.db")]
    assert main([*bad_db, "--artifacts", str(tmp_path)]) == 2
    assert _error(capsys) == "database_unavailable"
