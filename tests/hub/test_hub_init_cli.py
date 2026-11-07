"""``aar hub init``: create a fresh hub database and artifact store, never overwrite (S15a)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aar.cli import main
from aar.hub.events import EventLog


def _init_args(tmp_path: Path) -> list[str]:
    return [
        "hub",
        "init",
        "--db",
        str(tmp_path / "hub.db"),
        "--artifacts",
        str(tmp_path / "artifacts"),
    ]


def test_init_creates_a_usable_hub(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(_init_args(tmp_path)) == 0
    out = json.loads(capsys.readouterr().out)
    assert out == {"db": str(tmp_path / "hub.db"), "artifacts": str(tmp_path / "artifacts")}
    assert (tmp_path / "hub.db").is_file()
    assert (tmp_path / "artifacts").is_dir()
    # The database is a real, append-only hub log that the read surface can open as-is.
    EventLog(str(tmp_path / "hub.db"), create=False, read_only=True).close()


def test_init_refuses_an_existing_database(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "hub.db").write_text("already here")
    code = main(_init_args(tmp_path))
    assert code == 2
    err = json.loads(capsys.readouterr().err)
    assert err["error"]["code"] == "db_exists"
    # The artifact store was never created, because the db check comes first and bails.
    assert not (tmp_path / "artifacts").exists()


def test_init_refuses_an_existing_artifact_store(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "artifacts").mkdir()
    code = main(_init_args(tmp_path))
    assert code == 2
    err = json.loads(capsys.readouterr().err)
    assert err["error"]["code"] == "artifacts_exist"
    # The database must not be left behind when the artifact path already exists.
    assert not (tmp_path / "hub.db").exists()
