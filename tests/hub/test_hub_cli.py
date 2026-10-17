"""Tests for the ``aar hub`` CLI wiring (the serve entry point, not a bound server)."""

from __future__ import annotations

from pathlib import Path

import pytest

from aar.cli import main
from aar.hub.events import EventLog


def _hub_paths(tmp_path: Path) -> None:
    """Create an initialised (empty) hub database and artifact store the server can open."""
    EventLog(tmp_path / "hub.db").close()
    (tmp_path / "artifacts").mkdir()


def _stub_run(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
    """Replace MCPServer.run so serve never binds a socket; record its kwargs instead."""
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        "aar.hub.mcp_server.MCPServer.run",
        lambda self, **kwargs: calls.append(kwargs),
        raising=True,
    )
    return calls


def _serve_args(tmp_path: Path, *extra: str) -> list[str]:
    return [
        "hub",
        "serve",
        "--db",
        str(tmp_path / "hub.db"),
        "--artifacts",
        str(tmp_path / "artifacts"),
        *extra,
    ]


def test_serve_accepts_loopback_hosts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _hub_paths(tmp_path)
    calls = _stub_run(monkeypatch)
    for host in ("127.0.0.1", "localhost", "::1"):
        calls.clear()
        assert main(_serve_args(tmp_path, "--host", host)) == 0
        assert calls[-1]["host"] == host
        assert calls[-1]["transport"] == "streamable-http"


def test_serve_accepts_non_loopback_with_allow_remote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _hub_paths(tmp_path)
    calls = _stub_run(monkeypatch)
    assert main(_serve_args(tmp_path, "--host", "0.0.0.0", "--allow-remote")) == 0
    assert calls[-1]["host"] == "0.0.0.0"


def test_serve_refuses_non_loopback_without_allow_remote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls = _stub_run(monkeypatch)
    code = main(_serve_args(tmp_path, "--host", "0.0.0.0"))
    assert code == 2
    assert "allow-remote" in capsys.readouterr().err
    # The guard returns before serving or touching the filesystem.
    assert calls == []
    assert not (tmp_path / "hub.db").exists()


def test_serve_reports_a_missing_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls = _stub_run(monkeypatch)
    code = main(_serve_args(tmp_path, "--host", "127.0.0.1"))
    assert code == 2
    db = str(tmp_path / "hub.db")
    assert capsys.readouterr().err.strip() == f"aar hub serve: no event log database at {db!r}"
    assert calls == []
    assert not (tmp_path / "hub.db").exists()  # nothing was created


def test_serve_reports_a_directory_as_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls = _stub_run(monkeypatch)
    (tmp_path / "hub.db").mkdir()  # a directory where a file was expected
    code = main(_serve_args(tmp_path, "--host", "127.0.0.1"))
    assert code == 2
    db = str(tmp_path / "hub.db")
    assert capsys.readouterr().err.startswith(
        f"aar hub serve: cannot open event log database at {db!r}: "
    )
    assert calls == []


def test_serve_reports_a_non_sqlite_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls = _stub_run(monkeypatch)
    (tmp_path / "hub.db").write_text("this is not a sqlite database")
    code = main(_serve_args(tmp_path, "--host", "127.0.0.1"))
    assert code == 2
    db = str(tmp_path / "hub.db")
    assert capsys.readouterr().err.startswith(
        f"aar hub serve: cannot open event log database at {db!r}: "
    )
    assert calls == []


def test_serve_reports_a_memory_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "artifacts").mkdir()
    calls = _stub_run(monkeypatch)
    code = main(
        [
            "hub",
            "serve",
            "--db",
            ":memory:",
            "--artifacts",
            str(tmp_path / "artifacts"),
            "--host",
            "127.0.0.1",
        ]
    )
    assert code == 2
    assert (
        capsys.readouterr().err.strip()
        == "aar hub serve: a read-only event log needs a file-backed database, not ':memory:'"
    )
    assert calls == []


def test_serve_reports_a_schemaless_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "hub.db").touch()  # a 0-byte file is a valid but schema-less sqlite db
    (tmp_path / "artifacts").mkdir()
    calls = _stub_run(monkeypatch)
    code = main(_serve_args(tmp_path, "--host", "127.0.0.1"))
    assert code == 2
    db = str(tmp_path / "hub.db")
    assert capsys.readouterr().err.startswith(
        f"aar hub serve: event log at {db!r} is missing append-only triggers"
    )
    assert calls == []


def test_serve_reports_a_missing_artifacts_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    EventLog(tmp_path / "hub.db").close()  # valid db, but no artifacts dir
    calls = _stub_run(monkeypatch)
    code = main(_serve_args(tmp_path, "--host", "127.0.0.1"))
    assert code == 2
    art = tmp_path / "artifacts"
    assert capsys.readouterr().err.strip() == f"aar hub serve: no artifact store directory at {art}"
    assert calls == []
    assert not art.exists()  # not created by the failed open


def test_hub_requires_a_subcommand(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["hub"]) == 2
    assert "usage: aar hub" in capsys.readouterr().err


def test_top_level_still_prints_version() -> None:
    assert main(["--version"]) == 0


def test_serve_closes_log_when_artifacts_open_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    EventLog(tmp_path / "hub.db").close()  # a valid db, but the artifacts dir is missing
    import aar.hub.events as events_mod

    real_close = events_mod.EventLog.close
    closed: list = []

    def spy(self) -> None:
        closed.append(self)
        real_close(self)

    monkeypatch.setattr(events_mod.EventLog, "close", spy)
    _stub_run(monkeypatch)
    code = main(_serve_args(tmp_path, "--host", "127.0.0.1"))
    assert code == 2
    assert closed  # the read-only log opened by serve was closed before bailing out
