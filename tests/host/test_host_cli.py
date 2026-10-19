"""The ``aar host`` CLI: exit codes and JSON shapes, driven by a fake runner."""

from __future__ import annotations

import json

import pytest
from host_support import FakeRunner, sha256_upper

from aar.cli import main
from aar.host import cli as host_cli

REMOTE = "E:\\Show\\Review\\cut.mp4"


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("AAR_EDIT_SSH_ALIAS", "edit-pc")
    monkeypatch.setenv("AAR_EDIT_SSH_CONFIG", str(tmp_path / "ssh.cfg"))
    monkeypatch.setenv("AAR_EDIT_REMOTE_ROOTS", "E:\\Show")
    monkeypatch.setenv("AAR_EDIT_LOCAL_ROOTS", str(tmp_path))
    return tmp_path


def _use(monkeypatch, runner: FakeRunner) -> None:
    monkeypatch.setattr(host_cli, "_default_runner", lambda: runner)


def test_pull_json_success(env, monkeypatch, capsys) -> None:
    payload = b"a verified clip"
    _use(
        monkeypatch,
        FakeRunner(stat_result=(len(payload), sha256_upper(payload)), scp_payload=payload),
    )
    dest = env / "out.mp4"
    code = main(["host", "pull", REMOTE, str(dest), "--json"])
    assert code == 0
    body = json.loads(capsys.readouterr().out)
    assert body["ok"] is True
    assert body["op"] == "pull"
    assert body["size"] == len(payload)
    assert body["sha256"] == sha256_upper(payload)
    assert dest.read_bytes() == payload


def test_pull_human_success(env, monkeypatch, capsys) -> None:
    payload = b"x"
    _use(monkeypatch, FakeRunner(stat_result=(1, sha256_upper(payload)), scp_payload=payload))
    code = main(["host", "pull", REMOTE, str(env / "out.mp4")])
    assert code == 0
    assert "pulled 1 bytes" in capsys.readouterr().out


def test_pull_path_refused_is_exit_two(env, monkeypatch, capsys) -> None:
    _use(monkeypatch, FakeRunner())
    code = main(["host", "pull", "E:\\Show\\..\\secret", str(env / "out.mp4"), "--json"])
    assert code == 2
    body = json.loads(capsys.readouterr().out)
    assert body["ok"] is False
    assert body["error"]["reason"] == "dot-or-empty-component"


def test_size_mismatch_is_exit_one(env, monkeypatch, capsys) -> None:
    payload = b"hello"
    _use(monkeypatch, FakeRunner(stat_result=(999, sha256_upper(payload)), scp_payload=payload))
    code = main(["host", "pull", REMOTE, str(env / "out.mp4"), "--json"])
    assert code == 1
    body = json.loads(capsys.readouterr().out)
    assert body["error"]["type"] == "size-mismatch"


def test_missing_config_is_exit_two(tmp_path, monkeypatch, capsys) -> None:
    for name in (
        "AAR_EDIT_SSH_ALIAS",
        "AAR_EDIT_SSH_CONFIG",
        "AAR_EDIT_REMOTE_ROOTS",
        "AAR_EDIT_LOCAL_ROOTS",
    ):
        monkeypatch.delenv(name, raising=False)
    _use(monkeypatch, FakeRunner())
    code = main(["host", "pull", REMOTE, str(tmp_path / "out.mp4"), "--json"])
    assert code == 2
    assert json.loads(capsys.readouterr().out)["error"]["type"] == "config-error"


def test_hash_json(env, monkeypatch, capsys) -> None:
    _use(monkeypatch, FakeRunner(stat_result=(4096, "D" * 64)))
    code = main(["host", "hash", REMOTE, "--json"])
    assert code == 0
    body = json.loads(capsys.readouterr().out)
    assert body["op"] == "hash"
    assert body["size"] == 4096
    assert body["sha256"] == "D" * 64


def test_push_json_success(env, monkeypatch, capsys) -> None:
    src = env / "master.mov"
    src.write_bytes(b"content here")
    _use(monkeypatch, FakeRunner(stat_result=(len(b"content here"), sha256_upper(b"content here"))))
    code = main(["host", "push", str(src), "E:\\Show\\out.mov", "--json"])
    assert code == 0
    assert json.loads(capsys.readouterr().out)["op"] == "push"


def test_ps_runs_a_script_file(env, monkeypatch, capsys) -> None:
    script = env / "probe.ps1"
    script.write_text("Get-Date\n", encoding="utf-8")
    _use(monkeypatch, FakeRunner(ps_stdout=b"result\r\n"))
    code = main(["host", "ps", str(script)])
    assert code == 0
    assert capsys.readouterr().out == "result\n"


def test_ps_json_and_script_sent(env, monkeypatch, capsys) -> None:
    script = env / "probe.ps1"
    script.write_text("Write-Output 'hi'\n", encoding="utf-8")
    runner = FakeRunner(ps_stdout=b"hi\r\n")
    _use(monkeypatch, runner)
    code = main(["host", "ps", str(script), "--json"])
    assert code == 0
    body = json.loads(capsys.readouterr().out)
    assert body == {"ok": True, "op": "ps", "stdout": "hi\n"}
    assert runner.scripts == ["Write-Output 'hi'\n"]  # the file content, verbatim


def test_ps_missing_file_is_exit_two(env, monkeypatch, capsys) -> None:
    _use(monkeypatch, FakeRunner())
    code = main(["host", "ps", str(env / "nope.ps1"), "--json"])
    assert code == 2
    assert json.loads(capsys.readouterr().out)["ok"] is False


def test_ps_remote_failure_is_exit_one(env, monkeypatch, capsys) -> None:
    script = env / "probe.ps1"
    script.write_text("throw 'x'\n", encoding="utf-8")
    _use(monkeypatch, FakeRunner(ps_returncode=1, ps_stderr=b"boom"))
    code = main(["host", "ps", str(script), "--json"])
    assert code == 1
    assert json.loads(capsys.readouterr().out)["error"]["type"] == "remote-command-error"


def test_ps_utf16_script_is_exit_two_not_a_traceback(env, monkeypatch, capsys) -> None:
    # N3: a .ps1 saved as UTF-16 (Windows ISE default) is not UTF-8; it must become a typed exit-2
    # error, never an uncaught UnicodeDecodeError.
    script = env / "probe.ps1"
    script.write_text("Get-Date\n", encoding="utf-16")
    _use(monkeypatch, FakeRunner(ps_stdout=b"x\r\n"))
    code = main(["host", "ps", str(script), "--json"])
    assert code == 2
    assert json.loads(capsys.readouterr().out)["ok"] is False


def test_ps_script_too_long_is_exit_two_with_json(env, monkeypatch, capsys) -> None:
    script = env / "big.ps1"
    script.write_text("a" * 4000, encoding="utf-8")
    _use(monkeypatch, FakeRunner())
    code = main(["host", "ps", str(script), "--json"])
    body = json.loads(capsys.readouterr().out)
    assert code == 2 and body["error"]["type"] == "script-too-long"
    assert "AAR_EDIT_CMDLINE_MAX" in body["error"]["message"] and "8180" in body["error"]["message"]


def test_ps_env_cmdline_max_raises_cap_through_cli(env, monkeypatch, capsys) -> None:
    script = env / "big.ps1"
    script.write_text("a" * 4000, encoding="utf-8")
    _use(monkeypatch, FakeRunner(ps_stdout=b"ok\r\n"))
    monkeypatch.setenv("AAR_EDIT_CMDLINE_MAX", "32767")
    assert main(["host", "ps", str(script)]) == 0
    assert capsys.readouterr().out == "ok\n"
