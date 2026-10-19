"""Verified pull/push/hash/ps against a fake runner: exact argv, both forms, every failure."""

from __future__ import annotations

import pytest
from host_support import ALIAS, FakeRunner, make_config, sha256_upper

from aar.host.errors import (
    DestinationExists,
    HashMismatch,
    RemoteCommandError,
    SizeMismatch,
    TransportTimeout,
)
from aar.host.transport import EditHost, exists_script, move_script, stat_script

REMOTE = "E:\\Show\\Review\\cut v2.mp4"
LITERAL = "E:\\Show\\Review\\cut v2.mp4"
SPEC_DRIVE = f"{ALIAS}:E:/Show/Review/cut v2.mp4"
SPEC_SLASH = f"{ALIAS}:/E:/Show/Review/cut v2.mp4"


def build(tmp_path, runner, **cfg_over) -> EditHost:
    return EditHost(make_config(tmp_path, **cfg_over), runner, token_factory=lambda: "TOKEN")


# -- pull ------------------------------------------------------------------------------------


def test_pull_success_pins_argv_drive_form(tmp_path) -> None:
    payload = b"hello world"
    runner = FakeRunner(stat_result=(len(payload), sha256_upper(payload)), scp_payload=payload)
    host = build(tmp_path, runner)
    dest = tmp_path / "out.mp4"

    result = host.pull(REMOTE, str(dest))

    assert dest.read_bytes() == payload
    assert result.size == len(payload)
    assert result.sha256 == sha256_upper(payload)
    assert result.op == "pull"
    # temp cleaned up, destination in place
    assert not (tmp_path / "out.mp4.aarpart-TOKEN").exists()
    # exact argv: stat over ssh, then scp download
    assert runner.calls[0] == host.powershell_argv(stat_script(LITERAL))
    assert runner.calls[1] == [*host._scp_base(), SPEC_DRIVE, f"{dest}.aarpart-TOKEN"]
    assert "-O" not in runner.calls[1]


def test_pull_scp_argv_slash_form(tmp_path) -> None:
    payload = b"x"
    runner = FakeRunner(stat_result=(1, sha256_upper(payload)), scp_payload=payload)
    host = build(tmp_path, runner, path_form="slash")
    dest = tmp_path / "out.mp4"

    host.pull(REMOTE, str(dest))

    assert runner.calls[1] == [*host._scp_base(), SPEC_SLASH, f"{dest}.aarpart-TOKEN"]


def test_ssh_base_flags(tmp_path) -> None:
    host = build(tmp_path, FakeRunner(), connect_timeout=20)
    assert host._ssh_base() == [
        "ssh",
        "-F",
        str(tmp_path / "ssh.cfg"),
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=20",
        ALIAS,
    ]
    assert host._scp_base()[:7] == [
        "scp",
        "-F",
        str(tmp_path / "ssh.cfg"),
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=20",
    ]


def test_pull_size_mismatch_removes_temp(tmp_path) -> None:
    payload = b"hello world"
    runner = FakeRunner(stat_result=(len(payload) + 1, sha256_upper(payload)), scp_payload=payload)
    host = build(tmp_path, runner)
    dest = tmp_path / "out.mp4"

    with pytest.raises(SizeMismatch):
        host.pull(REMOTE, str(dest))
    assert not dest.exists()
    assert not (tmp_path / "out.mp4.aarpart-TOKEN").exists()


def test_pull_hash_mismatch_removes_temp(tmp_path) -> None:
    payload = b"hello world"
    runner = FakeRunner(stat_result=(len(payload), "A" * 64), scp_payload=payload)
    host = build(tmp_path, runner)
    dest = tmp_path / "out.mp4"

    with pytest.raises(HashMismatch):
        host.pull(REMOTE, str(dest))
    assert not dest.exists()
    assert not (tmp_path / "out.mp4.aarpart-TOKEN").exists()


def test_pull_scp_timeout(tmp_path) -> None:
    runner = FakeRunner(stat_result=(1, sha256_upper(b"x")), scp_timeout=True)
    host = build(tmp_path, runner)
    with pytest.raises(TransportTimeout):
        host.pull(REMOTE, str(tmp_path / "out.mp4"))


def test_pull_scp_nonzero_exit(tmp_path) -> None:
    runner = FakeRunner(
        stat_result=(1, sha256_upper(b"x")), scp_returncode=1, scp_stderr=b"scp: no such file"
    )
    host = build(tmp_path, runner)
    with pytest.raises(RemoteCommandError) as info:
        host.pull(REMOTE, str(tmp_path / "out.mp4"))
    assert info.value.returncode == 1


def test_pull_remote_stat_failure_runs_no_scp(tmp_path) -> None:
    runner = FakeRunner(stat_returncode=1, stat_stderr=b"Get-Item: not found")
    host = build(tmp_path, runner)
    with pytest.raises(RemoteCommandError):
        host.pull(REMOTE, str(tmp_path / "out.mp4"))
    assert len(runner.calls) == 1  # stat only; no transfer attempted


def test_pull_refuses_existing_destination_without_overwrite(tmp_path) -> None:
    dest = tmp_path / "out.mp4"
    dest.write_bytes(b"old")
    runner = FakeRunner(stat_result=(1, sha256_upper(b"x")), scp_payload=b"x")
    host = build(tmp_path, runner)
    with pytest.raises(DestinationExists):
        host.pull(REMOTE, str(dest))
    assert runner.calls == []  # refused before any command
    assert dest.read_bytes() == b"old"


def test_pull_overwrite_replaces_destination(tmp_path) -> None:
    dest = tmp_path / "out.mp4"
    dest.write_bytes(b"old")
    payload = b"brand new bytes"
    runner = FakeRunner(stat_result=(len(payload), sha256_upper(payload)), scp_payload=payload)
    host = build(tmp_path, runner)
    host.pull(REMOTE, str(dest), overwrite=True)
    assert dest.read_bytes() == payload


# -- push ------------------------------------------------------------------------------------


def _src(tmp_path, content: bytes = b"master bytes"):
    src = tmp_path / "master.mov"
    src.write_bytes(content)
    return src, content


def test_push_success_pins_argv(tmp_path) -> None:
    src, content = _src(tmp_path)
    runner = FakeRunner(exists=False, stat_result=(len(content), sha256_upper(content)))
    host = build(tmp_path, runner)

    result = host.push(str(src), "E:\\Show\\out.mp4")

    assert result.op == "push"
    assert result.size == len(content)
    assert result.sha256 == sha256_upper(content)
    tmp_literal = "E:\\Show\\out.mp4.aarpart-TOKEN"
    assert runner.calls[0] == host.powershell_argv(exists_script("E:\\Show\\out.mp4"))
    assert runner.calls[1] == [
        *host._scp_base(),
        str(src),
        f"{ALIAS}:E:/Show/out.mp4.aarpart-TOKEN",
    ]
    assert runner.calls[2] == host.powershell_argv(stat_script(tmp_literal))
    assert runner.calls[3] == host.powershell_argv(
        move_script(tmp_literal, "E:\\Show\\out.mp4", False)
    )
    assert "-O" not in runner.calls[1]


def test_push_overwrite_skips_existence_check_and_forces_move(tmp_path) -> None:
    src, content = _src(tmp_path)
    runner = FakeRunner(stat_result=(len(content), sha256_upper(content)))
    host = build(tmp_path, runner)

    host.push(str(src), "E:\\Show\\out.mp4", overwrite=True)

    # first call is the upload, not a Test-Path
    assert runner.calls[0][0] == "scp"
    assert runner.scripts[-1] == move_script(
        "E:\\Show\\out.mp4.aarpart-TOKEN", "E:\\Show\\out.mp4", True
    )


def test_push_refuses_existing_remote_without_overwrite(tmp_path) -> None:
    src, _ = _src(tmp_path)
    runner = FakeRunner(exists=True)
    host = build(tmp_path, runner)
    with pytest.raises(DestinationExists):
        host.push(str(src), "E:\\Show\\out.mp4")
    assert len(runner.calls) == 1  # only the existence check ran, no upload


def test_push_size_mismatch_cleans_remote_temp(tmp_path) -> None:
    src, content = _src(tmp_path)
    runner = FakeRunner(stat_result=(len(content) + 9, sha256_upper(content)))
    host = build(tmp_path, runner)
    with pytest.raises(SizeMismatch):
        host.push(str(src), "E:\\Show\\out.mp4")
    assert runner.removed, "the partial remote upload should be removed"


def test_push_hash_mismatch_cleans_remote_temp(tmp_path) -> None:
    src, content = _src(tmp_path)
    runner = FakeRunner(stat_result=(len(content), "B" * 64))
    host = build(tmp_path, runner)
    with pytest.raises(HashMismatch):
        host.push(str(src), "E:\\Show\\out.mp4")
    assert runner.removed


def test_push_move_failure_cleans_remote_temp(tmp_path) -> None:
    src, content = _src(tmp_path)
    runner = FakeRunner(
        stat_result=(len(content), sha256_upper(content)),
        move_returncode=1,
        move_stderr=b"Move-Item: destination exists",
    )
    host = build(tmp_path, runner)
    with pytest.raises(RemoteCommandError):
        host.push(str(src), "E:\\Show\\out.mp4")
    assert runner.removed


def test_push_scp_timeout_cleans_remote_temp(tmp_path) -> None:
    src, _ = _src(tmp_path)
    runner = FakeRunner(scp_timeout=True)
    host = build(tmp_path, runner)
    with pytest.raises(TransportTimeout):
        host.push(str(src), "E:\\Show\\out.mp4")
    assert runner.removed


def test_push_refuses_source_symlink_out_of_root(tmp_path) -> None:
    outside = tmp_path.parent / "outside-secret.bin"
    outside.write_bytes(b"not yours")
    link = tmp_path / "link.mov"
    link.symlink_to(outside)
    runner = FakeRunner()
    host = build(tmp_path, runner)
    with pytest.raises(Exception) as info:  # PathRefused
        host.push(str(link), "E:\\Show\\out.mp4")
    assert "outside-allowed-roots" in str(info.value)
    assert runner.calls == []


# -- hash and ps -----------------------------------------------------------------------------


def test_hash_pins_argv_and_returns_size_and_sha(tmp_path) -> None:
    runner = FakeRunner(stat_result=(123, "C" * 64))
    host = build(tmp_path, runner)
    result = host.file_hash("E:\\Show\\x.mp4")
    assert result.op == "hash"
    assert result.size == 123
    assert result.sha256 == "C" * 64
    assert runner.calls[0] == host.powershell_argv(stat_script("E:\\Show\\x.mp4"))


def test_run_powershell_returns_clean_stdout(tmp_path) -> None:
    runner = FakeRunner(ps_stdout=b"line one\r\nline two\r\n")
    host = build(tmp_path, runner)
    assert host.run_powershell("Get-Date") == "line one\nline two\n"


def test_run_powershell_timeout(tmp_path) -> None:
    runner = FakeRunner(ps_timeout=True)
    host = build(tmp_path, runner)
    with pytest.raises(TransportTimeout):
        host.run_powershell("Start-Sleep 999")


def test_run_powershell_nonzero_strips_clixml_from_error(tmp_path) -> None:
    clixml = (
        b"#< CLIXML\n"
        b'<Objs Version="1.1.0.1" xmlns="x"><S S="Error">Boom happened'
        b"_x000D__x000A_</S></Objs>\n"
    )
    runner = FakeRunner(ps_returncode=1, ps_stderr=clixml)
    host = build(tmp_path, runner)
    with pytest.raises(RemoteCommandError) as info:
        host.run_powershell("throw 'Boom'")
    assert "Boom happened" in info.value.stderr
    assert "CLIXML" not in info.value.stderr
