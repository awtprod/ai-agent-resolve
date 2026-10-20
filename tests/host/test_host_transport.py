"""Verified pull/push/hash/ps against a fake runner: exact argv, both forms, every failure."""

from __future__ import annotations

import errno

import pytest
from host_support import ALIAS, FakeRunner, make_config, sha256_upper

from aar.host import transport
from aar.host.errors import (
    DestinationExists,
    HashMismatch,
    HostError,
    PathRefused,
    RemoteCommandError,
    SizeMismatch,
    TransportTimeout,
)
from aar.host.transport import EditHost, exists_script, move_script, stat_script


def _raise_oserror(*_args, **_kwargs):
    raise OSError(errno.EOPNOTSUPP, "operation not supported")


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
    assert host._scp_base() == [
        "scp",
        "-s",  # always SFTP mode (no toggle)
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
    # line-level assertions (not move_script-against-itself): the actual move line, with -Force
    lines = runner.scripts[-1].splitlines()
    move = next(ln for ln in lines if ln.startswith("Move-Item"))
    assert (
        move == "Move-Item -LiteralPath 'E:\\Show\\out.mp4.aarpart-TOKEN' -Destination $dst -Force"
    )


def test_move_script_guard_order_and_targets() -> None:
    """Line-level pins for the move guards (replaces the keyword-only reparse test): the container
    and reparse throws precede Move-Item, the reparse guard checks the parent, and -Force tracks
    overwrite. Covers HIGH-2 (no Split-Path) and MED-2 (M31/M32/M33/M34/M35)."""
    for overwrite in (False, True):
        lines = move_script("E:\\Show\\a.aarpart-T", "E:\\Show\\a", overwrite).splitlines()
        assert lines[0] == "$ErrorActionPreference = 'Stop'"
        move = [i for i, ln in enumerate(lines) if ln.startswith("Move-Item")]
        cont = [i for i, ln in enumerate(lines) if "-PathType Container" in ln and "throw" in ln]
        rep = [i for i, ln in enumerate(lines) if "ReparsePoint" in ln and "throw" in ln]
        assert len(move) == 1 and cont and rep and max(cont + rep) < move[0]
        # the reparse guard targets the parent, computed lexically (not Split-Path, HIGH-2)
        assert "$parent = [System.IO.Path]::GetDirectoryName($dst)" in lines
        assert "$pitem = Get-Item -LiteralPath $parent -Force" in lines
        assert "Split-Path" not in "\n".join(lines)
        move_line = lines[move[0]]
        assert move_line.startswith(
            "Move-Item -LiteralPath 'E:\\Show\\a.aarpart-T' -Destination $dst"
        )
        assert move_line.endswith(" -Force") is overwrite


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
    with pytest.raises(PathRefused) as info:
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


# -- round 2 fixes ---------------------------------------------------------------------------


def test_pull_escapes_glob_metacharacters_in_remote_source(tmp_path) -> None:
    # item 1: scp globs a remote SOURCE, so a literal '[' / ']' must be escaped.
    payload = b"bracketed media"
    runner = FakeRunner(stat_result=(len(payload), sha256_upper(payload)), scp_payload=payload)
    host = build(tmp_path, runner)
    dest = tmp_path / "out.mp4"
    host.pull("E:\\Show\\cut [v2].mp4", str(dest))
    assert runner.calls[1][-2] == f"{ALIAS}:E:/Show/cut \\[v2\\].mp4"
    assert dest.read_bytes() == payload


def test_push_remote_dest_is_literal_not_glob_escaped(tmp_path) -> None:
    # item 1: scp does NOT glob an upload DESTINATION, so it must stay literal (no backslashes).
    src, content = _src(tmp_path)
    runner = FakeRunner(stat_result=(len(content), sha256_upper(content)))
    host = build(tmp_path, runner)
    host.push(str(src), "E:\\Show\\out [1].mp4")
    assert runner.calls[1][-1] == f"{ALIAS}:E:/Show/out [1].mp4.aarpart-TOKEN"


def test_pull_falls_back_when_hardlinks_unsupported(tmp_path, monkeypatch) -> None:
    # item 3: on FAT/exFAT/SMB/FUSE os.link raises OSError; the O_EXCL fallback still completes.
    payload = b"content bytes"
    runner = FakeRunner(stat_result=(len(payload), sha256_upper(payload)), scp_payload=payload)
    host = build(tmp_path, runner)
    dest = tmp_path / "out.mp4"
    monkeypatch.setattr(transport.os, "link", _raise_oserror)
    host.pull(REMOTE, str(dest))
    assert dest.read_bytes() == payload
    assert not (tmp_path / "out.mp4.aarpart-TOKEN").exists()


def test_pull_fallback_no_clobber_against_a_concurrent_creator(tmp_path, monkeypatch) -> None:
    # item 3 / MED-2 (M22): the up-front exists() check passes, then a creator wins the race
    # AFTER it, so the O_EXCL claim in the fallback is what must refuse (not the pre-check).
    dest = tmp_path / "out.mp4"
    payload = b"content bytes"

    def link_then_someone_creates_dest(*_a, **_k):
        dest.write_bytes(b"theirs")  # concurrent creator, after the pre-check
        raise OSError(errno.EOPNOTSUPP, "no hard links")

    monkeypatch.setattr(transport.os, "link", link_then_someone_creates_dest)
    runner = FakeRunner(stat_result=(len(payload), sha256_upper(payload)), scp_payload=payload)
    host = build(tmp_path, runner)
    with pytest.raises(DestinationExists):
        host.pull(REMOTE, str(dest))
    assert dest.read_bytes() == b"theirs"  # not clobbered
    assert not (tmp_path / "out.mp4.aarpart-TOKEN").exists()


def test_pull_placement_failure_is_typed_and_cleans_up(tmp_path, monkeypatch) -> None:
    # item 3: a raw OSError during placement surfaces as a typed HostError, partial removed.
    payload = b"content bytes"
    runner = FakeRunner(stat_result=(len(payload), sha256_upper(payload)), scp_payload=payload)
    host = build(tmp_path, runner)
    dest = tmp_path / "out.mp4"
    monkeypatch.setattr(transport.os, "link", _raise_oserror)
    monkeypatch.setattr(transport.os, "replace", _raise_oserror)
    with pytest.raises(HostError) as info:
        host.pull(REMOTE, str(dest))
    assert info.value.kind == "local-io-error"
    assert not (tmp_path / "out.mp4.aarpart-TOKEN").exists()
    assert not dest.exists()  # the empty O_EXCL claim is cleaned up


def test_pull_refuses_directory_destination(tmp_path) -> None:
    # item 4: confirm pull's local side already refuses a directory destination.
    adir = tmp_path / "adir"
    adir.mkdir()
    runner = FakeRunner(stat_result=(1, sha256_upper(b"x")), scp_payload=b"x")
    host = build(tmp_path, runner)
    with pytest.raises(PathRefused) as info:
        host.pull(REMOTE, str(adir))
    assert info.value.reason == "destination-is-a-directory"
    assert runner.calls == []


def test_push_refuses_existing_remote_container(tmp_path) -> None:
    # item 4: a remote directory counts as existing, so a non-overwrite push is refused.
    src, _ = _src(tmp_path)
    runner = FakeRunner(exists_container=True)
    host = build(tmp_path, runner)
    with pytest.raises(DestinationExists):
        host.push(str(src), "E:\\Show\\out.mp4")
    assert len(runner.calls) == 1  # existence check only, no upload


def test_push_overwrite_onto_directory_fails_closed(tmp_path) -> None:
    # item 4: --overwrite skips the check, but the move script refuses a directory destination.
    src, content = _src(tmp_path)
    runner = FakeRunner(
        stat_result=(len(content), sha256_upper(content)),
        move_returncode=1,
        move_stderr=b"aar: destination is a directory",
    )
    host = build(tmp_path, runner)
    with pytest.raises(RemoteCommandError):
        host.push(str(src), "E:\\Show\\out.mp4", overwrite=True)
    assert runner.removed  # the partial upload is cleaned up
    move = next(s for s in runner.scripts if "Move-Item" in s)
    assert "PathType Container" in move  # the guard is in the move script itself


def test_ps_uses_the_configured_cmdline_cap(tmp_path) -> None:
    # item 2: the cmd.exe-sized default cap refuses a multi-KB script that fits CreateProcess.
    runner = FakeRunner()
    host = build(tmp_path, runner)
    with pytest.raises(HostError) as info:
        host.run_powershell("a" * 4000)
    assert info.value.kind == "script-too-long"
    assert runner.calls == []  # refused before any ssh call


def test_ps_cmdline_cap_is_configurable(tmp_path) -> None:
    # item 2: raising AAR_EDIT_CMDLINE_MAX (a PowerShell DefaultShell) lets the script through.
    runner = FakeRunner(ps_stdout=b"ok\r\n")
    host = build(tmp_path, runner, cmdline_max=32767)
    assert host.run_powershell("a" * 4000) == "ok\n"
