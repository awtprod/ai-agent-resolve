"""Safety properties: placement/cleanup on every failure and interrupt, local destination kinds,
remote roots, remote_kind/remote_stat strictness, the cmd.exe cap boundary, config parsing, glob
escaping and TransferResult fields. Adapted from the validator's probe_transport (MED-2); the
probes that documented be555da's behaviour are rewritten to the round-3 behaviour. Fake runner
only -- no host, PowerShell or scp.
"""

from __future__ import annotations

import errno
import os

import pytest
from host_support import ALIAS, FakeRunner, make_config, sha256_upper

from aar.host import transport
from aar.host.config import env_config
from aar.host.errors import (
    ConfigError,
    DestinationExists,
    LocalIOError,
    PathRefused,
    RemoteCommandError,
    ScriptTooLong,
)
from aar.host.powershell import DEFAULT_REMOTE_CMDLINE_MAX, encode_command, powershell_argv
from aar.host.runner import RunResult
from aar.host.transport import EditHost
from aar.host.winpath import (
    normalize_root,
    remote_spec,
    scp_remote_dest,
    scp_remote_source,
    validate_windows_path,
)

REMOTE = "E:\\Show\\Review\\cut v2.mp4"
PAYLOAD = b"verified payload"


def host_for(tmp_path, runner=None, **cfg) -> tuple[EditHost, FakeRunner]:
    runner = runner or FakeRunner(
        stat_result=(len(PAYLOAD), sha256_upper(PAYLOAD)), scp_payload=PAYLOAD
    )
    return EditHost(make_config(tmp_path, **cfg), runner, token_factory=lambda: "TOKEN"), runner


def leftovers(tmp_path) -> list[str]:
    return sorted(p.name for p in tmp_path.iterdir() if ".aarpart-" in p.name)


# == no hard links: each errno falls back and completes (LOW-3 deliberately NOT narrowed) =====

ERRNOS = ["EPERM", "EOPNOTSUPP", "ENOTSUP", "EXDEV", "EMLINK", "EIO", "ENOSPC", "EACCES"]


@pytest.mark.parametrize("name", ERRNOS)
def test_link_errno_falls_back_and_completes(tmp_path, monkeypatch, name) -> None:
    code = getattr(errno, name)

    def no_link(*_a, **_k):
        raise OSError(code, os.strerror(code))

    monkeypatch.setattr(transport.os, "link", no_link)
    host, _ = host_for(tmp_path)
    dest = tmp_path / "out.mp4"
    host.pull(REMOTE, str(dest))
    assert dest.read_bytes() == PAYLOAD
    assert leftovers(tmp_path) == []


@pytest.mark.parametrize("name", ERRNOS)
def test_fallback_is_no_clobber_against_a_concurrent_creator(tmp_path, monkeypatch, name) -> None:
    code = getattr(errno, name)
    dest = tmp_path / "out.mp4"

    def link_then_someone_creates_dest(*_a, **_k):
        dest.write_bytes(b"theirs")  # a concurrent creator wins the race after the pre-check
        raise OSError(code, os.strerror(code))

    monkeypatch.setattr(transport.os, "link", link_then_someone_creates_dest)
    host, _ = host_for(tmp_path)
    with pytest.raises(DestinationExists):
        host.pull(REMOTE, str(dest))
    assert dest.read_bytes() == b"theirs"
    assert leftovers(tmp_path) == []


def test_fallback_open_eacces_is_typed_and_clean(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        transport.os, "link", lambda *a, **k: (_ for _ in ()).throw(OSError(errno.EPERM, "x"))
    )
    real_open = os.open

    def open_eacces(path, flags, *a):
        if str(path).endswith("out.mp4"):
            raise PermissionError(errno.EACCES, "denied")
        return real_open(path, flags, *a)

    monkeypatch.setattr(transport.os, "open", open_eacces)
    host, _ = host_for(tmp_path)
    with pytest.raises(LocalIOError) as info:
        host.pull(REMOTE, str(tmp_path / "out.mp4"))
    assert isinstance(info.value.__cause__, PermissionError)
    assert leftovers(tmp_path) == [] and not (tmp_path / "out.mp4").exists()


def test_eio_from_link_is_masked_by_the_fallback_error(tmp_path, monkeypatch) -> None:
    """LOW-3 (not implemented): an EIO from os.link is not reported; the fallback's error is."""
    monkeypatch.setattr(
        transport.os, "link", lambda *a, **k: (_ for _ in ()).throw(OSError(errno.EIO, "io"))
    )
    monkeypatch.setattr(
        transport.os,
        "replace",
        lambda *a, **k: (_ for _ in ()).throw(OSError(errno.ENOSPC, "full")),
    )
    host, _ = host_for(tmp_path)
    with pytest.raises(LocalIOError) as info:
        host.pull(REMOTE, str(tmp_path / "out.mp4"))
    assert info.value.__cause__.errno == errno.ENOSPC  # the original EIO is lost
    assert leftovers(tmp_path) == [] and not (tmp_path / "out.mp4").exists()


def test_fallback_cleanup_never_deletes_another_writers_file(tmp_path, monkeypatch) -> None:
    """LOW-1(i): another writer replaces our empty claim, then our replace fails; keep theirs."""
    dest = tmp_path / "out.mp4"
    monkeypatch.setattr(
        transport.os, "link", lambda *a, **k: (_ for _ in ()).throw(OSError(errno.EPERM, "x"))
    )
    real_replace = os.replace

    def their_replace_then_ours_fails(src, dst):
        if str(dst) == str(dest):
            theirs = tmp_path / "theirs.tmp"
            theirs.write_bytes(b"their finished file")
            real_replace(theirs, dest)  # another writer's rename lands on the claimed name
            raise OSError(errno.EIO, "io error")
        return real_replace(src, dst)

    monkeypatch.setattr(transport.os, "replace", their_replace_then_ours_fails)
    host, _ = host_for(tmp_path)
    with pytest.raises(LocalIOError):
        host.pull(REMOTE, str(dest))
    assert dest.exists() and dest.read_bytes() == b"their finished file", "their file was deleted"


# == interrupt (KeyboardInterrupt) at each step: no temp file left ============================


class InterruptingRunner(FakeRunner):
    """Raises KeyboardInterrupt on the Nth runner call; on an scp pull it first writes a partial."""

    def __init__(self, at: int, **kw) -> None:
        super().__init__(**kw)
        self.at = at

    def run(self, argv, *, stdin=None, timeout=None):
        if len(self.calls) == self.at:
            self.calls.append(list(argv))
            if argv[0] == "scp" and not argv[-1].startswith(f"{ALIAS}:"):
                with open(argv[-1], "wb") as fh:  # the partial a killed scp leaves
                    fh.write(PAYLOAD[:3])
            raise KeyboardInterrupt
        return super().run(argv, stdin=stdin, timeout=timeout)


def test_pull_interrupt_during_remote_stat_leaves_nothing(tmp_path) -> None:
    runner = InterruptingRunner(
        0, stat_result=(len(PAYLOAD), sha256_upper(PAYLOAD)), scp_payload=PAYLOAD
    )
    host, _ = host_for(tmp_path, runner)
    with pytest.raises(KeyboardInterrupt):
        host.pull(REMOTE, str(tmp_path / "out.mp4"))
    assert leftovers(tmp_path) == []


def test_pull_interrupt_during_scp_removes_the_partial(tmp_path) -> None:
    runner = InterruptingRunner(
        1, stat_result=(len(PAYLOAD), sha256_upper(PAYLOAD)), scp_payload=PAYLOAD
    )
    host, _ = host_for(tmp_path, runner)
    with pytest.raises(KeyboardInterrupt):
        host.pull(REMOTE, str(tmp_path / "out.mp4"))
    assert leftovers(tmp_path) == [], f"partial left behind: {leftovers(tmp_path)}"


def test_pull_runner_oserror_during_scp_is_typed_and_clean(tmp_path) -> None:
    class Boom(FakeRunner):
        def _scp(self, argv):
            with open(argv[-1], "wb") as fh:
                fh.write(b"par")
            raise OSError(errno.EPIPE, "broken pipe")

    host, _ = host_for(tmp_path, Boom(stat_result=(len(PAYLOAD), sha256_upper(PAYLOAD))))
    with pytest.raises(LocalIOError):
        host.pull(REMOTE, str(tmp_path / "out.mp4"))
    assert leftovers(tmp_path) == []


@pytest.mark.parametrize("target", ["getsize", "sha256_file", "link"])
def test_pull_interrupt_while_verifying_or_placing_leaves_nothing(
    tmp_path, monkeypatch, target
) -> None:
    def interrupt(*_a, **_k):
        raise KeyboardInterrupt

    if target == "getsize":
        monkeypatch.setattr(transport.os.path, "getsize", interrupt)
    elif target == "sha256_file":
        monkeypatch.setattr(transport, "sha256_file", interrupt)
    else:
        monkeypatch.setattr(transport.os, "link", interrupt)
    host, _ = host_for(tmp_path)
    with pytest.raises(KeyboardInterrupt):
        host.pull(REMOTE, str(tmp_path / "out.mp4"))
    assert leftovers(tmp_path) == []
    assert not (tmp_path / "out.mp4").exists()


def test_pull_interrupt_after_link_leaves_a_complete_dest_and_no_temp(
    tmp_path, monkeypatch
) -> None:
    real_unlink = os.unlink
    calls = {"n": 0}

    def unlink_interrupt_once(path, *a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise KeyboardInterrupt
        return real_unlink(path, *a, **k)

    monkeypatch.setattr(transport.os, "unlink", unlink_interrupt_once)
    host, _ = host_for(tmp_path)
    with pytest.raises(KeyboardInterrupt):
        host.pull(REMOTE, str(tmp_path / "out.mp4"))
    assert leftovers(tmp_path) == []
    assert (tmp_path / "out.mp4").read_bytes() == PAYLOAD  # verified data, consistent


def test_pull_interrupt_in_fallback_leaves_no_empty_destination(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        transport.os, "link", lambda *a, **k: (_ for _ in ()).throw(OSError(errno.EPERM, "x"))
    )
    monkeypatch.setattr(
        transport.os, "replace", lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt())
    )
    host, _ = host_for(tmp_path)
    dest = tmp_path / "out.mp4"
    with pytest.raises(KeyboardInterrupt):
        host.pull(REMOTE, str(dest))
    assert leftovers(tmp_path) == []
    assert not dest.exists(), "zero-byte claim left at the destination"


def _push_host(tmp_path, at: int, **kw):
    src = tmp_path / "master.mov"
    src.write_bytes(PAYLOAD)
    runner = InterruptingRunner(at, stat_result=(len(PAYLOAD), sha256_upper(PAYLOAD)), **kw)
    host, _ = host_for(tmp_path, runner)
    return host, runner, src


def test_push_interrupt_during_exists_check_creates_no_remote_temp(tmp_path) -> None:
    host, runner, src = _push_host(tmp_path, 0)
    with pytest.raises(KeyboardInterrupt):
        host.push(str(src), "E:\\Show\\out.mp4")
    assert [c[0] for c in runner.calls] == ["ssh"]  # nothing uploaded


def test_push_interrupt_during_scp_removes_the_remote_partial(tmp_path) -> None:
    host, runner, src = _push_host(tmp_path, 1)
    with pytest.raises(KeyboardInterrupt):
        host.push(str(src), "E:\\Show\\out.mp4")
    assert runner.removed, "no Remove-Item for the remote .aarpart temp after an interrupted upload"


@pytest.mark.parametrize("at", [2, 3], ids=["during-remote-stat", "during-move"])
def test_push_interrupt_after_upload_removes_the_remote_temp(tmp_path, at) -> None:
    host, runner, src = _push_host(tmp_path, at)
    with pytest.raises(KeyboardInterrupt):
        host.push(str(src), "E:\\Show\\out.mp4")
    assert runner.removed and "out.mp4.aarpart-TOKEN" in runner.removed[-1]


# == pull into a local directory, symlink, dangling symlink ==================================


def test_pull_into_existing_directory_refused_even_with_overwrite(tmp_path) -> None:
    (tmp_path / "d").mkdir()
    host, runner = host_for(tmp_path)
    for ow in (False, True):
        with pytest.raises(PathRefused) as info:
            host.pull(REMOTE, str(tmp_path / "d"), overwrite=ow)
        assert info.value.reason == "destination-is-a-directory"
    assert runner.calls == []


def test_pull_onto_symlink_to_dir_inside_root_refused(tmp_path) -> None:
    (tmp_path / "d").mkdir()
    (tmp_path / "ln").symlink_to(tmp_path / "d")
    host, runner = host_for(tmp_path)
    with pytest.raises(PathRefused):
        host.pull(REMOTE, str(tmp_path / "ln"), overwrite=True)
    assert runner.calls == []


def test_pull_onto_symlink_to_file_inside_root(tmp_path) -> None:
    target = tmp_path / "target.mp4"
    target.write_bytes(b"old")
    (tmp_path / "ln").symlink_to(target)
    host, _ = host_for(tmp_path)
    with pytest.raises(DestinationExists):
        host.pull(REMOTE, str(tmp_path / "ln"))
    assert target.read_bytes() == b"old"
    result = host.pull(REMOTE, str(tmp_path / "ln"), overwrite=True)
    assert result.local == str(target)  # the resolved target is replaced, the link is kept
    assert (tmp_path / "ln").is_symlink() and target.read_bytes() == PAYLOAD


def test_pull_onto_symlink_out_of_root_refused(tmp_path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside.bin"
    outside.write_bytes(b"secret")
    (tmp_path / "ln").symlink_to(outside)
    host, runner = host_for(tmp_path)
    with pytest.raises(PathRefused) as info:
        host.pull(REMOTE, str(tmp_path / "ln"), overwrite=True)
    assert info.value.reason == "outside-allowed-roots"
    assert outside.read_bytes() == b"secret" and runner.calls == []


def test_pull_onto_dangling_symlink_inside_root_creates_the_target(tmp_path) -> None:
    (tmp_path / "ln").symlink_to(tmp_path / "later.mp4")
    host, _ = host_for(tmp_path)
    result = host.pull(REMOTE, str(tmp_path / "ln"))
    assert result.local == str(tmp_path / "later.mp4")
    assert (tmp_path / "later.mp4").read_bytes() == PAYLOAD and leftovers(tmp_path) == []


def test_pull_dangling_symlink_in_fallback_is_not_followed(tmp_path, monkeypatch) -> None:
    """O_CREAT|O_EXCL must not follow a symlink planted at dest after validation."""
    dest = tmp_path / "out.mp4"
    outside = tmp_path.parent / f"{tmp_path.name}-planted.bin"

    def link_and_plant(*_a, **_k):
        dest.symlink_to(outside)  # planted after the checks
        raise OSError(errno.EPERM, "no links")

    monkeypatch.setattr(transport.os, "link", link_and_plant)
    host, _ = host_for(tmp_path)
    with pytest.raises(DestinationExists):
        host.pull(REMOTE, str(dest))
    assert not outside.exists()


# == remote path at a root / trailing separator (Nit N1: now refused) ========================


def test_remote_path_equal_to_a_root_is_refused() -> None:
    with pytest.raises(PathRefused) as info:
        validate_windows_path("E:\\Show", (normalize_root("E:\\Show"),))
    assert info.value.reason == "equals-allowed-root"


def test_trailing_separator_is_refused() -> None:
    with pytest.raises(PathRefused) as info:
        validate_windows_path("E:\\Show\\x.mp4\\", (normalize_root("E:\\Show"),))
    assert info.value.reason == "trailing-separator"


@pytest.mark.parametrize(
    ("root", "path"),
    [("E:", "E:\\"), ("E:", "E:"), ("\\\\srv\\share", "\\\\srv\\share")],
)
def test_bare_drive_or_share_root_as_path_is_refused(root, path) -> None:
    with pytest.raises(PathRefused) as info:
        validate_windows_path(path, (normalize_root(root),))
    assert info.value.reason in ("no-file-component", "not-a-drive-or-unc-path")


def test_push_onto_root_dir_is_refused(tmp_path) -> None:
    src = tmp_path / "m.mov"
    src.write_bytes(PAYLOAD)
    host, runner = host_for(tmp_path)
    with pytest.raises(PathRefused) as info:
        host.push(str(src), "E:\\Show")
    assert info.value.reason == "equals-allowed-root" and runner.calls == []


# == remote_kind / remote_stat strictness (LOW-2) ===========================================


class KindRunner(FakeRunner):
    def __init__(self, line: bytes, **kw) -> None:
        super().__init__(**kw)
        self.line = line

    def _ssh(self, script):
        if "Test-Path" in script and "Move-Item" not in script:
            return RunResult(0, self.line, b"")
        return super()._ssh(script)


@pytest.mark.parametrize(
    "line", [b"AAREXISTS\r\n", b"AAREXISTS maybe\r\n", b"AAREXISTS file extra\r\n"]
)
def test_remote_kind_rejects_unexpected_output(tmp_path, line) -> None:
    host, _ = host_for(tmp_path, KindRunner(line))
    with pytest.raises(RemoteCommandError):
        host.remote_kind(validate_windows_path("E:\\Show\\x.mp4", host.config.remote_roots_parsed))


def test_remote_kind_unexpected_output_makes_push_fail_closed(tmp_path) -> None:
    """LOW-2 fixed: a garbled AAREXISTS line now refuses the push instead of uploading."""
    src = tmp_path / "m.mov"
    src.write_bytes(PAYLOAD)
    host, runner = host_for(
        tmp_path, KindRunner(b"AAREXISTS\r\n", stat_result=(len(PAYLOAD), sha256_upper(PAYLOAD)))
    )
    with pytest.raises(RemoteCommandError):
        host.push(str(src), "E:\\Show\\out.mp4")
    assert [c[0] for c in runner.calls] == ["ssh"]  # the exists check failed closed, no upload


def test_remote_stat_rejects_non_ascii_digits(tmp_path) -> None:
    class StatRunner(FakeRunner):
        def _ssh(self, script):
            return RunResult(0, "AARSTAT \u00b2 ABCD\r\n".encode(), b"")

    host, _ = host_for(tmp_path, StatRunner())
    with pytest.raises(RemoteCommandError):  # superscript two is not an ASCII digit, not 64-hex
        host.file_hash("E:\\Show\\x.mp4")


def test_remote_stat_rejects_short_hash(tmp_path) -> None:
    class StatRunner(FakeRunner):
        def _ssh(self, script):
            return RunResult(0, b"AARSTAT 10 ABCD\r\n", b"")

    host, _ = host_for(tmp_path, StatRunner())
    with pytest.raises(RemoteCommandError):
        host.file_hash("E:\\Show\\x.mp4")


# == cmd.exe cap boundary, refusal text, config parsing (MED-2 M10-M12) ======================

PREFIX_LEN = len("powershell -NoProfile -NonInteractive -EncodedCommand ")


def _line_len(script: str) -> int:
    return PREFIX_LEN + len(encode_command(script))


def test_cap_boundary_exact_and_plus_one() -> None:
    script = "a" * 2500
    exact = _line_len(script)
    powershell_argv(script, cmdline_max=exact)  # exactly at the cap: accepted
    with pytest.raises(ScriptTooLong) as info:
        powershell_argv(script, cmdline_max=exact - 1)  # one over: refused
    assert info.value.length == exact and info.value.limit == exact - 1
    assert str(exact - 1) in str(info.value) and "AAR_EDIT_CMDLINE_MAX" in str(info.value)


def test_default_cap_value_and_boundary() -> None:
    assert DEFAULT_REMOTE_CMDLINE_MAX == 8191 - len("cmd.exe /c ") == 8180
    n_ok = max(n for n in range(3040, 3060) if _line_len("a" * n) <= 8180)
    assert _line_len("a" * n_ok) == 8178
    powershell_argv("a" * n_ok)
    n_over = n_ok + 1
    assert _line_len("a" * n_over) == 8182
    with pytest.raises(ScriptTooLong) as info:
        powershell_argv("a" * n_over)
    assert "8180-character" in str(info.value) and "AAR_EDIT_CMDLINE_MAX" in str(info.value)


def _env(tmp_path, **extra):
    env = {
        "AAR_EDIT_SSH_ALIAS": "edit-pc",
        "AAR_EDIT_SSH_CONFIG": str(tmp_path / "ssh.cfg"),
        "AAR_EDIT_REMOTE_ROOTS": "E:\\Show",
        "AAR_EDIT_LOCAL_ROOTS": str(tmp_path),
    }
    env.update(extra)
    return env


@pytest.mark.parametrize("raw", ["0", "-1", "abc", "8180.5", "1e4", "True"])
def test_env_cmdline_max_bad_values_rejected(tmp_path, raw) -> None:
    with pytest.raises(ConfigError):
        env_config(_env(tmp_path, AAR_EDIT_CMDLINE_MAX=raw))


def test_env_cmdline_max_whitespace_and_empty(tmp_path) -> None:
    assert env_config(_env(tmp_path, AAR_EDIT_CMDLINE_MAX=" 9000 ")).cmdline_max == 9000
    assert env_config(_env(tmp_path, AAR_EDIT_CMDLINE_MAX="")).cmdline_max == 8180


# == LOW-4: bool rejected where int is required ==============================================


def test_cmdline_max_true_is_rejected(tmp_path) -> None:
    with pytest.raises(ConfigError):
        make_config(tmp_path, cmdline_max=True)


def test_connect_timeout_true_is_rejected(tmp_path) -> None:
    with pytest.raises(ConfigError):
        make_config(tmp_path, connect_timeout=True)


def test_command_timeout_true_is_rejected(tmp_path) -> None:
    with pytest.raises(ConfigError):
        make_config(tmp_path, command_timeout=True)


# == LOW-5: alias hazards now refused ========================================================


@pytest.mark.parametrize("alias", ["edit/pc", "[edit", "edit pc", "edit:pc", "-edit", ""])
def test_alias_scp_parsing_hazards_are_refused(tmp_path, alias) -> None:
    with pytest.raises(ConfigError):
        make_config(tmp_path, alias=alias)


def test_plain_aliases_accepted(tmp_path) -> None:
    for alias in ("edit-pc", "e", "edit.pc_1@host"):
        assert make_config(tmp_path, alias=alias).alias == alias


# == glob escape: pull source only, dest literal, both forms, UNC (M05-M09) ==================


@pytest.mark.parametrize("form", ["drive", "slash"])
@pytest.mark.parametrize(
    ("root", "path"),
    [
        ("E:\\Show", "E:\\Show\\a [b] c\\d[1].mp4"),
        ("E:\\Show", "e:/show/x]y[z.mp4"),
        ("\\\\srv\\share", "\\\\srv\\share\\dir [1]\\x[2].mp4"),
        ("\\\\srv\\share", "//srv/share/dir/x.mp4"),
    ],
)
def test_glob_escape_source_only_both_forms(root, path, form) -> None:
    win = validate_windows_path(path, (normalize_root(root),))
    spec = remote_spec(win, form)
    src = scp_remote_source("edit-pc", win, form)
    dst = scp_remote_dest("edit-pc", win, form)
    assert "\\" not in spec  # remote_spec forward-slashes everything
    assert dst == f"edit-pc:{spec}"  # destination literal
    escaped = src.split(":", 1)[1]
    assert escaped.replace("\\[", "[").replace("\\]", "]") == spec
    assert escaped.count("\\") == spec.count("[") + spec.count("]")
    if form == "slash":
        assert spec.startswith("/")
    if win.unc:
        assert spec.startswith("//" if form == "drive" else "///")


# == TransferResult fields (M41-M43) =========================================================


def test_transfer_result_fields(tmp_path) -> None:
    host, runner = host_for(tmp_path)
    res = host.pull("E:\\Show\\c [1].mp4", str(tmp_path / "out.mp4"))
    assert res.remote == "E:/Show/c [1].mp4"  # literal, unescaped, no alias
    assert res.local == str(tmp_path / "out.mp4")
    assert res.argv == runner.calls[1] and res.argv[-2] == "edit-pc:E:/Show/c \\[1\\].mp4"
    assert res.to_dict()["ok"] is True and res.to_dict()["argv"] is not res.argv

    src = tmp_path / "m.mov"
    src.write_bytes(PAYLOAD)
    host2, runner2 = host_for(
        tmp_path, FakeRunner(stat_result=(len(PAYLOAD), sha256_upper(PAYLOAD)))
    )
    res2 = host2.push(str(src), "E:\\Show\\o [1].mov")
    assert res2.remote == "E:/Show/o [1].mov"  # the final name, not the temp
    assert res2.local == str(src) and res2.argv == runner2.calls[1]
    assert res2.argv[-1] == "edit-pc:E:/Show/o [1].mov.aarpart-TOKEN"
