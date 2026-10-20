"""S13f3 follow-ups (NF1-NF8) and the validator's mutant-killing probe cases (P13/P15/P17-P19/
P23/P24/P27). Fake runner only; no host, no ssh."""

from __future__ import annotations

import errno
import json
import os

import pytest
from host_support import ALIAS, FakeRunner, decode_encoded_command, make_config, sha256_upper

from aar.cli import main
from aar.host import cli as host_cli
from aar.host import transport
from aar.host.config import MAX_TIMEOUT_SECONDS, EditHostConfig, env_config
from aar.host.errors import (
    ConfigError,
    HashMismatch,
    HostError,
    LocalIOError,
    PathRefused,
    SizeMismatch,
)
from aar.host.powershell import powershell_argv
from aar.host.transport import EditHost, _unlink_if_claim
from aar.host.winpath import normalize_root, validate_windows_path

PAYLOAD = b"verified payload"
REMOTE = "E:\\Show\\cut.mp4"
CAP = MAX_TIMEOUT_SECONDS


def _env(tmp_path, **extra):
    env = {
        "AAR_EDIT_SSH_ALIAS": "edit-pc",
        "AAR_EDIT_SSH_CONFIG": str(tmp_path / "ssh.cfg"),
        "AAR_EDIT_REMOTE_ROOTS": "E:\\Show",
        "AAR_EDIT_LOCAL_ROOTS": str(tmp_path),
    }
    env.update(extra)
    return env


def _cli_env(tmp_path, monkeypatch, runner, **extra):
    for k, v in _env(tmp_path, **extra).items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(host_cli, "_default_runner", lambda: runner)


def _host(tmp_path, runner):
    return EditHost(make_config(tmp_path), runner, token_factory=lambda: "T")


class StepRunner(FakeRunner):
    """Raise per-call: ``plan`` maps call index -> exception factory."""

    def __init__(self, plan, **kw) -> None:
        kw.setdefault("stat_result", (len(PAYLOAD), sha256_upper(PAYLOAD)))
        kw.setdefault("scp_payload", PAYLOAD)
        super().__init__(**kw)
        self.plan = plan

    def run(self, argv, *, stdin=None, timeout=None):
        n = len(self.calls)
        if n in self.plan:
            self.calls.append(list(argv))
            if argv[0] == "ssh":
                self.scripts.append(decode_encoded_command(argv))
            if argv[0] == "scp" and not argv[-1].startswith(f"{ALIAS}:"):
                with open(argv[-1], "wb") as fh:
                    fh.write(PAYLOAD[:3])
            raise self.plan[n]()
        return super().run(argv, stdin=stdin, timeout=timeout)


def oserr():
    return BlockingIOError(errno.EAGAIN, "fork failed")


def kbi():
    return KeyboardInterrupt()


# == NF1: ASCII-only root fold ====================================================================

OVERFOLD = [
    ("E:\\Show", "E:\\\u017fhow\\x.mp4"),  # long s (U+017F) vs S
    ("E:\\Show", "E:\\Sho\u017f\\x.mp4"),
    ("E:\\Media", "E:\\Med\u0131a\\x.mp4"),  # dotless i (U+0131) vs i
    ("E:\\Kelvin", "E:\\\u212aelvin\\x.mp4"),  # Kelvin sign (U+212A) vs K
    ("E:\\Istanbul", "E:\\\u0130stanbul\\x.mp4"),  # dotted I (U+0130) vs I
    ("E:\\Show", "E:\\\uff33how\\x.mp4"),  # full-width S (U+FF33)
    ("E:\\Caf\u00e9", "E:\\CAF\u00c9\\x.mp4"),  # Café vs CAFÉ (non-ASCII case)
    ("E:\\Show\\Edits", "E:\\Show\\Edit\u017f\\x.mp4"),  # deep
]


@pytest.mark.parametrize(("root", "path"), OVERFOLD, ids=[p for _, p in OVERFOLD])
def test_nf1_non_ascii_case_is_outside_the_root(root, path) -> None:
    with pytest.raises(PathRefused) as info:
        validate_windows_path(path, (normalize_root(root),))
    assert info.value.reason == "outside-allowed-roots"


@pytest.mark.parametrize("path", ["E:\\SHOW\\x.mp4", "E:\\show\\x.mp4", "E:\\ShOw\\x.mp4"])
def test_nf1_ascii_case_differences_still_accepted(path) -> None:
    win = validate_windows_path(path, (normalize_root("E:\\Show"),))
    assert win.parts[0].lower() == "show"


@pytest.mark.parametrize(
    ("root", "path", "reason"),
    [
        ("E:\\Caf\u00e9", "E:\\Caf\u00e9", "equals-allowed-root"),  # exact non-ASCII: equals root
        ("E:\\Caf\u00e9", "E:\\CAF\u00c9", "outside-allowed-roots"),  # non-ASCII case: outside
        ("E:\\Show", "E:\\SHOW", "equals-allowed-root"),  # ASCII case folds: equals root
        ("E:\\Show", "E:\\\u017fhow", "outside-allowed-roots"),  # long s: outside, not equals
    ],
    ids=["cafe-exact", "cafe-case", "ascii-case", "long-s"],
)
def test_nf1_equals_root_with_folding(root, path, reason) -> None:
    with pytest.raises(PathRefused) as info:
        validate_windows_path(path, (normalize_root(root),))
    assert info.value.reason == reason


# == NF2: huge int / finite config ================================================================


def test_nf2_huge_int_command_timeout_is_a_config_error(tmp_path) -> None:
    # 10**400 made math.isfinite raise OverflowError before the range check; now a typed error.
    with pytest.raises(ConfigError):
        make_config(tmp_path, command_timeout=10**400)


@pytest.mark.parametrize("value", [CAP, float(CAP), 0.001, 1, 300.0])
def test_nf6_command_timeout_boundary_accepted(tmp_path, value) -> None:
    assert make_config(tmp_path, command_timeout=value).command_timeout == value


@pytest.mark.parametrize(
    "value",
    [
        0,
        0.0,
        -1,
        -0.0,
        CAP + 0.0001,
        CAP + 1,
        10**400,
        float("1e400"),
        float("nan"),
        float("inf"),
        float("-inf"),
        "10",
        None,
        True,
        False,
    ],
    ids=repr,
)
def test_nf6_command_timeout_boundary_refused(tmp_path, value) -> None:
    with pytest.raises(ConfigError):
        make_config(tmp_path, command_timeout=value)


@pytest.mark.parametrize(
    ("raw", "ok"),
    [
        ("604800", True),
        (" 604800 ", True),
        ("604800.0001", False),
        ("604801", False),
        ("1e400", False),
        ("1e-9", True),
        ("nan", False),
        ("inf", False),
        ("abc", False),
    ],
)
def test_nf6_env_command_timeout_boundary(tmp_path, raw, ok) -> None:
    env = _env(tmp_path, AAR_EDIT_COMMAND_TIMEOUT=raw)
    if ok:
        assert env_config(env).command_timeout == float(raw)
    else:
        with pytest.raises(ConfigError):
            env_config(env)


# == NF3: library timeout override validated ======================================================


@pytest.mark.parametrize("timeout", [float("inf"), float("nan"), 1e300, 0, -1, CAP + 1])
def test_nf3_run_powershell_override_is_validated(tmp_path, timeout) -> None:
    class NoRun(FakeRunner):
        def run(self, *a, **k):
            raise AssertionError("runner should not be reached for a bad timeout")

    host = _host(tmp_path, NoRun())
    with pytest.raises(ConfigError):
        host.run_powershell("Get-Date", timeout=timeout)


def test_nf3_good_override_reaches_the_runner(tmp_path) -> None:
    runner = FakeRunner(ps_stdout=b"ok\r\n")
    assert _host(tmp_path, runner).run_powershell("Get-Date", timeout=float(CAP)) == "ok\n"
    assert runner.timeouts == [float(CAP)]


def test_nf3_check_timeout_helper_is_shared() -> None:
    # local import: check_timeout is new in S13f3 (NF2/NF3), so importing it here (not at module
    # level) lets the whole file still import against the base for fail-before.
    from aar.host.config import check_timeout

    assert check_timeout(300) == 300.0
    with pytest.raises(ConfigError):
        check_timeout(10**400)


# == NF5: push source read error is typed =========================================================


def test_nf5_push_unreadable_source_is_typed(tmp_path) -> None:
    src = tmp_path / "master.mov"
    src.write_bytes(PAYLOAD)
    os.chmod(src, 0)
    try:
        if os.access(src, os.R_OK):
            pytest.skip("running as a user that can read mode-000 files")
        with pytest.raises(HostError):
            _host(tmp_path, FakeRunner()).push(str(src), "E:\\Show\\out.mp4")
    finally:
        os.chmod(src, 0o600)


# == NF6 mutant-killers: P13 (__cause__), P15/P17/P18 (notes), P19 (no spurious note) =============


@pytest.mark.parametrize(
    "call",
    [
        lambda h, t: h.pull(REMOTE, str(t / "o")),
        lambda h, t: h.file_hash(REMOTE),
        lambda h, t: h.run_powershell("Get-Date"),
        lambda h, t: h.remote_exists(validate_windows_path(REMOTE, h.config.remote_roots_parsed)),
    ],
    ids=["pull-stat", "hash", "ps", "exists"],
)
def test_p13_runner_oserror_is_typed_with_cause(tmp_path, call) -> None:
    host = _host(tmp_path, StepRunner({0: oserr}))
    with pytest.raises(LocalIOError) as info:
        call(host, tmp_path)
    assert "could not run ssh" in str(info.value)
    assert isinstance(info.value.__cause__, BlockingIOError)  # P13


def _push(tmp_path, runner, overwrite=False):
    src = tmp_path / "master.mov"
    src.write_bytes(PAYLOAD)
    return _host(tmp_path, runner).push(str(src), "E:\\Show\\out.mp4", overwrite=overwrite)


@pytest.mark.parametrize("original", ["size", "kbi", "oserror"])
@pytest.mark.parametrize("errname", ["EACCES", "EBUSY", "EROFS", "EIO"])
def test_p15_pull_original_wins_over_unlink_error(tmp_path, monkeypatch, original, errname):
    code = getattr(errno, errname)
    real_unlink = os.unlink

    def unlink(path, *a, **k):
        if ".aarpart-" in str(path):
            raise OSError(code, os.strerror(code))
        return real_unlink(path, *a, **k)

    if original == "size":
        runner = StepRunner({}, stat_result=(len(PAYLOAD) + 1, sha256_upper(PAYLOAD)))
        want = SizeMismatch
    elif original == "kbi":
        runner, want = StepRunner({1: kbi}), KeyboardInterrupt
    else:
        runner, want = StepRunner({1: oserr}), LocalIOError
    monkeypatch.setattr(transport.os, "unlink", unlink)
    with pytest.raises(BaseException) as info:
        _host(tmp_path, runner).pull(REMOTE, str(tmp_path / "out.mp4"))
    assert type(info.value) is want
    assert any("could not remove" in n for n in getattr(info.value, "__notes__", []))  # P15


def test_p17_push_cleanup_remote_failure_is_noted(tmp_path) -> None:
    class R(StepRunner):
        def _ssh(self, script):
            from aar.host.runner import RunResult

            if "Remove-Item" in script:
                return RunResult(5, b"", b"access denied")
            return super()._ssh(script)

    with pytest.raises(HashMismatch) as info:
        _push(tmp_path, R({}, stat_result=(len(PAYLOAD), sha256_upper(b"other"))))
    assert any("Remove-Item failed" in n for n in info.value.__notes__)  # P17/P18


def test_p19_no_spurious_note_when_no_partial_exists(tmp_path) -> None:
    class R(FakeRunner):
        def _scp(self, argv):
            raise KeyboardInterrupt  # interrupted before writing a temp

    with pytest.raises(KeyboardInterrupt) as info:
        _host(tmp_path, R(stat_result=(1, sha256_upper(b"x")))).pull(REMOTE, str(tmp_path / "o"))
    assert getattr(info.value, "__notes__", []) == []  # P19


# == NF6 mutant-killers: P23 (claim unlink failure), P24 (fstat interrupt), P27 (high surrogate) ==


def _no_link(monkeypatch):
    monkeypatch.setattr(
        transport.os, "link", lambda *a, **k: (_ for _ in ()).throw(OSError(errno.EPERM, "x"))
    )


def test_p23_claim_unlink_failure_is_reported(tmp_path, monkeypatch) -> None:
    _no_link(monkeypatch)
    dest = tmp_path / "out.mp4"
    monkeypatch.setattr(
        transport.os, "replace", lambda *a, **k: (_ for _ in ()).throw(OSError(errno.EIO, "io"))
    )
    real_unlink = os.unlink

    def unlink(path, *a, **k):
        if str(path) == str(dest):
            raise PermissionError(errno.EACCES, "denied")
        return real_unlink(path, *a, **k)

    monkeypatch.setattr(transport.os, "unlink", unlink)
    with pytest.raises(LocalIOError) as info:
        _host(tmp_path, StepRunner({})).pull(REMOTE, str(dest))
    assert "left in place" in str(info.value) and dest.exists()  # P23


@pytest.mark.parametrize("exc", [lambda: OSError(errno.EIO, "io"), lambda: KeyboardInterrupt()])
def test_p24_fstat_failure_is_reported(tmp_path, monkeypatch, exc) -> None:
    _no_link(monkeypatch)
    monkeypatch.setattr(transport.os, "fstat", lambda *a, **k: (_ for _ in ()).throw(exc()))
    with pytest.raises(BaseException) as info:
        _host(tmp_path, StepRunner({})).pull(REMOTE, str(tmp_path / "out.mp4"))
    text = str(info.value) + " ".join(getattr(info.value, "__notes__", []))
    assert "left in place" in text and str(tmp_path / "out.mp4") in text  # P24
    assert not [p for p in tmp_path.iterdir() if ".aarpart-" in p.name]


@pytest.mark.parametrize("bad", ["\ud800", "\udfff", "\udcff", "\ud83d\ude00"])
def test_p27_surrogates_refused_high_and_low(tmp_path, bad) -> None:
    with pytest.raises(PathRefused) as info:
        validate_windows_path(f"E:\\Show\\x{bad}.mp4", (normalize_root("E:\\Show"),))
    assert info.value.reason == "invalid-unicode"  # P27 (high surrogate too)


# == NF7: a vanished claim is not reported as "left in place" =====================================


def test_nf7_vanished_claim_is_not_reported_as_left_in_place(tmp_path, monkeypatch) -> None:
    _no_link(monkeypatch)
    dest = tmp_path / "out.mp4"
    real_replace = os.replace

    def replace(src, dst):
        if str(dst) == str(dest):
            os.unlink(dest)  # our claim vanishes before we look
            raise OSError(errno.EIO, "io")
        return real_replace(src, dst)

    monkeypatch.setattr(transport.os, "replace", replace)
    with pytest.raises(LocalIOError) as info:
        _host(tmp_path, StepRunner({})).pull(REMOTE, str(dest))
    text = str(info.value) + " ".join(getattr(info.value, "__notes__", []))
    assert not dest.exists()
    assert "left in place" not in text, f"vanished claim reported as left in place: {text}"


def test_nf7_unlink_if_claim_vanished_returns_true(tmp_path) -> None:
    p = tmp_path / "c"
    fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    claim = os.fstat(fd)
    os.close(fd)
    os.unlink(p)  # gone
    assert _unlink_if_claim(str(p), claim) is True


# == nits: NUL in powershell_argv; NF8 DESIGN smart quote =========================================


def test_nit_nul_refused_in_powershell_argv() -> None:
    with pytest.raises(ValueError, match="NUL"):
        powershell_argv("Get-Date\x00")


def test_f1_toggle_cli_still_refused(tmp_path, monkeypatch, capsys) -> None:
    # guard that the removed toggle still refuses through the CLI (no regression)
    runner = FakeRunner()
    _cli_env(tmp_path, monkeypatch, runner, AAR_EDIT_SCP_SFTP="0")
    assert main(["host", "hash", "E:\\Show\\x.mp4", "--json"]) == 2
    assert json.loads(capsys.readouterr().out)["error"]["type"] == "config-error"
    assert runner.calls == []


def test_nf8_design_smart_quote_is_u2019() -> None:
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(transport.__file__))))
    design = os.path.join(root, "docs", "DESIGN.md")
    with open(design, encoding="utf-8") as fh:
        text = fh.read()
    assert "Don\u2019t" in text, "the DESIGN smart-quote example must use U+2019, not an ASCII '"
    assert "`Don't ...`" not in text


def test_config_does_not_expose_scp_sftp() -> None:
    from dataclasses import fields

    assert "scp_use_sftp" not in {f.name for f in fields(EditHostConfig)}
