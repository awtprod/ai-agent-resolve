"""S13f2 LOW-2 (strict remote parsing), F3 (bounded numeric config), F7 (alias fullmatch),
N1 (equal-to-root / trailing separator, incl. mixed case), F6 (lone surrogate), N3 (undecodable
.ps1). Adapted from the validator's test_v3_parse_config. Fake runner only."""

from __future__ import annotations

import json
import math

import pytest
from host_support import FakeRunner, make_config

from aar.cli import main
from aar.host import cli as host_cli
from aar.host.config import env_config
from aar.host.errors import ConfigError, PathRefused, RemoteCommandError
from aar.host.runner import RunResult
from aar.host.transport import EditHost
from aar.host.winpath import normalize_root, validate_windows_path

H64 = "A" * 64


class OutRunner(FakeRunner):
    def __init__(self, out: bytes, **kw):
        super().__init__(**kw)
        self.out = out

    def _ssh(self, script):
        return RunResult(0, self.out, b"")


def _kind(tmp_path, out: bytes):
    host = EditHost(make_config(tmp_path), OutRunner(out))
    win = validate_windows_path("E:\\Show\\x.mp4", host.config.remote_roots_parsed)
    return host.remote_kind(win)


def _stat(tmp_path, out: bytes):
    return EditHost(make_config(tmp_path), OutRunner(out)).file_hash("E:\\Show\\x.mp4")


# -- LOW-2: remote_kind -----------------------------------------------------------------------

KIND_BAD = [
    b"AAREXISTS\r\n",
    b"AAREXISTS file extra\r\n",
    b"AAREXISTS FILE\r\n",
    b"AAREXISTS File\r\n",
    b"AAREXISTS maybe\r\n",
    "\ufeffAAREXISTS file\r\n".encode(),
    b"",
    b"aarexists file\r\n",
    "AAREXISTS f\u0131le\r\n".encode(),  # dotless i
]
KIND_OK = [
    (b"AAREXISTS file\r\n", "file"),
    (b"AAREXISTS none\n", "none"),
    (b"AAREXISTS container\r", "container"),
    (b"noise\r\nAAREXISTS file\r\n", "file"),
    (b"  AAREXISTS   file  \r\n", "file"),
]


@pytest.mark.parametrize("out", KIND_BAD)
def test_remote_kind_strict_refusals(tmp_path, out) -> None:
    with pytest.raises(RemoteCommandError):
        _kind(tmp_path, out)


@pytest.mark.parametrize(("out", "want"), KIND_OK)
def test_remote_kind_accepts(tmp_path, out, want) -> None:
    assert _kind(tmp_path, out) == want


# -- LOW-2: remote_stat (fullwidth digits, prefix match, 65-char hash, kill N33/N35/N36) -----

STAT_BAD = [
    b"AARSTAT\r\n",
    b"AARSTAT 10\r\n",
    f"AARSTAT 10 {H64} extra\r\n".encode(),
    f"AARSTAT \u00b2 {H64}\r\n".encode(),  # superscript two
    f"AARSTAT \uff11\uff10 {H64}\r\n".encode(),  # fullwidth 10
    f"AARSTAT \u0661\u0660 {H64}\r\n".encode(),  # arabic-indic 10 (int() would accept)
    f"AARSTAT -1 {H64}\r\n".encode(),
    f"AARSTAT +1 {H64}\r\n".encode(),
    f"AARSTAT 1_0 {H64}\r\n".encode(),
    f"AARSTAT 10 {'A' * 63}\r\n".encode(),
    f"AARSTAT 10 {'A' * 65}\r\n".encode(),
    f"AARSTAT 10 {'G' * 64}\r\n".encode(),
    f"AARSTAT 10 {chr(0xFF21) * 64}\r\n".encode(),  # fullwidth A
    f"AARSTAT 10 0x{'A' * 62}\r\n".encode(),
    f"\ufeffAARSTAT 10 {H64}\r\n".encode(),
    b"AARSTAT  \r\n",
]


@pytest.mark.parametrize("out", STAT_BAD)
def test_remote_stat_strict_refusals(tmp_path, out) -> None:
    with pytest.raises(RemoteCommandError):
        _stat(tmp_path, out)


def test_remote_stat_accepts_crlf_and_lowercase(tmp_path) -> None:
    res = _stat(tmp_path, f"AARSTAT 0010 {'a' * 64}\r\n".encode())
    assert res.size == 10 and res.sha256 == "A" * 64


# -- F3 / LOW-4: numeric config ---------------------------------------------------------------

BAD_INTS = [True, False, 0, -1, math.nan, math.inf, -math.inf, 10.0, "10", None]


@pytest.mark.parametrize("field", ["connect_timeout", "cmdline_max"])
@pytest.mark.parametrize("value", BAD_INTS, ids=repr)
def test_int_fields_refuse_bad_values(tmp_path, field, value) -> None:
    with pytest.raises(ConfigError):
        make_config(tmp_path, **{field: value})


@pytest.mark.parametrize(
    "value", [True, False, 0, -1, -0.0, math.nan, math.inf, -math.inf, 1e300, "10", None], ids=repr
)
def test_command_timeout_refuses_bad_values(tmp_path, value) -> None:
    with pytest.raises(ConfigError):
        make_config(tmp_path, command_timeout=value)


def test_command_timeout_accepts_a_finite_positive(tmp_path) -> None:
    assert make_config(tmp_path, command_timeout=300.0).command_timeout == 300.0


def _env(tmp_path, **extra):
    env = {
        "AAR_EDIT_SSH_ALIAS": "edit-pc",
        "AAR_EDIT_SSH_CONFIG": str(tmp_path / "ssh.cfg"),
        "AAR_EDIT_REMOTE_ROOTS": "E:\\Show",
        "AAR_EDIT_LOCAL_ROOTS": str(tmp_path),
    }
    env.update(extra)
    return env


@pytest.mark.parametrize(
    ("var", "raw"),
    [
        ("AAR_EDIT_CONNECT_TIMEOUT", "True"),
        ("AAR_EDIT_CONNECT_TIMEOUT", "0"),
        ("AAR_EDIT_CONNECT_TIMEOUT", "-5"),
        ("AAR_EDIT_CONNECT_TIMEOUT", "1.5"),
        ("AAR_EDIT_CONNECT_TIMEOUT", "nan"),
        ("AAR_EDIT_COMMAND_TIMEOUT", "True"),
        ("AAR_EDIT_COMMAND_TIMEOUT", "0"),
        ("AAR_EDIT_COMMAND_TIMEOUT", "-1"),
        ("AAR_EDIT_COMMAND_TIMEOUT", "nan"),
        ("AAR_EDIT_COMMAND_TIMEOUT", "inf"),
        ("AAR_EDIT_COMMAND_TIMEOUT", "Infinity"),
        ("AAR_EDIT_COMMAND_TIMEOUT", "1e400"),
        ("AAR_EDIT_CMDLINE_MAX", "True"),
        ("AAR_EDIT_CMDLINE_MAX", "inf"),
    ],
)
def test_env_bad_numbers_refused(tmp_path, var, raw) -> None:
    with pytest.raises(ConfigError):
        env_config(_env(tmp_path, **{var: raw}))


def test_env_unicode_digits_normalise(tmp_path) -> None:
    cfg = env_config(_env(tmp_path, AAR_EDIT_CONNECT_TIMEOUT="\u0661\u0660"))  # arabic-indic 10
    assert cfg.connect_timeout == 10
    assert "ConnectTimeout=10" in EditHost(cfg, FakeRunner())._ssh_base()


# -- F7: alias ---------------------------------------------------------------------------------

ALIAS_REFUSED = [
    "edit/pc",
    "[edit",
    "edit pc",
    "edit:pc",
    "-edit",
    "",
    "edit\tpc",
    "\u00e9dit",
    "edit\u3000pc",
    "@edit",
    ".edit",
    "_edit",
    "edit\\pc",
    "edit;pc",
    "edit$pc",
    "edit[pc",
    "a@[x",
    "edit]pc",
    "edit-pc\n",
    "edit-pc\n\n",
]


@pytest.mark.parametrize("alias", ALIAS_REFUSED, ids=repr)
def test_alias_refused(tmp_path, alias) -> None:
    with pytest.raises(ConfigError):
        make_config(tmp_path, alias=alias)


def test_env_alias_trailing_newline_refused(tmp_path) -> None:
    with pytest.raises(ConfigError):
        env_config(_env(tmp_path, AAR_EDIT_SSH_ALIAS="edit-pc\n"))


@pytest.mark.parametrize("alias", ["edit-pc", "e", "user@edit-pc", "a@b@c", "a.b", "a-", "a_"])
def test_alias_accepted(tmp_path, alias) -> None:
    assert make_config(tmp_path, alias=alias).alias == alias


# -- N1: equal-to-root (incl. mixed case) and trailing separator ------------------------------

N1_REFUSED = [
    ("E:\\Show", "E:\\Show"),
    ("E:\\Show", "e:/show"),  # mixed case equal-to-root (kills N47)
    ("E:\\Show", "E:\\SHOW\\"),
    ("E:\\Show", "E:\\Show\\x.mp4\\"),
    ("E:\\Show", "E:\\Show\\x.mp4/"),
    ("E:", "E:\\"),
    ("E:", "E:"),
    ("\\\\srv\\share", "\\\\srv\\share"),
    ("\\\\srv\\share", "\\\\SRV\\SHARE"),  # mixed case UNC equal-to-root
    ("E:\\Caf\u00e9", "E:\\CAF\u00c9"),  # NTFS-folded equal-to-root
]


@pytest.mark.parametrize(("root", "path"), N1_REFUSED, ids=repr)
def test_n1_refused(root, path) -> None:
    with pytest.raises(PathRefused):
        validate_windows_path(path, (normalize_root(root),))


def test_n1_mixed_case_equal_root_reason(tmp_path) -> None:
    with pytest.raises(PathRefused) as info:
        validate_windows_path("e:/show", (normalize_root("E:\\Show"),))
    assert info.value.reason == "equals-allowed-root"


@pytest.mark.parametrize(
    ("roots", "path"),
    [
        (("E:",), "E:\\x.mp4"),
        (("\\\\srv\\share",), "\\\\srv\\share\\x.mp4"),
        (("E:\\Show", "E:\\Show\\Sub"), "E:\\Show\\Sub\\x.mp4"),
    ],
)
def test_n1_accepted_directly_under_a_root(roots, path) -> None:
    validate_windows_path(path, tuple(normalize_root(r) for r in roots))


# -- F6: lone surrogate ------------------------------------------------------------------------


def test_lone_surrogate_validate_is_refused() -> None:
    with pytest.raises(PathRefused) as info:
        validate_windows_path("E:\\Show\\bad\udcff.mp4", (normalize_root("E:\\Show"),))
    assert info.value.reason == "invalid-unicode"


def test_lone_surrogate_in_root_is_refused() -> None:
    with pytest.raises(PathRefused):
        normalize_root("E:\\Sh\udcffow")


def _cli_env(tmp_path, monkeypatch, runner):
    for k, v in _env(tmp_path).items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(host_cli, "_default_runner", lambda: runner)


def test_lone_surrogate_cli_is_typed_refusal(tmp_path, monkeypatch, capsys) -> None:
    runner = FakeRunner()
    _cli_env(tmp_path, monkeypatch, runner)
    code = main(["host", "hash", "E:\\Show\\bad\udcff.mp4", "--json"])
    assert code in (1, 2)
    assert runner.calls == []  # nothing sent


# -- N3: undecodable .ps1 ---------------------------------------------------------------------


@pytest.mark.parametrize("as_json", [True, False])
@pytest.mark.parametrize(
    "data",
    [
        "Get-Date\n".encode("utf-16"),  # UTF-16LE with BOM
        "Get-Date\n".encode("utf-16-be"),
        b"\xff\xfe\x00",
        "caf\u00e9\n".encode("latin-1"),
        "Get-Date\n".encode("utf-16-le"),  # BOM-less UTF-16 -> NULs -> refused
    ],
)
def test_ps_undecodable_is_exit_two(tmp_path, monkeypatch, capsys, data, as_json) -> None:
    script = tmp_path / "s.ps1"
    script.write_bytes(data)
    runner = FakeRunner(ps_stdout=b"x")
    _cli_env(tmp_path, monkeypatch, runner)
    argv = ["host", "ps", str(script)] + (["--json"] if as_json else [])
    assert main(argv) == 2
    out = capsys.readouterr()
    if as_json:
        assert json.loads(out.out)["ok"] is False
    else:
        assert "error" in out.err
    assert runner.calls == []


def test_ps_utf8_bom_is_stripped(tmp_path, monkeypatch) -> None:
    from host_support import decode_encoded_command

    script = tmp_path / "s.ps1"
    script.write_bytes(b"\xef\xbb\xbfGet-Date\n")
    runner = FakeRunner(ps_stdout=b"x")
    _cli_env(tmp_path, monkeypatch, runner)
    assert main(["host", "ps", str(script)]) == 0
    sent = decode_encoded_command(runner.calls[0])
    assert not sent.startswith("\ufeff") and sent == "Get-Date\n"
