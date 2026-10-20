"""F1: scp always runs with -s (SFTP); the AAR_EDIT_SCP_SFTP toggle is gone and refused if set."""

from __future__ import annotations

import pytest
from host_support import FakeRunner, make_config, sha256_upper

from aar.host.config import env_config
from aar.host.errors import ConfigError
from aar.host.transport import EditHost


def _env(tmp_path, **extra):
    env = {
        "AAR_EDIT_SSH_ALIAS": "edit-pc",
        "AAR_EDIT_SSH_CONFIG": str(tmp_path / "ssh.cfg"),
        "AAR_EDIT_REMOTE_ROOTS": "E:\\Show",
        "AAR_EDIT_LOCAL_ROOTS": str(tmp_path),
    }
    env.update(extra)
    return env


def test_scp_base_always_has_s(tmp_path) -> None:
    base = EditHost(make_config(tmp_path), FakeRunner())._scp_base()
    assert base[:2] == ["scp", "-s"] and base.count("-s") == 1 and "-O" not in base


def test_pull_scp_argv_has_s(tmp_path) -> None:
    runner = FakeRunner(stat_result=(1, sha256_upper(b"x")), scp_payload=b"x")
    host = EditHost(make_config(tmp_path), runner, token_factory=lambda: "T")
    host.pull("E:\\Show\\c.mp4", str(tmp_path / "o"))
    scp = next(c for c in runner.calls if c[0] == "scp")
    assert "-s" in scp


def test_push_scp_argv_has_s(tmp_path) -> None:
    src = tmp_path / "m.mov"
    src.write_bytes(b"x")
    runner = FakeRunner(stat_result=(1, sha256_upper(b"x")))
    host = EditHost(make_config(tmp_path), runner, token_factory=lambda: "T")
    host.push(str(src), "E:\\Show\\o.mp4")
    scp = next(c for c in runner.calls if c[0] == "scp")
    assert "-s" in scp


@pytest.mark.parametrize("raw", ["0", "1", "false", "true", "no", "on", ""])
def test_removed_env_toggle_is_a_typed_refusal(tmp_path, raw) -> None:
    with pytest.raises(ConfigError):
        env_config(_env(tmp_path, AAR_EDIT_SCP_SFTP=raw))


def test_legacy_raw_shape_cannot_be_produced(tmp_path) -> None:
    """The N4 legacy shape (scp spec without -s, so a metacharacter name reaches the remote shell)
    can no longer be produced: the toggle is refused and every scp argv carries -s even for a name
    with shell metacharacters like `a & calc.mp4`."""
    with pytest.raises(ConfigError):
        env_config(_env(tmp_path, AAR_EDIT_SCP_SFTP="0"))
    runner = FakeRunner(stat_result=(1, sha256_upper(b"x")), scp_payload=b"x")
    host = EditHost(make_config(tmp_path), runner, token_factory=lambda: "T")
    host.pull("E:\\Show\\a & calc.mp4", str(tmp_path / "o"))
    scp = next(c for c in runner.calls if c[0] == "scp")
    assert "-s" in scp and "-O" not in scp
