"""Host configuration: explicit object, env parsing, and the 'no default host' rule."""

from __future__ import annotations

import pytest

from aar.host.config import EditHostConfig, env_config
from aar.host.errors import ConfigError


def test_valid_config_parses_roots(tmp_path) -> None:
    cfg = EditHostConfig(
        alias="edit-pc",
        ssh_config=str(tmp_path / "ssh.cfg"),
        remote_roots=("E:\\Show", "F:\\CCN"),
        local_roots=(str(tmp_path),),
    )
    assert cfg.path_form == "drive"
    assert cfg.connect_timeout == 10
    assert len(cfg.remote_roots_parsed) == 2
    # local roots are canonicalised to realpaths
    assert cfg.local_roots == (str(tmp_path.resolve()),)


@pytest.mark.parametrize(
    "overrides",
    [
        {"alias": "-edit"},  # looks like an option
        {"alias": "edit:pc"},  # colon breaks alias:spec
        {"alias": ""},
        {"ssh_config": ""},
        {"path_form": "weird"},
        {"connect_timeout": 0},
        {"connect_timeout": -5},
        {"remote_roots": ()},
        {"local_roots": ()},
        {"remote_roots": ("Show",)},  # relative root
        {"remote_roots": ("E:\\CON",)},  # reserved device in root
        {"local_roots": ("relative/dir",)},  # not absolute
    ],
)
def test_invalid_config_is_rejected(tmp_path, overrides) -> None:
    params = {
        "alias": "edit-pc",
        "ssh_config": str(tmp_path / "ssh.cfg"),
        "remote_roots": ("E:\\Show",),
        "local_roots": (str(tmp_path),),
    }
    params.update(overrides)
    with pytest.raises(ConfigError):
        EditHostConfig(**params)


def test_env_config_requires_every_variable() -> None:
    # no default host: an empty environment fails and names what is missing
    with pytest.raises(ConfigError) as info:
        env_config({})
    for name in (
        "AAR_EDIT_SSH_ALIAS",
        "AAR_EDIT_SSH_CONFIG",
        "AAR_EDIT_REMOTE_ROOTS",
        "AAR_EDIT_LOCAL_ROOTS",
    ):
        assert name in str(info.value)


def test_env_config_json_array_roots(tmp_path) -> None:
    cfg = env_config(
        {
            "AAR_EDIT_SSH_ALIAS": "edit-pc",
            "AAR_EDIT_SSH_CONFIG": str(tmp_path / "ssh.cfg"),
            "AAR_EDIT_REMOTE_ROOTS": '["E:\\\\Show", "F:\\\\CCN"]',
            "AAR_EDIT_LOCAL_ROOTS": f'["{tmp_path}"]',
            "AAR_EDIT_PATH_FORM": "slash",
            "AAR_EDIT_CONNECT_TIMEOUT": "7",
        }
    )
    assert cfg.remote_roots == ("E:\\Show", "F:\\CCN")
    assert cfg.path_form == "slash"
    assert cfg.connect_timeout == 7


def test_env_config_newline_roots(tmp_path) -> None:
    cfg = env_config(
        {
            "AAR_EDIT_SSH_ALIAS": "edit-pc",
            "AAR_EDIT_SSH_CONFIG": str(tmp_path / "ssh.cfg"),
            "AAR_EDIT_REMOTE_ROOTS": "E:\\Show\nF:\\CCN",
            "AAR_EDIT_LOCAL_ROOTS": str(tmp_path),
        }
    )
    assert cfg.remote_roots == ("E:\\Show", "F:\\CCN")
    assert cfg.path_form == "drive"


def test_env_config_bad_timeout(tmp_path) -> None:
    with pytest.raises(ConfigError):
        env_config(
            {
                "AAR_EDIT_SSH_ALIAS": "edit-pc",
                "AAR_EDIT_SSH_CONFIG": str(tmp_path / "ssh.cfg"),
                "AAR_EDIT_REMOTE_ROOTS": "E:\\Show",
                "AAR_EDIT_LOCAL_ROOTS": str(tmp_path),
                "AAR_EDIT_CONNECT_TIMEOUT": "soon",
            }
        )
