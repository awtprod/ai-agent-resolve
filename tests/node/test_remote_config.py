"""RemoteConfig validation and token-file handling (S15a item 5): no creds, no leak, mode check."""

from __future__ import annotations

import os
import secrets
from pathlib import Path

import pytest

from aar.node.remote import RemoteConfig, _NodeClient, _read_token


def test_hub_url_must_be_http_or_https_without_credentials(tmp_path: Path) -> None:
    token_file = tmp_path / "t"
    token_file.write_text("x")
    for bad in ("ftp://host:8789", "file:///x", "ws://host", "http://"):
        with pytest.raises(ValueError):
            RemoteConfig(hub_url=bad, token_file=token_file)
    with pytest.raises(ValueError, match="credentials"):
        RemoteConfig(hub_url="http://user:pass@host:8789", token_file=token_file)
    # A clean http/https URL is accepted.
    RemoteConfig(hub_url="http://host:8789", token_file=token_file)
    RemoteConfig(hub_url="https://host/base", token_file=token_file)


def test_timeouts_and_cap_are_validated(tmp_path: Path) -> None:
    token_file = tmp_path / "t"
    token_file.write_text("x")
    with pytest.raises(ValueError):
        RemoteConfig(hub_url="http://h", token_file=token_file, connect_timeout=0)
    with pytest.raises(ValueError):
        RemoteConfig(hub_url="http://h", token_file=token_file, read_timeout=-1)
    with pytest.raises(ValueError):
        RemoteConfig(hub_url="http://h", token_file=token_file, response_cap=0)


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits")
def test_token_file_must_not_be_group_or_other_readable(tmp_path: Path) -> None:
    token = secrets.token_urlsafe(32)
    path = tmp_path / "token"
    path.write_text(token)
    path.chmod(0o644)  # group- and other-readable: refused, like ssh
    with pytest.raises(ValueError) as exc:
        _read_token(path)
    # The refusal names the path and the mode, never the token itself.
    assert token not in str(exc.value)
    assert "readable" in str(exc.value)
    # 0o600 is accepted and returns the token.
    path.chmod(0o600)
    assert _read_token(path) == token


def test_token_is_not_in_the_config_repr(tmp_path: Path) -> None:
    token = secrets.token_urlsafe(32)
    path = tmp_path / "token"
    path.write_text(token)
    path.chmod(0o600)
    config = RemoteConfig(hub_url="http://host:8789", token_file=path)
    # The config holds only the path; the token is read on demand, never stored or repr'd.
    assert token not in repr(config)
    client = _NodeClient(config)
    assert token not in repr(client)


def test_empty_token_file_is_refused_without_leaking(tmp_path: Path) -> None:
    path = tmp_path / "token"
    path.write_text("   \n")
    path.chmod(0o600)
    with pytest.raises(ValueError, match="empty"):
        _read_token(path)
