"""Edit-PC host configuration: the host exists only here, never as a default in code.

A config names the edit PC by an ssh alias and an ssh config file (passed to ``ssh -F``), lists
the allowed remote (Windows) roots and local (hub) roots that bound every transfer, picks the
scp path form, and sets the connect/command timeouts. It can be built explicitly by a caller or
read from ``AAR_EDIT_*`` environment variables. The ssh config file is referenced by path only;
its contents and any key material are never read here.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field

from aar.host.errors import ConfigError, PathRefused
from aar.host.powershell import DEFAULT_REMOTE_CMDLINE_MAX
from aar.host.winpath import PathForm, _Win, normalize_root

__all__ = ["MAX_TIMEOUT_SECONDS", "EditHostConfig", "env_config"]

# Upper bound on any timeout (seconds). 7 days: long enough for any real master transfer, finite
# enough that inf/nan/1e300 never reach subprocess/selectors (which raise OverflowError there).
MAX_TIMEOUT_SECONDS = 7 * 24 * 60 * 60  # 604800

_ENV_ALIAS = "AAR_EDIT_SSH_ALIAS"
_ENV_CONFIG = "AAR_EDIT_SSH_CONFIG"
_ENV_REMOTE_ROOTS = "AAR_EDIT_REMOTE_ROOTS"
_ENV_LOCAL_ROOTS = "AAR_EDIT_LOCAL_ROOTS"
_ENV_PATH_FORM = "AAR_EDIT_PATH_FORM"
_ENV_CONNECT_TIMEOUT = "AAR_EDIT_CONNECT_TIMEOUT"
_ENV_COMMAND_TIMEOUT = "AAR_EDIT_COMMAND_TIMEOUT"
_ENV_CMDLINE_MAX = "AAR_EDIT_CMDLINE_MAX"
_ENV_SCP_SFTP = "AAR_EDIT_SCP_SFTP"

_PATH_FORMS = ("drive", "slash")
# A strict ssh host/alias: an alphanumeric start, then alphanumerics, dot, underscore, @, hyphen.
# This refuses the characters that change scp's host:path splitting -- '/', '[', ':', spaces -- and
# a leading '-' (which would look like an option).
_ALIAS_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._@-]*$")


def _check_token(value: str, name: str) -> str:
    """A scalar that becomes an argv token: no control chars, no leading ``-``."""
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{name} is required")
    if any(ord(ch) < 0x20 for ch in value):
        raise ConfigError(f"{name} contains a control character")
    if value.startswith("-"):
        raise ConfigError(
            f"{name} may not start with '-' (it would look like an option): {value!r}"
        )
    return value


def _check_int(value: object, name: str) -> None:
    """Require a positive ``int`` and reject ``bool`` (which is an ``int`` subclass)."""
    if type(value) is bool or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class EditHostConfig:
    """A validated edit-PC transport configuration.

    ``remote_roots`` are Windows prefixes (drive or UNC); ``local_roots`` are absolute hub
    directories. A transfer outside both is refused before any command runs. ``path_form``
    selects the scp remote spelling (``drive`` -> ``alias:E:/...``, ``slash`` -> ``alias:/E:/...``).
    ``path_map`` optionally pairs edit-PC prefixes with hub prefixes for :class:`PathMapper`.
    """

    alias: str
    ssh_config: str
    remote_roots: tuple[str, ...]
    local_roots: tuple[str, ...]
    path_form: PathForm = "drive"
    connect_timeout: int = 10
    command_timeout: float = 300.0
    cmdline_max: int = DEFAULT_REMOTE_CMDLINE_MAX
    path_map: tuple[tuple[str, str], ...] = ()
    _remote_roots_parsed: tuple[_Win, ...] = field(default=(), repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.alias, str) or not _ALIAS_RE.match(self.alias):
            raise ConfigError(
                f"{_ENV_ALIAS} must match {_ALIAS_RE.pattern} (a plain ssh host/alias): "
                f"{self.alias!r}"
            )
        _check_token(self.ssh_config, _ENV_CONFIG)
        if self.path_form not in _PATH_FORMS:
            raise ConfigError(f"path form must be one of {_PATH_FORMS}, not {self.path_form!r}")
        _check_int(self.connect_timeout, "connect timeout")
        ct = self.command_timeout
        if (
            type(ct) is bool
            or not isinstance(ct, (int, float))
            or not math.isfinite(ct)
            or not (0 < ct <= MAX_TIMEOUT_SECONDS)
        ):
            raise ConfigError(
                f"command timeout must be a finite number of seconds in (0, {MAX_TIMEOUT_SECONDS}]"
            )
        _check_int(self.cmdline_max, "cmdline max")
        if not self.remote_roots:
            raise ConfigError("at least one remote root is required")
        if not self.local_roots:
            raise ConfigError("at least one local root is required")
        try:
            parsed = tuple(normalize_root(root) for root in self.remote_roots)
        except PathRefused as exc:
            raise ConfigError(f"invalid remote root {exc.path!r}: {exc.reason}") from exc
        object.__setattr__(self, "_remote_roots_parsed", parsed)
        local = tuple(self._normalize_local(root) for root in self.local_roots)
        object.__setattr__(self, "local_roots", local)

    @staticmethod
    def _normalize_local(root: str) -> str:
        if not isinstance(root, str) or not root:
            raise ConfigError("a local root must be a non-empty path")
        if "\x00" in root:
            raise ConfigError("a local root may not contain NUL")
        if not os.path.isabs(root):
            raise ConfigError(f"a local root must be absolute: {root!r}")
        return os.path.realpath(root)

    @property
    def remote_roots_parsed(self) -> tuple[_Win, ...]:
        """The remote roots, parsed and validated (built once in ``__post_init__``)."""
        return self._remote_roots_parsed


def _split_list(raw: str) -> list[str]:
    """Parse a root list from one env var: a JSON array, or newline-separated entries."""
    text = raw.strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"invalid JSON list: {exc}") from exc
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            raise ConfigError("a root list JSON value must be a list of strings")
        return value
    return [line.strip() for line in text.splitlines() if line.strip()]


def env_config(env: Mapping[str, str] | None = None) -> EditHostConfig:
    """Build an :class:`EditHostConfig` from ``AAR_EDIT_*`` variables (default ``os.environ``).

    Raises :class:`ConfigError` if a required variable is missing or a value is invalid. There is
    no default host: without ``AAR_EDIT_SSH_ALIAS`` and the roots this fails.
    """
    env = os.environ if env is None else env
    missing = [
        name
        for name in (_ENV_ALIAS, _ENV_CONFIG, _ENV_REMOTE_ROOTS, _ENV_LOCAL_ROOTS)
        if not env.get(name)
    ]
    if missing:
        raise ConfigError(f"missing required environment variables: {', '.join(missing)}")
    if _ENV_SCP_SFTP in env:
        # The SFTP toggle was removed (S13f2 F1): scp always runs with -s. A still-set value means
        # someone expects legacy SCP, which is unsafe, so refuse rather than silently ignore it.
        raise ConfigError(
            f"{_ENV_SCP_SFTP} is no longer supported: scp always uses SFTP mode (-s). "
            f"Unset {_ENV_SCP_SFTP}. Legacy SCP is unsupported because it is unsafe "
            "(the remote shell parses the path); the hub needs OpenSSH >= 8.7."
        )
    connect = env.get(_ENV_CONNECT_TIMEOUT)
    command = env.get(_ENV_COMMAND_TIMEOUT)
    try:
        connect_timeout = int(connect) if connect else 10
    except ValueError as exc:
        raise ConfigError(f"{_ENV_CONNECT_TIMEOUT} must be an integer: {connect!r}") from exc
    try:
        command_timeout = float(command) if command else 300.0
    except ValueError as exc:
        raise ConfigError(f"{_ENV_COMMAND_TIMEOUT} must be a number: {command!r}") from exc
    cmdline = env.get(_ENV_CMDLINE_MAX)
    try:
        cmdline_max = int(cmdline) if cmdline else DEFAULT_REMOTE_CMDLINE_MAX
    except ValueError as exc:
        raise ConfigError(f"{_ENV_CMDLINE_MAX} must be an integer: {cmdline!r}") from exc
    form = env.get(_ENV_PATH_FORM) or "drive"
    return EditHostConfig(
        alias=env[_ENV_ALIAS],
        ssh_config=env[_ENV_CONFIG],
        remote_roots=tuple(_split_list(env[_ENV_REMOTE_ROOTS])),
        local_roots=tuple(_split_list(env[_ENV_LOCAL_ROOTS])),
        path_form=form,  # type: ignore[arg-type]  # validated in __post_init__
        connect_timeout=connect_timeout,
        command_timeout=command_timeout,
        cmdline_max=cmdline_max,
    )
