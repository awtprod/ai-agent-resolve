"""Edit-PC transport: configuration, a runner seam, PowerShell, and verified scp transfers.

Cold work runs on the hub; Resolve, renders and masters stay on the edit PC (DESIGN R6). This
package is the one module that moves files between them and runs PowerShell there. The host is
named only by configuration (:class:`EditHostConfig`, or ``AAR_EDIT_*``); there is no default
host in code. Every path is validated against allowed roots before any command runs, and every
transfer is checked by size and SHA-256.
"""

from __future__ import annotations

from aar.host.config import EditHostConfig, env_config
from aar.host.errors import (
    ConfigError,
    DestinationExists,
    HashMismatch,
    HostError,
    PathRefused,
    RemoteCommandError,
    ScriptTooLong,
    SizeMismatch,
    TransportError,
    TransportTimeout,
)
from aar.host.pathmap import HostPathMap
from aar.host.runner import Runner, RunResult, SubprocessRunner
from aar.host.transport import EditHost, TransferResult

__all__ = [
    "ConfigError",
    "DestinationExists",
    "EditHost",
    "EditHostConfig",
    "HashMismatch",
    "HostError",
    "HostPathMap",
    "PathRefused",
    "RemoteCommandError",
    "RunResult",
    "Runner",
    "ScriptTooLong",
    "SizeMismatch",
    "SubprocessRunner",
    "TransferResult",
    "TransportError",
    "TransportTimeout",
    "env_config",
]
