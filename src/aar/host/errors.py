"""Typed errors for the edit-PC transport.

Every failure that is not a plain programming bug is one of these, so callers (and the CLI) can
map them to stable exit codes and JSON without parsing messages. Input/configuration problems
(:class:`ConfigError`, :class:`PathRefused`, :class:`ScriptTooLong`) are refusals raised before
anything is run; the rest are transfer failures (:class:`TransportError` and subclasses).
"""

from __future__ import annotations

__all__ = [
    "ConfigError",
    "DestinationExists",
    "HashMismatch",
    "HostError",
    "PathRefused",
    "RemoteCommandError",
    "ScriptTooLong",
    "SizeMismatch",
    "TransportError",
    "TransportTimeout",
]


class HostError(Exception):
    """Base class for every edit-PC transport error."""

    kind = "host-error"


class ConfigError(HostError):
    """The host configuration is missing or unusable (no alias, bad root, unknown path form)."""

    kind = "config-error"


class PathRefused(HostError):
    """A path was rejected before any command ran. ``reason`` is a stable machine token."""

    kind = "path-refused"

    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"path refused ({reason}): {path!r}")


class ScriptTooLong(HostError):
    """The encoded PowerShell command line would exceed the Windows command-line limit."""

    kind = "script-too-long"

    def __init__(self, length: int, limit: int) -> None:
        self.length = length
        self.limit = limit
        super().__init__(
            f"the encoded PowerShell command line is {length} characters, over the Windows "
            f"command-line limit of {limit}"
        )


class TransportError(HostError):
    """Base class for a transfer that started but did not complete correctly."""

    kind = "transport-error"


class SizeMismatch(TransportError):
    """The transferred file's size did not match the source."""

    kind = "size-mismatch"

    def __init__(self, expected: int, actual: int) -> None:
        self.expected = expected
        self.actual = actual
        super().__init__(f"size mismatch: expected {expected} bytes, got {actual}")


class HashMismatch(TransportError):
    """The transferred file's SHA-256 did not match the source."""

    kind = "hash-mismatch"

    def __init__(self, expected: str, actual: str) -> None:
        self.expected = expected
        self.actual = actual
        super().__init__(f"sha256 mismatch: expected {expected}, got {actual}")


class TransportTimeout(TransportError):
    """A remote command or transfer exceeded its timeout."""

    kind = "transport-timeout"


class RemoteCommandError(TransportError):
    """A remote command exited non-zero. ``stderr`` is already cleaned (CRLF/CLIXML)."""

    kind = "remote-command-error"

    def __init__(self, returncode: int, stderr: str) -> None:
        self.returncode = returncode
        self.stderr = stderr
        trimmed = stderr.strip()
        detail = f": {trimmed}" if trimmed else ""
        super().__init__(f"remote command exited {returncode}{detail}")


class DestinationExists(TransportError):
    """The destination already exists and the caller did not ask to overwrite it."""

    kind = "destination-exists"

    def __init__(self, destination: str) -> None:
        self.destination = destination
        super().__init__(f"destination already exists: {destination}")
