"""Validate and normalise edit-PC (Windows) paths, and build scp remote specs.

Every path from a caller is treated as hostile. A path is accepted only if it is an absolute
drive path (or a UNC path when the configuration's roots include one), has no ``.`` / ``..`` /
empty components, no reserved device name, no illegal or control character, no trailing dot or
space, and resolves inside an allowed root under NTFS case rules. The parse, the NTFS upcasing
and the path map all reuse :mod:`aar.drp.modify.paths` (owned by S8c; not edited here) so there
is a single source of truth for Windows path rules.
"""

from __future__ import annotations

from typing import Literal

# Reuse the Windows path grammar and NTFS rules from the DRP path module. These names are its
# internal parsing primitives; the brief directs S13f to reuse them rather than re-implement NTFS
# path parsing. The module is owned by S8c and is not modified here.
from aar.drp.modify.paths import (
    _WIN_BAD,
    _WIN_RESERVED,
    PathMapper,
    _irregular,
    _parse_win,
    _Win,
)
from aar.host.errors import PathRefused

__all__ = [
    "PathForm",
    "PathMapper",
    "normalize_root",
    "remote_spec",
    "scp_target",
    "validate_windows_path",
    "windows_literal",
    "with_suffix",
]

PathForm = Literal["drive", "slash"]

# Windows verbatim / device namespaces. A path in one of these bypasses normal path rules on
# Windows, so it is refused outright before parsing.
_DEVICE_PREFIXES = ("\\\\?\\", "\\\\.\\", "//?/", "//./", "\\\\?/", "//?\\")


def _component_ok(component: str) -> str | None:
    """Return a refusal reason for a single path component, or ``None`` if it is clean."""
    if _WIN_BAD.search(component):
        return "illegal-character"
    if _WIN_RESERVED.match(component):
        return "reserved-device-name"
    if component.endswith((" ", ".")):
        return "trailing-dot-or-space"
    return None


def _unc_server_share(win: _Win) -> tuple[str, str]:
    """Split a parsed UNC root ``\\\\server\\share`` into its server and share components."""
    body = win.root[2:]  # drop the leading "\\"
    server, _, share = body.partition("\\")
    return server, share


def _check_components(win: _Win) -> str | None:
    """Validate every component of a parsed path (UNC server/share included)."""
    if win.unc:
        server, share = _unc_server_share(win)
        for component in (server, share):
            reason = _component_ok(component)
            if reason is not None:
                return reason
    for part in win.parts:
        reason = _component_ok(part)
        if reason is not None:
            return reason
    return None


def normalize_root(prefix: str) -> _Win:
    """Parse and validate a configured allowed-root prefix (drive or UNC).

    Raises :class:`PathRefused` with reason ``bad-root`` if the prefix is not a clean Windows
    drive or UNC prefix. A bare drive (``E:``) or ``\\\\server\\share`` with no further components
    is allowed as a root.
    """
    if not isinstance(prefix, str) or not prefix:
        raise PathRefused(str(prefix), "bad-root")
    if any(ord(ch) < 0x20 for ch in prefix):
        raise PathRefused(prefix, "bad-root")
    if prefix.startswith(_DEVICE_PREFIXES):
        raise PathRefused(prefix, "bad-root")
    win = _parse_win(prefix)
    if win is None or _irregular(win.parts) or _check_components(win) is not None:
        raise PathRefused(prefix, "bad-root")
    return win


def _within(path: _Win, root: _Win) -> bool:
    """Whether ``path`` is at or under ``root`` by whole NTFS-upcased components."""
    if path.unc != root.unc:
        return False
    return path.key()[: len(root.parts) + 1] == root.key()


def validate_windows_path(path: str, roots: tuple[_Win, ...]) -> _Win:
    """Validate ``path`` as a transferable edit-PC file path inside one of ``roots``.

    Returns the parsed, validated path. Raises :class:`PathRefused` otherwise. ``roots`` are
    already-parsed roots (from :func:`normalize_root`); a UNC path is accepted only if ``roots``
    holds a UNC root that contains it.
    """
    if not isinstance(path, str) or not path:
        raise PathRefused(str(path), "empty")
    if any(ord(ch) < 0x20 for ch in path):
        raise PathRefused(path, "control-character")
    if path.startswith(_DEVICE_PREFIXES):
        raise PathRefused(path, "device-namespace-path")
    win = _parse_win(path)
    if win is None:
        raise PathRefused(path, "not-a-drive-or-unc-path")
    if _irregular(win.parts):
        raise PathRefused(path, "dot-or-empty-component")
    if not win.parts:
        raise PathRefused(path, "no-file-component")
    reason = _check_components(win)
    if reason is not None:
        raise PathRefused(path, reason)
    if win.unc and not any(root.unc for root in roots):
        raise PathRefused(path, "unc-not-allowed")
    if not any(_within(win, root) for root in roots):
        raise PathRefused(path, "outside-allowed-roots")
    return win


def windows_literal(win: _Win) -> str:
    """Rebuild a validated path's canonical backslash spelling (for PowerShell -LiteralPath)."""
    if not win.parts:
        return win.root + "\\"
    return win.root + "\\" + "\\".join(win.parts)


def remote_spec(win: _Win, form: PathForm) -> str:
    """Build the scp path portion (no ``alias:``) from a validated path, forward-slashed.

    ``drive`` form yields ``E:/Show/Review/cut v2.mp4``; ``slash`` form prefixes a ``/``. The
    spec is rebuilt from the validated components, so no raw separators or alternate spellings
    from the input can leak through.
    """
    if win.unc:
        server, share = _unc_server_share(win)
        base = "//" + "/".join((server, share, *win.parts))
    else:
        base = win.root + "/" + "/".join(win.parts)  # win.root is "E:" (upper-cased)
    if form == "slash":
        return "/" + base
    return base


def scp_target(alias: str, win: _Win, form: PathForm) -> str:
    """Build the full ``alias:spec`` scp argument for a validated path."""
    return f"{alias}:{remote_spec(win, form)}"


def with_suffix(win: _Win, suffix: str) -> _Win:
    """Return a copy of ``win`` with ``suffix`` on its final component (for a temp name)."""
    parts = (*win.parts[:-1], win.parts[-1] + suffix)
    return _Win(win.root, win.unc, parts, False)
