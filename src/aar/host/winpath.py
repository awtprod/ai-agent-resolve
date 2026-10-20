"""Validate and normalise edit-PC (Windows) paths, and build scp remote specs.

Every path from a caller is treated as hostile. A path is accepted only if it is an absolute
drive path (or a UNC path when the configuration's roots include one), has no ``.`` / ``..`` /
empty components, no reserved device name, no illegal or control character, no trailing dot or
space, and resolves inside an allowed root. Root containment uses an ASCII-only case fold, so a
root component that differs only by *non-ASCII* case (long s vs S, dotless i vs I, ``Café`` vs
``CAFÉ``) is refused as ``outside-allowed-roots`` -- the full NTFS upcase over-folds pairs NTFS
keeps distinct, so under-folding is the safe direction (NF1). The parse, the NTFS upcasing and the
path map all reuse :mod:`aar.drp.modify.paths` (owned by S8c/S8d; not edited here) so there is a
single source of truth for Windows path rules.

Validation is lexical: it reasons about the path string, never about the remote filesystem.
A junction or symlink (a reparse point) under an allowed root can therefore still redirect a
transfer outside that root on the edit PC. The push path adds a remote reparse-point check on the
destination's immediate parent as a partial mitigation; a reparse point higher in the tree is a
known residual limit (see DESIGN "Edit-PC transport"). This is acceptable for a trusted,
single-tenant edit PC on a tailnet, and a fully safe fix would need an atomic remote
resolve-and-move the SFTP protocol does not offer (any check-then-move still races).
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
    _ascii_key,
    _irregular,
    _parse_win,
    _Win,
)
from aar.host.errors import PathRefused

__all__ = [
    "PathForm",
    "PathMapper",
    "glob_escape",
    "normalize_root",
    "remote_spec",
    "scp_remote_dest",
    "scp_remote_source",
    "validate_windows_path",
    "windows_literal",
    "with_suffix",
]

PathForm = Literal["drive", "slash"]

# scp in SFTP mode (OpenSSH 9 default) globs a remote *source* path client-side with escaping
# enabled: sftp_glob(conn, abs_src, GLOB_NOCHECK|GLOB_MARK) (scp.c:1571), QUOTE='\\' (glob.c:96),
# metacharacters * ? [ ]. A remote *destination* is NOT globbed (prepare_remote_path only,
# scp.c:1350-1354 "No need to glob here"), so a destination must stay literal.
_GLOB_META = frozenset("\\*?[]")

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


def _has_lone_surrogate(text: str) -> bool:
    """Whether ``text`` contains any UTF-16 surrogate code point (U+D800..U+DFFF)."""
    return any(0xD800 <= ord(ch) <= 0xDFFF for ch in text)


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
    if _has_lone_surrogate(prefix):
        raise PathRefused(prefix, "invalid-unicode")
    if prefix.startswith(_DEVICE_PREFIXES):
        raise PathRefused(prefix, "bad-root")
    win = _parse_win(prefix)
    if win is None or _irregular(win.parts) or _check_components(win) is not None:
        raise PathRefused(prefix, "bad-root")
    return win


def _within(path: _Win, root: _Win) -> bool:
    """Whether ``path`` is at or under ``root`` by whole components, ASCII-folded.

    Root containment uses an ASCII-only case fold (``_ascii_key``): ASCII case differences fold,
    but a non-ASCII character must match exactly. The full NTFS upcase (``_Win.key()``) over-folds
    246+ pairs NTFS keeps distinct (long s vs S, dotless i vs I, Kelvin sign vs K, ...), which would
    place e.g. ``E:\\\u017fhow`` inside root ``E:\\Show`` -- a different directory on NTFS. Under-
    folding here fails closed: a root component differing only by non-ASCII case is refused (NF1).
    """
    if path.unc != root.unc:
        return False
    return _ascii_key(path)[: len(root.parts) + 1] == _ascii_key(root)


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
    if _has_lone_surrogate(path):
        # A Linux argv with invalid UTF-8 arrives as surrogateescape code points (U+DC80..U+DCFF);
        # these cannot be UTF-16LE-encoded for -EncodedCommand. Refuse before anything is built.
        raise PathRefused(path, "invalid-unicode")
    if path.startswith(_DEVICE_PREFIXES):
        raise PathRefused(path, "device-namespace-path")
    win = _parse_win(path)
    if win is None:
        raise PathRefused(path, "not-a-drive-or-unc-path")
    if _irregular(win.parts):
        raise PathRefused(path, "dot-or-empty-component")
    if not win.parts:
        raise PathRefused(path, "no-file-component")
    if win.trailing_sep:
        # e.g. "E:\\Show\\x.mp4\\" -- a trailing separator names a directory, not a file.
        raise PathRefused(path, "trailing-separator")
    reason = _check_components(win)
    if reason is not None:
        raise PathRefused(path, reason)
    if win.unc and not any(root.unc for root in roots):
        raise PathRefused(path, "unc-not-allowed")
    within = False
    for root in roots:
        if _within(win, root):
            within = True
            if _ascii_key(win) == _ascii_key(root):
                # the path IS an allowed root (a directory), not a file under it (ASCII-folded; a
                # non-ASCII-only case difference is refused above as outside-allowed-roots)
                raise PathRefused(path, "equals-allowed-root")
    if not within:
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


def glob_escape(text: str) -> str:
    """Backslash-escape scp/SFTP glob metacharacters (``\\ * ? [ ]``) in ``text``."""
    return "".join("\\" + ch if ch in _GLOB_META else ch for ch in text)


def scp_remote_source(alias: str, win: _Win, form: PathForm) -> str:
    """The scp argument for a remote *source* (download): path portion is glob-escaped.

    scp globs a remote source client-side, so a literal ``[``/``]`` in a name must be escaped or
    it would fail to match or match a sibling. ``*``/``?`` cannot occur in a validated path.
    """
    return f"{alias}:{glob_escape(remote_spec(win, form))}"


def scp_remote_dest(alias: str, win: _Win, form: PathForm) -> str:
    """The scp argument for a remote *destination* (upload): path portion stays literal.

    scp does not glob an upload destination, so backslash escaping would reach the server as
    literal filename characters; the path must be sent as-is.
    """
    return f"{alias}:{remote_spec(win, form)}"


def with_suffix(win: _Win, suffix: str) -> _Win:
    """Return a copy of ``win`` with ``suffix`` on its final component (for a temp name)."""
    parts = (*win.parts[:-1], win.parts[-1] + suffix)
    return _Win(win.root, win.unc, parts, False)
