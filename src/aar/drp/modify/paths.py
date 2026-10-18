"""Path mapping between Windows drive (or UNC) paths and Mac paths, from an explicit table.

A table entry pairs a Windows prefix (``E:``, ``E:\\Media`` or ``\\\\server\\share\\...``) with a
Mac prefix (``/Volumes/Media``, or any absolute POSIX prefix). Matching is by whole path
components, and the longest matching prefix wins:

* Windows side: separators ``\\`` and ``/`` are both accepted, and every component is compared
  case-insensitively the way NTFS does it (:func:`ntfs_upper`: a simple one-to-one uppercase
  mapping, so ``Straße`` is not ``STRASSE``), drive letters included. The rest of the path
  keeps its own spelling; drive letters are written upper-case.
* Mac side: compared exactly (case-sensitively), because Mac volumes can be case-sensitive.

UNC paths are mapped only by an entry whose Windows side is a UNC prefix; a UNC path no entry
covers is reported as ``unc-not-mapped``. A path that cannot be written on the other system
(a Mac name with ``\\`` or ``:``, a Windows-reserved character or device name such as ``CON``,
a trailing space or dot, an empty component) is reported,
never changed. Nothing here touches files; it only rewrites strings.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

__all__ = ["Direction", "MapResult", "PathMapper", "TableError", "ntfs_upper"]

Direction = Literal["windows_to_mac", "mac_to_windows"]

_DRIVE = re.compile(r"^([A-Za-z]):(?:[\\/](.*))?$", re.S)
_UNC = re.compile(r"^(?:\\\\|//)([^\\/]+)[\\/]([^\\/]+)(?:[\\/](.*))?$", re.S)
_WIN_SEP = re.compile(r"[\\/]")
_WIN_BAD = re.compile(r'[<>:"|?*\\/\x00-\x1f]')
_WIN_RESERVED = re.compile(r"^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(\..*)?$", re.I)
_MAX_TABLE = 256
_MAX_PREFIX = 4096


class TableError(ValueError):
    """The mapping table itself is unusable (a bad prefix, or two entries for one prefix)."""


@dataclass(frozen=True, slots=True)
class _Win:
    """A parsed Windows path: root (``E:`` or ``\\\\server\\share``) plus components."""

    root: str
    unc: bool
    parts: tuple[str, ...]
    trailing_sep: bool

    def key(self) -> tuple[str, ...]:
        return (ntfs_upper(self.root), *(ntfs_upper(p) for p in self.parts))


@dataclass(frozen=True, slots=True)
class _Mac:
    parts: tuple[str, ...]
    trailing_sep: bool


@dataclass(frozen=True, slots=True)
class MapResult:
    """``new`` is the mapped path, or ``None`` with ``reason`` saying why it was not mapped."""

    new: str | None
    reason: str | None = None


def ntfs_upper(name: str) -> str:
    """``name`` upper-cased one character at a time, the way NTFS compares names.

    NTFS maps each UTF-16 unit through its upcase table (a simple, one-to-one mapping), so
    ``Straße`` and ``STRASSE`` are different names, and so are ``k`` and the Kelvin sign.
    Python's ``casefold`` and ``upper`` use full mappings (``ß`` -> ``ss``/``SS``), which would
    call them equal. A character whose full uppercase is not one character keeps its spelling.
    """
    out = []
    for ch in name:
        up = ch.upper()
        out.append(up if len(up) == 1 else ch)
    return "".join(out)


def _parse_win(path: str) -> _Win | None:
    m = _UNC.match(path)
    if m is not None:
        rest = m.group(3)
        root = f"\\\\{m.group(1)}\\{m.group(2)}"
        return _Win(root, True, _split(rest, _WIN_SEP), _trailing(rest))
    m = _DRIVE.match(path)
    if m is not None:
        rest = m.group(2)
        return _Win(m.group(1).upper() + ":", False, _split(rest, _WIN_SEP), _trailing(rest))
    return None


def _trailing(rest: str | None) -> bool:
    """Whether the path after its root ends with a separator (``E:\\`` counts, ``E:`` not)."""
    return rest is not None and (rest == "" or rest[-1] in "\\/")


def _split(rest: str | None, sep: re.Pattern[str]) -> tuple[str, ...]:
    if not rest:
        return ()
    parts = sep.split(rest)
    if parts and parts[-1] == "":
        parts.pop()  # trailing separator
    return tuple(parts)


def _parse_mac(path: str) -> _Mac | None:
    if not path.startswith("/") or path.startswith("//"):
        return None
    body = path[1:]
    parts = tuple(body.split("/")) if body else ()
    trailing = path.endswith("/") and len(path) > 1
    if trailing:
        parts = parts[:-1]
    return _Mac(parts, trailing)


def _irregular(parts: tuple[str, ...]) -> bool:
    return any(p in ("", ".", "..") for p in parts)


@dataclass(frozen=True, slots=True)
class _Entry:
    win: _Win
    mac: _Mac
    index: int


class PathMapper:
    """Maps paths in one direction with a validated table (see the module docstring)."""

    def __init__(self, direction: Direction, table: list[tuple[str, str]]) -> None:
        if direction not in ("windows_to_mac", "mac_to_windows"):
            raise TableError(f"unknown direction {direction!r}")
        if not table:
            raise TableError("the mapping table is empty")
        if len(table) > _MAX_TABLE:
            raise TableError(f"the mapping table has more than {_MAX_TABLE} entries")
        self.direction = direction
        self.entries: list[_Entry] = []
        seen: dict[tuple[str, ...], int] = {}
        for i, (win, mac) in enumerate(table):
            if len(win) > _MAX_PREFIX or len(mac) > _MAX_PREFIX:
                raise TableError(f"entry {i}: prefix longer than {_MAX_PREFIX} characters")
            w = _parse_win(win)
            if w is None or _irregular(w.parts):
                raise TableError(f"entry {i}: {win!r} is not a Windows drive or UNC prefix")
            m = _parse_mac(mac)
            if m is None or not m.parts or _irregular(m.parts) or "\\" in mac:
                raise TableError(f"entry {i}: {mac!r} is not an absolute Mac path prefix")
            source = w.key() if direction == "windows_to_mac" else ("/", *m.parts)
            if source in seen:
                raise TableError(f"entries {seen[source]} and {i} map the same prefix")
            seen[source] = i
            self.entries.append(_Entry(w, m, i))

    def map(self, path: str) -> MapResult:
        if self.direction == "windows_to_mac":
            return self._to_mac(path)
        return self._to_windows(path)

    def _to_mac(self, path: str) -> MapResult:
        w = _parse_win(path)
        if w is None:
            return MapResult(None, "other-form")
        if _irregular(w.parts):
            return MapResult(None, "irregular-path")
        best: _Entry | None = None
        for e in self.entries:
            n = len(e.win.parts)
            matches = e.win.unc == w.unc and w.key()[: n + 1] == e.win.key()
            if matches and (best is None or n > len(best.win.parts)):
                best = e
        if best is None:
            return MapResult(None, "unc-not-mapped" if w.unc else "no-mapping")
        rest = w.parts[len(best.win.parts) :]
        if any("/" in p or "\x00" in p for p in rest):
            return MapResult(None, "unrepresentable")
        parts = (*best.mac.parts, *rest)
        return MapResult("/" + "/".join(parts) + ("/" if w.trailing_sep else ""))

    def _to_windows(self, path: str) -> MapResult:
        m = _parse_mac(path)
        if m is None:
            return MapResult(None, "other-form")
        if _irregular(m.parts):
            return MapResult(None, "irregular-path")
        best: _Entry | None = None
        for e in self.entries:
            n = len(e.mac.parts)
            if m.parts[:n] == e.mac.parts and (best is None or n > len(best.mac.parts)):
                best = e
        if best is None:
            return MapResult(None, "no-mapping")
        rest = m.parts[len(best.mac.parts) :]
        if any(
            _WIN_BAD.search(p) or _WIN_RESERVED.match(p) or p.endswith((" ", ".")) for p in rest
        ):
            return MapResult(None, "unrepresentable")
        parts = (*best.win.parts, *rest)
        text = best.win.root + "\\" + "\\".join(parts) if parts else best.win.root + "\\"
        if parts and m.trailing_sep:
            text += "\\"
        return MapResult(text)
