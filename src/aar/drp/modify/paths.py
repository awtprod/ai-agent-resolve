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


# Characters whose Unicode *simple* uppercase is a single code point but whose *full* uppercase
# (what ``str.upper`` gives) expands to several, so the per-character rule in ``ntfs_upper`` would
# otherwise leave them unfolded. These are the Greek small letters with iota subscript
# (ypogegrammeni); NTFS folds each onto its precomposed capital (prosgegrammeni) form, e.g.
# U+1F80 -> U+1F88. Verified complete against the running Unicode database by
# test_ff_simple_uppercase_table_is_the_unicode_set.
_SIMPLE_UPPER_EXPANDING: dict[str, str] = {
    chr(cp): chr(cp + 8) for base in (0x1F80, 0x1F90, 0x1FA0) for cp in range(base, base + 8)
}
_SIMPLE_UPPER_EXPANDING.update({"ᾳ": "ᾼ", "ῃ": "ῌ", "ῳ": "ῼ"})


def ntfs_upper(name: str) -> str:
    """``name`` upper-cased one character at a time, the way NTFS compares names.

    This follows Unicode *simple* (one-to-one) case mapping, which is what NTFS's ``$UpCase``
    table does: a code point maps to a single upper-case code point or to itself, never to
    several. So ``Straße`` and ``STRASSE`` are different names (``ß`` has no simple uppercase, it
    folds to itself), ``ﬁ`` is not ``FI``, and ``k`` is not the Kelvin sign. Python's
    ``str.upper``/``casefold`` use *full* case mapping, which expands those (``ß`` -> ``SS``),
    so we take ``c.upper()`` only when it is a single code point and otherwise fall back to the
    simple mapping (:data:`_SIMPLE_UPPER_EXPANDING`, which covers the Greek iota-subscript forms
    whose full uppercase expands) or, failing that, to ``c`` unchanged. Dotless i (U+0131) still
    folds to ``I``.

    This is Unicode *simple* case mapping; it is close to, but not identical to, any one NTFS
    ``$UpCase`` table (those are frozen when a volume is formatted, and differ between XP, Vista
    and Win7-10; Win11 and exFAT tables are not established). It never *under*-folds a pair those
    tables fold, but it OVER-folds pairs NTFS keeps distinct -- aar treats them as one name, NTFS
    does not: U+0131 (dotless i) vs I, U+017F (long s) vs S, U+00B5 (micro) vs Greek mu, U+0345
    and U+1FBE vs iota, the titlecase digraphs U+01C5/01C8/01CB/01F2, U+1E9B, every astral cased
    letter (Deseret, Osage, Adlam, Warang Citi, ...), and -- on the Win7-10 table -- final sigma
    vs sigma. These are Unicode 1.1 characters NTFS never folded, so "code points added after the
    volume was formatted" does not excuse them.

    ``ntfs_upper`` is used only to match ``path_map`` table prefixes (:class:`PathMapper`); it
    does not decide output placement (the OS's own case rules do) and aar never writes to a media
    file. An over-fold is therefore not a media overwrite, but it is NOT harmless: a table prefix
    can match a path under a *different* directory that NTFS considers distinct, mapping it to the
    wrong Mac location -- a wrong relink candidate in the new output file, which the human, the
    post-write gate and the sandbox round trip must still catch.
    """
    out = []
    for ch in name:
        up = ch.upper()
        if len(up) == 1:
            out.append(up)
        else:
            out.append(_SIMPLE_UPPER_EXPANDING.get(ch, ch))
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
