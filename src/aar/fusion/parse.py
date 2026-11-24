"""Parse, inspect, and serialize DaVinci Resolve Fusion ``.setting`` and ``.drfx`` templates.

Stdlib-only: no Pillow or Resolve runtime required.
"""

from __future__ import annotations

import copy
import os
import re
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aar.fusion.errors import FusionError

__all__ = [
    "ENV_TEMPLATES_DIR",
    "MAX_ASSET_BYTES",
    "MAX_PARSE_DEPTH",
    "MAX_SETTING_BYTES",
    "FusionAssetRef",
    "FusionInputSpec",
    "FusionTemplate",
    "LuaEntry",
    "LuaTable",
    "dump_setting",
    "load_template",
    "map_lines_to_inputs",
    "parse_setting",
    "read_zip_member_bounded",
    "resolve_asset_path",
    "resolve_template_path",
    "unwrap_value",
]

ENV_TEMPLATES_DIR = "AAR_FUSION_TEMPLATES_DIR"
MAX_PARSE_DEPTH = 128
MAX_SETTING_BYTES = 16 * 1024 * 1024  # 16 MiB
MAX_ASSET_BYTES = 64 * 1024 * 1024  # 64 MiB

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_LUA_KEYWORDS = frozenset(
    {
        "and",
        "break",
        "do",
        "else",
        "elseif",
        "end",
        "false",
        "for",
        "function",
        "goto",
        "if",
        "in",
        "local",
        "nil",
        "not",
        "or",
        "repeat",
        "return",
        "then",
        "true",
        "until",
        "while",
    }
)


def read_zip_member_bounded(zf: zipfile.ZipFile, member: str, max_bytes: int) -> bytes:
    """Read ``member`` from ``zf`` while capping expanded bytes to ``max_bytes``."""
    try:
        info = zf.getinfo(member)
    except KeyError as exc:
        raise FusionError("template_not_found", f"archive member not found: {member!r}") from exc
    if info.file_size > max_bytes:
        raise FusionError(
            "invalid_setting",
            f"archive member {member!r} uncompressed size ({info.file_size} bytes) "
            f"exceeds limit of {max_bytes} bytes",
        )
    chunks: list[bytes] = []
    total = 0
    with zf.open(info, "r") as stream:
        while True:
            chunk = stream.read(65536)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise FusionError(
                    "invalid_setting",
                    f"archive member {member!r} expanded beyond limit of {max_bytes} bytes",
                )
            chunks.append(chunk)
    return b"".join(chunks)


@dataclass
class LuaEntry:
    """One field or positional item inside a :class:`LuaTable`.

    ``key is None`` marks a positional element (``{ 0.5, 0.5 }``). ``bracketed=True`` preserves
    ``[0] = ...`` or ``["Key"] = ...`` syntax when serializing back to Lua.
    """

    key: str | int | float | None
    value: Any
    bracketed: bool = False


@dataclass
class LuaTable:
    """Ordered Lua table AST node preserving constructor tags such as ``MacroOperator``."""

    constructor: str | None = None
    ordered: bool = False
    entries: list[LuaEntry] = field(default_factory=list)

    def get(self, key: str | int | float, default: Any = None) -> Any:
        for entry in self.entries:
            if entry.key == key:
                return entry.value
        return default

    def __contains__(self, key: str | int | float) -> bool:
        return any(entry.key == key for entry in self.entries)

    def __getitem__(self, key: str | int | float) -> Any:
        for entry in self.entries:
            if entry.key == key:
                return entry.value
        raise KeyError(key)

    def set(self, key: str | int | float, value: Any, *, bracketed: bool | None = None) -> None:
        for entry in self.entries:
            if entry.key == key:
                entry.value = value
                if bracketed is not None:
                    entry.bracketed = bracketed
                return
        is_bracketed = (
            bracketed
            if bracketed is not None
            else (not isinstance(key, str) or not _IDENT_RE.match(key) or key in _LUA_KEYWORDS)
        )
        self.entries.append(LuaEntry(key=key, value=value, bracketed=is_bracketed))

    def remove(self, key: str | int | float) -> bool:
        before = len(self.entries)
        self.entries = [entry for entry in self.entries if entry.key != key]
        return len(self.entries) < before

    def positional(self) -> list[Any]:
        return [entry.value for entry in self.entries if entry.key is None]

    def items_dict(self) -> dict[str | int | float, Any]:
        return {entry.key: entry.value for entry in self.entries if entry.key is not None}

    def copy(self) -> LuaTable:
        return copy.deepcopy(self)


# ---------------------------------------------------------------------------
# Recursive-descent Lua table parser
# ---------------------------------------------------------------------------


class _Token:
    __slots__ = ("kind", "pos", "value")

    def __init__(self, kind: str, value: Any, pos: int) -> None:
        self.kind = kind
        self.value = value
        self.pos = pos


def _tokenize(text: str) -> list[_Token]:
    tokens: list[_Token] = []
    n = len(text)
    i = 0

    while i < n:
        ch = text[i]
        # Whitespace
        if ch in " \t\r\n\f\v":
            i += 1
            continue

        # Comments: --[[ ... ]] or -- ...
        if ch == "-" and i + 1 < n and text[i + 1] == "-":
            i += 2
            if i < n and text[i] == "[":
                eq = 0
                j = i + 1
                while j < n and text[j] == "=":
                    eq += 1
                    j += 1
                if j < n and text[j] == "[":
                    closer = "]" + ("=" * eq) + "]"
                    end = text.find(closer, j + 1)
                    if end == -1:
                        raise FusionError("invalid_setting", "unterminated Lua block comment")
                    i = end + len(closer)
                    continue
            end = text.find("\n", i)
            i = n if end == -1 else end + 1
            continue

        # Long bracket string: [[ ... ]] or [=[ ... ]=]
        if ch == "[":
            eq = 0
            j = i + 1
            while j < n and text[j] == "=":
                eq += 1
                j += 1
            if j < n and text[j] == "[":
                start = j + 1
                if start < n and text[start] == "\n":
                    start += 1
                closer = "]" + ("=" * eq) + "]"
                end = text.find(closer, start)
                if end == -1:
                    raise FusionError("invalid_setting", f"unterminated long string at offset {i}")
                tokens.append(_Token("STRING", text[start:end], i))
                i = end + len(closer)
                continue

        # Quoted strings
        if ch in ('"', "'"):
            quote = ch
            start = i
            i += 1
            buf: list[str] = []
            while i < n and text[i] != quote:
                c = text[i]
                if c == "\\":
                    i += 1
                    if i >= n:
                        raise FusionError(
                            "invalid_setting", f"unterminated escape in string at offset {start}"
                        )
                    esc = text[i]
                    if esc == "n":
                        buf.append("\n")
                    elif esc == "r":
                        buf.append("\r")
                    elif esc == "t":
                        buf.append("\t")
                    elif esc == "\\":
                        buf.append("\\")
                    elif esc == '"':
                        buf.append('"')
                    elif esc == "'":
                        buf.append("'")
                    elif esc == "\n":
                        buf.append("\n")
                    elif esc.isdigit():
                        digits = esc
                        for _ in range(2):
                            if i + 1 < n and text[i + 1].isdigit():
                                i += 1
                                digits += text[i]
                            else:
                                break
                        buf.append(chr(int(digits)))
                    else:
                        buf.append(esc)
                    i += 1
                else:
                    buf.append(c)
                    i += 1
            if i >= n or text[i] != quote:
                raise FusionError("invalid_setting", f"unterminated string at offset {start}")
            i += 1
            tokens.append(_Token("STRING", "".join(buf), start))
            continue

        # Numbers (including unary - or + when directly followed by digit or dot-digit)
        if (
            ch.isdigit()
            or (ch == "." and i + 1 < n and text[i + 1].isdigit())
            or (
                ch in "+-"
                and i + 1 < n
                and (
                    text[i + 1].isdigit()
                    or (text[i + 1] == "." and i + 2 < n and text[i + 2].isdigit())
                )
            )
        ):
            start = i
            if ch in "+-":
                i += 1
            if text[i : i + 2] in ("0x", "0X"):
                i += 2
                hex_start = i
                while i < n and text[i] in "0123456789abcdefABCDEF":
                    i += 1
                raw = text[start:i]
                if i == hex_start:
                    raise FusionError(
                        "invalid_setting", f"invalid hex literal {raw!r} at offset {start}"
                    )
                try:
                    hex_val = int(raw, 16)
                except ValueError as exc:
                    raise FusionError(
                        "invalid_setting", f"invalid hex literal {raw!r} at offset {start}"
                    ) from exc
                tokens.append(_Token("NUMBER", hex_val, start))
                continue
            is_float = False
            while i < n and text[i].isdigit():
                i += 1
            if i < n and text[i] == ".":
                is_float = True
                i += 1
                while i < n and text[i].isdigit():
                    i += 1
            if i < n and text[i] in "eE":
                is_float = True
                i += 1
                if i < n and text[i] in "+-":
                    i += 1
                while i < n and text[i].isdigit():
                    i += 1
            raw = text[start:i]
            try:
                val: int | float = float(raw) if is_float else int(raw)
            except ValueError as exc:
                raise FusionError(
                    "invalid_setting", f"invalid number literal {raw!r} at offset {start}"
                ) from exc
            tokens.append(_Token("NUMBER", val, start))
            continue

        # Identifiers and keywords
        if ch.isalpha() or ch == "_":
            start = i
            i += 1
            while i < n and (text[i].isalnum() or text[i] == "_"):
                i += 1
            word = text[start:i]
            if word == "true":
                tokens.append(_Token("BOOL", True, start))
            elif word == "false":
                tokens.append(_Token("BOOL", False, start))
            elif word == "nil":
                tokens.append(_Token("NIL", None, start))
            else:
                tokens.append(_Token("IDENT", word, start))
            continue

        # Punctuation
        if ch in "{}[]()=,;":
            tokens.append(_Token(ch, ch, i))
            i += 1
            continue

        raise FusionError("invalid_setting", f"unexpected character {ch!r} at offset {i}")

    tokens.append(_Token("EOF", None, n))
    return tokens


class _Parser:
    def __init__(self, tokens: list[_Token]) -> None:
        self._tokens = tokens
        self._pos = 0
        self._depth = 0

    def _enter(self) -> None:
        self._depth += 1
        if self._depth > MAX_PARSE_DEPTH:
            raise FusionError(
                "invalid_setting",
                f".setting table nesting exceeds maximum depth of {MAX_PARSE_DEPTH}",
            )

    def _leave(self) -> None:
        self._depth -= 1

    def _peek(self, offset: int = 0) -> _Token:
        idx = self._pos + offset
        if idx >= len(self._tokens):
            return self._tokens[-1]
        return self._tokens[idx]

    def _advance(self) -> _Token:
        tok = self._tokens[self._pos]
        self._pos += 1
        return tok

    def _expect(self, kind: str) -> _Token:
        tok = self._peek()
        if tok.kind != kind:
            raise FusionError(
                "invalid_setting",
                f"expected {kind!r} but found {tok.kind!r} ({tok.value!r}) at offset {tok.pos}",
            )
        return self._advance()

    def parse_root(self) -> LuaTable:
        # Some Resolve files start with `return { ... }`
        if self._peek().kind == "IDENT" and self._peek().value == "return":
            self._advance()
        val = self.parse_value()
        if not isinstance(val, LuaTable):
            raise FusionError(
                "invalid_setting",
                f"top-level .setting must be a Lua table, got {type(val).__name__}",
            )
        if self._peek().kind != "EOF":
            tok = self._peek()
            raise FusionError(
                "invalid_setting",
                f"unexpected trailing token {tok.kind!r} ({tok.value!r}) at offset {tok.pos}",
            )
        return val

    def parse_value(self) -> Any:
        tok = self._peek()
        if tok.kind == "STRING":
            self._advance()
            return tok.value
        if tok.kind == "NUMBER":
            self._advance()
            return tok.value
        if tok.kind == "BOOL":
            self._advance()
            return tok.value
        if tok.kind == "NIL":
            self._advance()
            return None
        if tok.kind == "{":
            return self.parse_table(constructor=None, ordered=False)
        if tok.kind == "IDENT":
            ident = self._advance().value
            # ordered() { ... }
            if ident == "ordered" and self._peek().kind == "(":
                self._advance()
                self._expect(")")
                if self._peek().kind == "{":
                    return self.parse_table(constructor=None, ordered=True)
                return LuaTable(constructor="ordered", ordered=True)
            # Constructor { ... } e.g. MacroOperator { ... }, Input { ... }, FuID { "Arial" }
            if self._peek().kind == "{":
                return self.parse_table(constructor=ident, ordered=False)
            # Constructor() { ... } or Constructor(args)
            if self._peek().kind == "(":
                self._advance()
                self._enter()
                try:
                    args: list[Any] = []
                    while self._peek().kind != ")":
                        args.append(self.parse_value())
                        if self._peek().kind in (",", ";"):
                            self._advance()
                    self._expect(")")
                finally:
                    self._leave()
                if self._peek().kind == "{":
                    return self.parse_table(constructor=ident, ordered=False)
                tbl = LuaTable(constructor=ident, ordered=False)
                for a in args:
                    tbl.entries.append(LuaEntry(key=None, value=a))
                return tbl
            return ident

        raise FusionError(
            "invalid_setting",
            f"unexpected token {tok.kind!r} ({tok.value!r}) at offset {tok.pos}",
        )

    def parse_table(self, *, constructor: str | None, ordered: bool) -> LuaTable:
        self._expect("{")
        self._enter()
        try:
            table = LuaTable(constructor=constructor, ordered=ordered)
            while self._peek().kind != "}":
                tok = self._peek()
                if tok.kind == "[":
                    self._advance()
                    key_tok = self._peek()
                    if key_tok.kind in ("STRING", "NUMBER"):
                        key = self._advance().value
                    else:
                        raise FusionError(
                            "invalid_setting",
                            f"expected string or number key inside [...] at offset {key_tok.pos}",
                        )
                    self._expect("]")
                    self._expect("=")
                    value = self.parse_value()
                    table.entries.append(LuaEntry(key=key, value=value, bracketed=True))
                elif tok.kind == "IDENT" and self._peek(1).kind == "=":
                    key = self._advance().value
                    self._advance()  # '='
                    value = self.parse_value()
                    table.entries.append(LuaEntry(key=key, value=value, bracketed=False))
                else:
                    value = self.parse_value()
                    table.entries.append(LuaEntry(key=None, value=value, bracketed=False))

                if self._peek().kind in (",", ";"):
                    self._advance()
                elif self._peek().kind != "}":
                    nxt = self._peek()
                    raise FusionError(
                        "invalid_setting",
                        f"expected ',' or '}}' in table at offset {nxt.pos}, found {nxt.kind!r}",
                    )
            self._expect("}")
            return table
        finally:
            self._leave()


def parse_setting(text: str) -> LuaTable:
    """Parse a Fusion ``.setting`` Lua document into a :class:`LuaTable` AST."""
    if not text or not text.strip():
        raise FusionError("invalid_setting", "empty .setting content")
    tokens = _tokenize(text)
    return _Parser(tokens).parse_root()


# ---------------------------------------------------------------------------
# Serializer
# ---------------------------------------------------------------------------


def _escape_lua_string(value: str) -> str:
    escaped = (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\r", "\\r")
        .replace("\n", "\\n")
        .replace("\t", "\\t")
    )
    return f'"{escaped}"'


def _format_number(value: int | float) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    # Clean float representation without trailing .0 when integral? Preserve float readability
    return repr(value)


def _is_compact_table(table: LuaTable) -> bool:
    if table.ordered or len(table.entries) > 4:
        return False
    return all(not isinstance(e.value, LuaTable) for e in table.entries)


def _dump_value(value: Any, indent: int) -> str:
    if value is None:
        return "nil"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return _format_number(value)
    if isinstance(value, str):
        return _escape_lua_string(value)
    if isinstance(value, LuaTable):
        return _dump_table(value, indent)
    raise FusionError("invalid_setting", f"cannot serialize value of type {type(value).__name__}")


def _dump_table(table: LuaTable, indent: int) -> str:
    prefix = ""
    if table.ordered:
        prefix = "ordered() "
    elif table.constructor:
        prefix = f"{table.constructor} "

    if not table.entries:
        return f"{prefix}{{}}"

    if _is_compact_table(table):
        parts: list[str] = []
        for entry in table.entries:
            val_s = _dump_value(entry.value, indent)
            if entry.key is None:
                parts.append(val_s)
            elif entry.bracketed:
                key_s = (
                    _escape_lua_string(entry.key)
                    if isinstance(entry.key, str)
                    else _format_number(entry.key)
                )
                parts.append(f"[{key_s}] = {val_s}")
            else:
                parts.append(f"{entry.key} = {val_s}")
        return f"{prefix}{{ {', '.join(parts)} }}"

    pad = "\t" * (indent + 1)
    close_pad = "\t" * indent
    lines = [f"{prefix}{{"]
    for entry in table.entries:
        val_s = _dump_value(entry.value, indent + 1)
        if entry.key is None:
            lines.append(f"{pad}{val_s},")
        elif entry.bracketed:
            key_s = (
                _escape_lua_string(entry.key)
                if isinstance(entry.key, str)
                else _format_number(entry.key)
            )
            lines.append(f"{pad}[{key_s}] = {val_s},")
        else:
            lines.append(f"{pad}{entry.key} = {val_s},")
    lines.append(f"{close_pad}}}")
    return "\n".join(lines)


def dump_setting(table: LuaTable) -> str:
    """Serialize a :class:`LuaTable` AST back into valid Fusion ``.setting`` Lua text."""
    return _dump_table(table, 0) + "\n"


# ---------------------------------------------------------------------------
# High-level template inspection & resolution
# ---------------------------------------------------------------------------


def unwrap_value(value: Any) -> Any:
    """Convert common Fusion AST wrappers (``FuID``, 2-element point tables) to Python scalars."""
    if isinstance(value, LuaTable):
        if value.constructor == "FuID":
            pos = value.positional()
            return pos[0] if pos else ""
        pos = value.positional()
        if pos and len(pos) == len(value.entries):
            return [unwrap_value(v) for v in pos]
        return {k: unwrap_value(v) for k, v in value.items_dict().items()}
    return value


def _infer_value_type(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, str):
        return "string"
    if isinstance(value, (int, float)):
        return "number"
    if (
        isinstance(value, list)
        and len(value) == 2
        and all(isinstance(x, (int, float)) for x in value)
    ):
        return "point"
    return "unknown"


@dataclass(frozen=True)
class FusionInputSpec:
    """One published input (``InstanceInput``) on a Fusion ``MacroOperator``/``GroupOperator``."""

    key: str
    name: str
    source_op: str
    source: str
    default: Any = None
    value_type: str = "unknown"
    control_group: int | None = None
    page: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "key": self.key,
            "name": self.name,
            "source_op": self.source_op,
            "source": self.source,
            "default": self.default,
            "value_type": self.value_type,
        }
        if self.control_group is not None:
            out["control_group"] = self.control_group
        if self.page is not None:
            out["page"] = self.page
        return out


@dataclass(frozen=True)
class FusionAssetRef:
    """External media asset referenced by a ``Loader`` node inside a Fusion template."""

    tool: str
    filename: str
    resolved_path: Path | None = None
    archive_member: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "filename": self.filename,
            "resolved_path": str(self.resolved_path) if self.resolved_path else None,
            "archive_member": self.archive_member,
        }


@dataclass
class FusionTemplate:
    """Parsed Fusion macro/comp template ready for inspection, customization, or render."""

    name: str
    path: Path
    operator_type: str
    ast: LuaTable
    inputs: list[FusionInputSpec]
    tools: dict[str, LuaTable]
    output_tool: str | None
    assets: list[FusionAssetRef]
    fonts: list[tuple[str, str]]
    member: str | None = None
    macro_node: LuaTable | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": str(self.path),
            "member": self.member,
            "operator_type": self.operator_type,
            "output_tool": self.output_tool,
            "tool_count": len(self.tools),
            "tools": [
                {"name": name, "type": tbl.constructor or "Unknown"}
                for name, tbl in self.tools.items()
            ],
            "inputs": [inp.to_dict() for inp in self.inputs],
            "assets": [asset.to_dict() for asset in self.assets],
            "fonts": [{"font": f, "style": s} for f, s in self.fonts],
        }


def _env_template_dirs() -> list[Path]:
    raw = os.environ.get(ENV_TEMPLATES_DIR, "").strip()
    if not raw:
        return []
    return [Path(p) for p in raw.split(os.pathsep) if p.strip()]


def _is_within(path: Path, root: Path) -> bool:
    try:
        return path.resolve().is_relative_to(root.resolve())
    except OSError:
        return False


def _has_parent_traversal(raw: str) -> bool:
    return any(part == ".." for part in re.split(r"[\\/]", raw))


def _is_windows_drive_path(raw: str) -> bool:
    return len(raw) >= 2 and raw[1] == ":" and raw[0].isalpha()


def resolve_template_path(
    ref: str | Path,
    *,
    search_dirs: Sequence[Path] = (),
    allow_direct: bool = True,
    allowed_files: Sequence[Path] | None = None,
) -> Path | None:
    """Resolve a template path or name to an existing ``.setting`` or ``.drfx`` file.

    When ``allowed_files`` is supplied, resolution is strictly restricted to that exact file list
    (no directory traversal or scanning). When ``allow_direct=False``, candidates (including
    symlink targets) must resolve inside ``search_dirs`` / ``AAR_FUSION_TEMPLATES_DIR``.
    """
    raw = str(ref).strip()
    if not raw or "\x00" in raw:
        return None
    if _has_parent_traversal(raw):
        return None

    if allowed_files is not None:
        approved: list[Path] = []
        seen_files: set[Path] = set()
        for f in allowed_files:
            rf = Path(f).expanduser().resolve()
            if rf.is_file() and rf.suffix.lower() in (".setting", ".drfx") and rf not in seen_files:
                seen_files.add(rf)
                approved.append(rf)
        raw_path = Path(raw)
        raw_lower = raw.lower()
        has_sep = "/" in raw or "\\" in raw
        for rf in approved:
            if has_sep and raw_path.is_absolute():
                try:
                    if raw_path.resolve() == rf:
                        return rf
                except OSError:
                    continue
            if rf.name.lower() == raw_lower:
                return rf
            if not has_sep and rf.stem.lower() == Path(raw).stem.lower():
                return rf
        return None

    dirs: list[Path] = []
    seen: set[Path] = set()
    for d in (*search_dirs, *_env_template_dirs()):
        resolved = d.expanduser().resolve()
        if resolved.is_dir() and resolved not in seen:
            seen.add(resolved)
            dirs.append(resolved)

    direct = Path(raw)
    if direct.is_file() and direct.suffix.lower() in (".setting", ".drfx"):
        if allow_direct or any(_is_within(direct, base) for base in dirs):
            return direct.resolve()
        return None

    if direct.is_absolute() or _is_windows_drive_path(raw):
        return None

    rel = Path(raw)
    candidates_rel = [rel]
    if rel.suffix.lower() not in (".setting", ".drfx"):
        candidates_rel.append(Path(f"{raw}.setting"))
        candidates_rel.append(Path(f"{raw}.drfx"))

    for base in dirs:
        for cand in candidates_rel:
            p = base / cand
            if p.is_file() and p.suffix.lower() in (".setting", ".drfx") and _is_within(p, base):
                return p.resolve()

    # Stem lookup inside search_dirs (common Resolve layout: Edit/Titles/<Pack>/<Name>.setting)
    target_stem = Path(raw).stem.lower()
    raw_lower = raw.lower()
    for base in dirs:
        for ext in ("*.setting", "*.drfx"):
            for match in sorted(base.rglob(ext)):
                if (
                    match.stem.lower() == target_stem or match.name.lower() == raw_lower
                ) and _is_within(match, base):
                    return match.resolve()
    return None


def resolve_asset_path(
    filename: str,
    *,
    template_path: Path,
    asset_dirs: Sequence[Path] = (),
    archive_members: Sequence[str] = (),
    confined_roots: Sequence[Path] | None = None,
) -> tuple[Path | None, str | None]:
    """Resolve a Fusion ``Loader`` ``Filename`` (including ``Templates:...`` paths).

    Returns ``(filesystem_path, archive_member)`` — at most one is non-``None``.
    Filesystem resolutions (including symlink targets) are confined to approved roots and reject
    ``..`` traversal segments.
    """
    cleaned = filename.replace("\\", "/").strip()
    if not cleaned or "\x00" in cleaned:
        return None, None

    # Strip Fusion path-map prefix like "Templates:" or "Setting:"
    has_prefix = ":" in cleaned and not _is_windows_drive_path(cleaned)
    rel_part = cleaned
    if has_prefix:
        _, rel_part = cleaned.split(":", 1)
        rel_part = rel_part.lstrip("/")

    if _has_parent_traversal(cleaned) or _has_parent_traversal(rel_part):
        return None, None

    basename = Path(rel_part).name

    # 1. Check .drfx archive members if loaded from a zip
    if archive_members:
        norm_members = {m.replace("\\", "/").lstrip("/"): m for m in archive_members}
        if rel_part in norm_members:
            return None, norm_members[rel_part]
        for norm, orig in norm_members.items():
            if norm.endswith(f"/{rel_part}"):
                return None, orig
        for norm, orig in norm_members.items():
            if basename and Path(norm).name == basename:
                return None, orig

    # 2. Compute approved roots (strictly confined; never climb arbitrary parent directories)
    approved_roots: list[Path] = []
    seen: set[Path] = set()

    def _add_root(candidate_root: Path) -> None:
        resolved = candidate_root.expanduser().resolve()
        if resolved.is_dir() and resolved not in seen:
            seen.add(resolved)
            approved_roots.append(resolved)

    if confined_roots is not None:
        bound_roots = [r.expanduser().resolve() for r in confined_roots if r.expanduser().is_dir()]
        if not bound_roots:
            return None, None
        tpl_parent = template_path.expanduser().resolve().parent
        if any(_is_within(tpl_parent, b) for b in bound_roots):
            _add_root(tpl_parent)
        for d in asset_dirs:
            rd = d.expanduser().resolve()
            if rd.is_dir() and any(_is_within(rd, b) for b in bound_roots):
                _add_root(rd)
        for b in bound_roots:
            _add_root(b)
    else:
        _add_root(template_path.parent)
        for d in asset_dirs:
            _add_root(d)
        for d in _env_template_dirs():
            _add_root(d)

    if not approved_roots:
        return None, None

    # 3. Direct filesystem path ONLY if it resolves within an approved root
    direct = Path(cleaned)
    if (direct.is_absolute() or _is_windows_drive_path(cleaned)) and direct.is_file():
        if any(_is_within(direct, root) for root in approved_roots):
            return direct.resolve(), None
        return None, None

    # 4. Relative resolution inside approved roots (rejecting symlink escapes)
    if Path(rel_part).is_absolute() or _is_windows_drive_path(rel_part):
        return None, None

    for root in approved_roots:
        candidates = [root / rel_part]
        if basename and basename not in (".", ".."):
            candidates.append(root / basename)
        for candidate in candidates:
            if candidate.is_file() and _is_within(candidate, root):
                return candidate.resolve(), None

    return None, None


def _extract_tool_input_default(tool: LuaTable, source: str) -> Any:
    inputs_tbl = tool.get("Inputs")
    if not isinstance(inputs_tbl, LuaTable):
        return None
    inp = inputs_tbl.get(source)
    if not isinstance(inp, LuaTable):
        return None
    if "Value" in inp:
        return unwrap_value(inp.get("Value"))
    return None


def _find_macro_or_tools(
    root: LuaTable, default_name: str
) -> tuple[str, str, LuaTable | None, dict[str, LuaTable], str | None]:
    top_tools = root.get("Tools")
    if not isinstance(top_tools, LuaTable):
        raise FusionError("invalid_setting", ".setting root table has no 'Tools' table")

    top_dict: dict[str, LuaTable] = {
        str(k): v for k, v in top_tools.items_dict().items() if isinstance(v, LuaTable)
    }
    if not top_dict:
        raise FusionError("invalid_setting", ".setting 'Tools' table is empty")

    # Look for MacroOperator or GroupOperator wrapping inner Tools
    active = root.get("ActiveTool")
    ordered_names = list(top_dict.keys())
    if isinstance(active, str) and active in top_dict:
        ordered_names = [active, *[n for n in ordered_names if n != active]]

    for name in ordered_names:
        node = top_dict[name]
        if node.constructor in ("MacroOperator", "GroupOperator"):
            inner_tbl = node.get("Tools")
            if isinstance(inner_tbl, LuaTable):
                inner_tools = {
                    str(k): v for k, v in inner_tbl.items_dict().items() if isinstance(v, LuaTable)
                }
                output_tool: str | None = None
                outputs_tbl = node.get("Outputs")
                if isinstance(outputs_tbl, LuaTable):
                    for out_val in outputs_tbl.items_dict().values():
                        if isinstance(out_val, LuaTable) and isinstance(
                            out_val.get("SourceOp"), str
                        ):
                            output_tool = out_val.get("SourceOp")
                            break
                if output_tool is None and inner_tools:
                    output_tool = next(reversed(inner_tools))
                return name, node.constructor, node, inner_tools, output_tool

    # Plain composition without a MacroOperator wrapper
    output_tool = (
        active if isinstance(active, str) and active in top_dict else next(reversed(top_dict))
    )
    return default_name, "Comp", None, top_dict, output_tool


def _extract_inputs(
    macro_node: LuaTable | None, tools: dict[str, LuaTable]
) -> list[FusionInputSpec]:
    specs: list[FusionInputSpec] = []
    if macro_node is not None:
        macro_inputs = macro_node.get("Inputs")
        if isinstance(macro_inputs, LuaTable):
            for key, val in macro_inputs.items_dict().items():
                if not isinstance(val, LuaTable) or val.constructor != "InstanceInput":
                    continue
                source_op = val.get("SourceOp")
                source = val.get("Source")
                if not isinstance(source_op, str) or not isinstance(source, str):
                    continue
                display_name = val.get("Name")
                name = (
                    str(display_name) if isinstance(display_name, str) and display_name else source
                )
                target_tool = tools.get(source_op)
                default = (
                    _extract_tool_input_default(target_tool, source)
                    if target_tool is not None
                    else None
                )
                if default is None and "Default" in val:
                    default = unwrap_value(val.get("Default"))
                cg = val.get("ControlGroup")
                page = val.get("Page")
                specs.append(
                    FusionInputSpec(
                        key=str(key),
                        name=name,
                        source_op=source_op,
                        source=source,
                        default=default,
                        value_type=_infer_value_type(default),
                        control_group=int(cg) if isinstance(cg, int) else None,
                        page=str(page) if isinstance(page, str) else None,
                    )
                )
            return specs

    # Plain composition: expose StyledText of any TextPlus tools as implicit inputs
    idx = 1
    for tool_name, tool in tools.items():
        if tool.constructor == "TextPlus":
            default = _extract_tool_input_default(tool, "StyledText")
            specs.append(
                FusionInputSpec(
                    key=f"Input{idx}",
                    name=tool_name,
                    source_op=tool_name,
                    source="StyledText",
                    default=default,
                    value_type="string",
                )
            )
            idx += 1
    return specs


def _extract_assets_and_fonts(
    tools: dict[str, LuaTable],
    *,
    template_path: Path,
    asset_dirs: Sequence[Path],
    archive_members: Sequence[str],
    confined_roots: Sequence[Path] | None = None,
) -> tuple[list[FusionAssetRef], list[tuple[str, str]]]:
    assets: list[FusionAssetRef] = []
    fonts: list[tuple[str, str]] = []
    seen_fonts: set[tuple[str, str]] = set()

    for tool_name, tool in tools.items():
        if tool.constructor == "Loader":
            clips = tool.get("Clips")
            if isinstance(clips, LuaTable):
                for entry in clips.entries:
                    clip = entry.value
                    if isinstance(clip, LuaTable):
                        fn = clip.get("Filename")
                        if isinstance(fn, str) and fn:
                            resolved, member = resolve_asset_path(
                                fn,
                                template_path=template_path,
                                asset_dirs=asset_dirs,
                                archive_members=archive_members,
                                confined_roots=confined_roots,
                            )
                            assets.append(
                                FusionAssetRef(
                                    tool=tool_name,
                                    filename=fn,
                                    resolved_path=resolved,
                                    archive_member=member,
                                )
                            )
        elif tool.constructor == "TextPlus":
            font = _extract_tool_input_default(tool, "Font")
            style = _extract_tool_input_default(tool, "Style")
            font_s = str(font) if isinstance(font, str) and font else "Open Sans"
            style_s = str(style) if isinstance(style, str) and style else "Regular"
            pair = (font_s, style_s)
            if pair not in seen_fonts:
                seen_fonts.add(pair)
                fonts.append(pair)

    return assets, fonts


def load_template(
    path: str | Path,
    *,
    member: str | None = None,
    asset_dirs: Sequence[Path] = (),
    confined_roots: Sequence[Path] | None = None,
) -> FusionTemplate:
    """Load and parse a Fusion ``.setting`` file or ``.drfx`` zip archive."""
    tpl_path = Path(path)
    if not tpl_path.is_file():
        raise FusionError("template_not_found", f"Fusion template not found: {tpl_path}")

    archive_members: list[str] = []
    chosen_member: str | None = None

    if tpl_path.suffix.lower() == ".drfx":
        try:
            with zipfile.ZipFile(tpl_path, "r") as zf:
                archive_members = zf.namelist()
                setting_members = [
                    m
                    for m in archive_members
                    if m.lower().endswith(".setting") and not m.startswith("__MACOSX/")
                ]
                if not setting_members:
                    raise FusionError(
                        "invalid_setting", f".drfx archive contains no .setting file: {tpl_path}"
                    )
                if member is not None:
                    matches = [
                        m
                        for m in setting_members
                        if m == member or Path(m).name == member or Path(m).stem == member
                    ]
                    if not matches:
                        raise FusionError(
                            "template_not_found",
                            f".drfx archive {tpl_path.name} has no member matching {member!r}",
                            {"available": setting_members},
                        )
                    chosen_member = matches[0]
                else:
                    chosen_member = sorted(setting_members)[0]
                raw_bytes = read_zip_member_bounded(zf, chosen_member, MAX_SETTING_BYTES)
                try:
                    text = raw_bytes.decode("utf-8-sig")
                except UnicodeDecodeError as exc:
                    raise FusionError(
                        "invalid_setting", f"cannot decode {chosen_member!r} as UTF-8: {exc}"
                    ) from exc
        except zipfile.BadZipFile as exc:
            raise FusionError(
                "invalid_setting", f"corrupt .drfx archive {tpl_path}: {exc}"
            ) from exc
    else:
        try:
            size = tpl_path.stat().st_size
            if size > MAX_SETTING_BYTES:
                raise FusionError(
                    "invalid_setting",
                    f"Fusion template {tpl_path} ({size} bytes) exceeds limit of "
                    f"{MAX_SETTING_BYTES} bytes",
                )
            text = tpl_path.read_text(encoding="utf-8-sig")
        except OSError as exc:
            raise FusionError(
                "template_not_found", f"cannot read Fusion template {tpl_path}: {exc}"
            ) from exc
        except UnicodeDecodeError as exc:
            raise FusionError(
                "invalid_setting", f"cannot decode Fusion template {tpl_path} as UTF-8: {exc}"
            ) from exc

    ast = parse_setting(text)
    default_name = Path(chosen_member).stem if chosen_member else tpl_path.stem
    name, op_type, macro_node, tools, output_tool = _find_macro_or_tools(ast, default_name)
    inputs = _extract_inputs(macro_node, tools)
    assets, fonts = _extract_assets_and_fonts(
        tools,
        template_path=tpl_path,
        asset_dirs=asset_dirs,
        archive_members=archive_members,
        confined_roots=confined_roots,
    )

    return FusionTemplate(
        name=name,
        path=tpl_path.resolve(),
        operator_type=op_type,
        ast=ast,
        inputs=inputs,
        tools=tools,
        output_tool=output_tool,
        assets=assets,
        fonts=fonts,
        member=chosen_member,
        macro_node=macro_node,
    )


def _match_input_spec(template: FusionTemplate, key: str) -> FusionInputSpec | None:
    # 1. Exact published name
    for spec in template.inputs:
        if spec.name == key:
            return spec
    # 2. Exact InputN key or Tool.Param
    for spec in template.inputs:
        if spec.key == key or f"{spec.source_op}.{spec.source}" == key:
            return spec
    # 3. Case-insensitive published name
    lower = key.lower()
    for spec in template.inputs:
        if spec.name.lower() == lower or spec.key.lower() == lower:
            return spec
    return None


def map_lines_to_inputs(
    template: FusionTemplate,
    lines: Sequence[str] = (),
    inputs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve explicit ``inputs`` and fallback positional ``lines`` into canonical input names.

    Returns a mapping ``{spec.name: value}`` validated against ``template.inputs``.
    Raises :class:`FusionError` (``unknown_input``) when an explicit input key does not match any
    published input (or ``Tool.Param``) on the template, or ``invalid_input_value`` when more
    positional ``lines`` are supplied than the template's published text inputs.
    """
    resolved: dict[str, Any] = {}

    if lines:
        string_specs = [
            s for s in template.inputs if s.value_type == "string" or s.source == "StyledText"
        ]
        if len(lines) > len(string_specs):
            if (
                len(string_specs) == 1
                and template.operator_type not in ("MacroOperator", "GroupOperator")
                and string_specs[0].source == "StyledText"
            ):
                resolved[string_specs[0].name] = "\n".join(lines)
            else:
                raise FusionError(
                    "invalid_input_value",
                    f"template {template.name!r} has {len(string_specs)} text input(s), "
                    f"but {len(lines)} line(s) were supplied",
                    {
                        "available_text_inputs": [s.name for s in string_specs],
                        "lines": len(lines),
                    },
                )
        else:
            for line, spec in zip(lines, string_specs, strict=False):
                resolved[spec.name] = line

    if inputs:
        for raw_key, val in inputs.items():
            spec = _match_input_spec(template, raw_key)
            if spec is None:
                # Allow direct Tool.Param override when the tool exists inside the macro
                if "." in raw_key:
                    tool_name, _param = raw_key.split(".", 1)
                    if tool_name in template.tools:
                        resolved[raw_key] = val
                        continue
                raise FusionError(
                    "unknown_input",
                    f"unknown input {raw_key!r} for Fusion template {template.name!r}",
                    {"available_inputs": [s.name for s in template.inputs]},
                )
            resolved[spec.name] = val

    return resolved
