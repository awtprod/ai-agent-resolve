"""Safe recursive-descent evaluator for DaVinci Resolve Fusion Lua expressions.

Supports arithmetic (including right-associative ``^``), comparisons, boolean logic,
``Point(x, y)``, ``min``/``max``/``clamp``/``iif``/``math.*``, ``time``, ``comp.RenderStart`` /
``comp.RenderEnd``, self-property and sibling-tool lookups, and 1-based Lua indexing
(such as ``NameText.Output.DataWindow[3]``). Never calls ``eval()``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from aar.fusion.errors import FusionError

__all__ = [
    "ExprContext",
    "Point2D",
    "evaluate_expression",
    "expression_references",
]

MAX_EXPR_LEN = 8192


@dataclass(frozen=True)
class Point2D:
    """2D normalized coordinate returned by Fusion's ``Point(x, y)``."""

    x: float
    y: float

    def __getitem__(self, idx: int) -> float:
        # 1-based Lua indexing when accessed via evaluator; 0-based for Python tuple unpacking
        if idx == 0:
            return self.x
        if idx == 1:
            return self.y
        raise IndexError(idx)


@dataclass
class ExprContext:
    """Evaluation context supplied by the Fusion node graph for a single tool at frame ``time``."""

    time: float
    render_start: float
    render_end: float
    current_tool: str
    resolve_property: Callable[[str, str], Any]
    resolve_output_attr: Callable[[str, str], Any]


class _ExprToken:
    __slots__ = ("kind", "pos", "value")

    def __init__(self, kind: str, value: Any, pos: int) -> None:
        self.kind = kind
        self.value = value
        self.pos = pos


_TWO_CHAR_OPS = {"==", "~=", "!=", "<=", ">="}
_ONE_CHAR_OPS = set("+-*/%^<>()[].,")
_KEYWORDS = {"and", "or", "not", "true", "false", "nil"}


def _tokenize_expr(expr: str) -> list[_ExprToken]:
    if len(expr) > MAX_EXPR_LEN:
        raise FusionError("expression_error", f"expression exceeds {MAX_EXPR_LEN} characters")
    tokens: list[_ExprToken] = []
    n = len(expr)
    i = 0

    while i < n:
        ch = expr[i]
        if ch in " \t\r\n":
            i += 1
            continue

        if ch in ('"', "'"):
            quote = ch
            start = i
            i += 1
            buf: list[str] = []
            while i < n and expr[i] != quote:
                if expr[i] == "\\" and i + 1 < n:
                    i += 1
                    esc = expr[i]
                    buf.append({"n": "\n", "r": "\r", "t": "\t"}.get(esc, esc))
                else:
                    buf.append(expr[i])
                i += 1
            if i >= n:
                raise FusionError(
                    "expression_error", f"unterminated string in expression: {expr!r}"
                )
            i += 1
            tokens.append(_ExprToken("STRING", "".join(buf), start))
            continue

        if ch.isdigit() or (ch == "." and i + 1 < n and expr[i + 1].isdigit()):
            start = i
            while i < n and expr[i].isdigit():
                i += 1
            if i < n and expr[i] == ".":
                i += 1
                while i < n and expr[i].isdigit():
                    i += 1
            if i < n and expr[i] in "eE":
                i += 1
                if i < n and expr[i] in "+-":
                    i += 1
                while i < n and expr[i].isdigit():
                    i += 1
            raw = expr[start:i]
            try:
                val = float(raw)
            except ValueError as exc:
                raise FusionError("expression_error", f"invalid number {raw!r}") from exc
            tokens.append(_ExprToken("NUMBER", val, start))
            continue

        if ch.isalpha() or ch == "_":
            start = i
            i += 1
            while i < n and (expr[i].isalnum() or expr[i] == "_"):
                i += 1
            word = expr[start:i]
            if word in _KEYWORDS:
                tokens.append(_ExprToken(word.upper(), word, start))
            else:
                tokens.append(_ExprToken("IDENT", word, start))
            continue

        if i + 1 < n and expr[i : i + 2] in _TWO_CHAR_OPS:
            op = expr[i : i + 2]
            tokens.append(_ExprToken(op, op, i))
            i += 2
            continue

        if ch in _ONE_CHAR_OPS:
            tokens.append(_ExprToken(ch, ch, i))
            i += 1
            continue

        raise FusionError(
            "expression_error", f"unexpected character {ch!r} at offset {i} in expression {expr!r}"
        )

    tokens.append(_ExprToken("EOF", None, n))
    return tokens


def expression_references(expr: str) -> tuple[bool, set[str]]:
    """Return ``(uses_time, referenced_identifiers)`` for dependency & caching analysis."""
    tokens = _tokenize_expr(expr)
    idents: set[str] = set()
    uses_time = False
    for idx, tok in enumerate(tokens):
        if tok.kind == "IDENT":
            name = tok.value
            if name == "time":
                uses_time = True
            elif name == "comp" and idx + 2 < len(tokens) and tokens[idx + 1].kind == ".":
                attr = tokens[idx + 2].value
                if attr == "CurrentTime":
                    uses_time = True
            # Collect top-level identifiers (not immediately preceded by '.')
            if idx == 0 or tokens[idx - 1].kind != ".":
                idents.add(name)
    return uses_time, idents


def _as_num(val: Any) -> float:
    if isinstance(val, bool):
        return 1.0 if val else 0.0
    if isinstance(val, (int, float)):
        return float(val)
    raise FusionError("expression_error", f"expected number, got {val!r}")


def _truthy(val: Any) -> bool:
    return val is not None and val is not False and val != 0


def _builtins() -> dict[str, Any]:
    def _clamp(x: Any, lo: Any, hi: Any) -> float:
        return max(_as_num(lo), min(_as_num(hi), _as_num(x)))

    def _iif(cond: Any, a: Any, b: Any) -> Any:
        return a if _truthy(cond) else b

    def _point(x: Any, y: Any) -> Point2D:
        return Point2D(_as_num(x), _as_num(y))

    funcs: dict[str, Any] = {
        "Point": _point,
        "min": lambda *args: min(_as_num(a) for a in args),
        "max": lambda *args: max(_as_num(a) for a in args),
        "abs": lambda x: abs(_as_num(x)),
        "floor": lambda x: float(math.floor(_as_num(x))),
        "ceil": lambda x: float(math.ceil(_as_num(x))),
        "round": lambda x: float(round(_as_num(x))),
        "clamp": _clamp,
        "iif": _iif,
        "sin": lambda x: math.sin(_as_num(x)),
        "cos": lambda x: math.cos(_as_num(x)),
        "tan": lambda x: math.tan(_as_num(x)),
        "asin": lambda x: math.asin(_as_num(x)),
        "acos": lambda x: math.acos(_as_num(x)),
        "atan": lambda x: math.atan(_as_num(x)),
        "atan2": lambda y, x: math.atan2(_as_num(y), _as_num(x)),
        "sqrt": lambda x: math.sqrt(_as_num(x)),
        "pow": lambda a, b: math.pow(_as_num(a), _as_num(b)),
        "exp": lambda x: math.exp(_as_num(x)),
        "log": lambda x: math.log(_as_num(x)),
        "rad": lambda x: math.radians(_as_num(x)),
        "deg": lambda x: math.degrees(_as_num(x)),
        "pi": math.pi,
    }
    return funcs


_BUILTINS = _builtins()


class _ToolProxy:
    """Deferred reference to a sibling tool inside an expression (e.g. ``NameText``)."""

    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name


class _OutputProxy:
    """Deferred reference to ``Tool.Output`` inside an expression."""

    __slots__ = ("tool_name",)

    def __init__(self, tool_name: str) -> None:
        self.tool_name = tool_name


class _CompProxy:
    __slots__ = ("ctx",)

    def __init__(self, ctx: ExprContext) -> None:
        self.ctx = ctx


class _MathProxy:
    pass


class _ExprParser:
    def __init__(self, tokens: list[_ExprToken], ctx: ExprContext) -> None:
        self._tokens = tokens
        self._pos = 0
        self._ctx = ctx

    def _peek(self, offset: int = 0) -> _ExprToken:
        idx = self._pos + offset
        if idx >= len(self._tokens):
            return self._tokens[-1]
        return self._tokens[idx]

    def _advance(self) -> _ExprToken:
        tok = self._tokens[self._pos]
        self._pos += 1
        return tok

    def _expect(self, kind: str) -> _ExprToken:
        tok = self._peek()
        if tok.kind != kind:
            raise FusionError(
                "expression_error",
                f"expected {kind!r}, got {tok.kind!r} ({tok.value!r}) at offset {tok.pos}",
            )
        return self._advance()

    def parse(self) -> Any:
        val = self._parse_or()
        if self._peek().kind != "EOF":
            tok = self._peek()
            raise FusionError(
                "expression_error",
                f"unexpected trailing token {tok.kind!r} ({tok.value!r}) at offset {tok.pos}",
            )
        return val

    def _parse_or(self) -> Any:
        left = self._parse_and()
        while self._peek().kind == "OR":
            self._advance()
            right = self._parse_and()
            left = left if _truthy(left) else right
        return left

    def _parse_and(self) -> Any:
        left = self._parse_compare()
        while self._peek().kind == "AND":
            self._advance()
            right = self._parse_compare()
            left = right if _truthy(left) else left
        return left

    def _parse_compare(self) -> Any:
        left = self._parse_add()
        while self._peek().kind in ("==", "~=", "!=", "<", "<=", ">", ">="):
            op = self._advance().kind
            right = self._parse_add()
            if op == "==":
                left = left == right
            elif op in ("~=", "!="):
                left = left != right
            elif op == "<":
                left = _as_num(left) < _as_num(right)
            elif op == "<=":
                left = _as_num(left) <= _as_num(right)
            elif op == ">":
                left = _as_num(left) > _as_num(right)
            elif op == ">=":
                left = _as_num(left) >= _as_num(right)
        return left

    def _parse_add(self) -> Any:
        left = self._parse_mul()
        while self._peek().kind in ("+", "-"):
            op = self._advance().kind
            right = self._parse_mul()
            left = _as_num(left) + _as_num(right) if op == "+" else _as_num(left) - _as_num(right)
        return left

    def _parse_mul(self) -> Any:
        left = self._parse_unary()
        while self._peek().kind in ("*", "/", "%"):
            op = self._advance().kind
            right = self._parse_unary()
            if op == "*":
                left = _as_num(left) * _as_num(right)
            elif op == "/":
                denom = _as_num(right)
                if denom == 0.0:
                    raise FusionError("expression_error", "division by zero in expression")
                left = _as_num(left) / denom
            else:
                denom = _as_num(right)
                if denom == 0.0:
                    raise FusionError("expression_error", "modulo by zero in expression")
                left = _as_num(left) % denom
        return left

    def _parse_unary(self) -> Any:
        tok = self._peek()
        if tok.kind == "-":
            self._advance()
            return -_as_num(self._parse_unary())
        if tok.kind == "+":
            self._advance()
            return _as_num(self._parse_unary())
        if tok.kind == "NOT":
            self._advance()
            return not _truthy(self._parse_unary())
        return self._parse_power()

    def _parse_power(self) -> Any:
        base = self._parse_postfix()
        if self._peek().kind == "^":
            self._advance()
            # Lua ^ is right-associative and tighter than unary on the RHS: -2^2 == -(2^2)
            exp = self._parse_unary()
            return _as_num(base) ** _as_num(exp)
        return base

    def _parse_postfix(self) -> Any:
        val = self._parse_primary()
        while True:
            tok = self._peek()
            if tok.kind == ".":
                self._advance()
                attr = self._expect("IDENT").value
                val = self._getattr(val, attr)
            elif tok.kind == "[":
                self._advance()
                idx_val = self._parse_or()
                self._expect("]")
                val = self._getitem(val, idx_val)
            elif tok.kind == "(":
                self._advance()
                args: list[Any] = []
                while self._peek().kind != ")":
                    args.append(self._parse_or())
                    if self._peek().kind == ",":
                        self._advance()
                self._expect(")")
                if not callable(val):
                    raise FusionError("expression_error", f"object {val!r} is not callable")
                val = val(*args)
            else:
                break
        return val

    def _parse_primary(self) -> Any:
        tok = self._peek()
        if tok.kind == "NUMBER":
            self._advance()
            return tok.value
        if tok.kind == "STRING":
            self._advance()
            return tok.value
        if tok.kind == "TRUE":
            self._advance()
            return True
        if tok.kind == "FALSE":
            self._advance()
            return False
        if tok.kind == "NIL":
            self._advance()
            return None
        if tok.kind == "(":
            self._advance()
            val = self._parse_or()
            self._expect(")")
            return val
        if tok.kind == "IDENT":
            name = self._advance().value
            if name == "time":
                return self._ctx.time
            if name == "comp":
                return _CompProxy(self._ctx)
            if name == "math":
                return _MathProxy()
            if name == "self":
                return _ToolProxy(self._ctx.current_tool)
            if name in _BUILTINS:
                return _BUILTINS[name]
            # If followed by '.', treat as sibling tool reference (e.g., NameBarMask.Width)
            if self._peek().kind == ".":
                return _ToolProxy(name)
            # Otherwise resolve as a property on the current tool (e.g., Width, Height)
            return self._ctx.resolve_property(self._ctx.current_tool, name)

        raise FusionError(
            "expression_error",
            f"unexpected token {tok.kind!r} ({tok.value!r}) at offset {tok.pos}",
        )

    def _getattr(self, obj: Any, attr: str) -> Any:
        if isinstance(obj, _CompProxy):
            if attr == "RenderStart":
                return obj.ctx.render_start
            if attr == "RenderEnd":
                return obj.ctx.render_end
            if attr == "CurrentTime":
                return obj.ctx.time
            raise FusionError("expression_error", f"unsupported comp attribute {attr!r}")

        if isinstance(obj, _MathProxy):
            if attr in _BUILTINS:
                return _BUILTINS[attr]
            raise FusionError("expression_error", f"unsupported math attribute {attr!r}")

        if isinstance(obj, _ToolProxy):
            if attr == "Output":
                return _OutputProxy(obj.name)
            return self._ctx.resolve_property(obj.name, attr)

        if isinstance(obj, _OutputProxy):
            return self._ctx.resolve_output_attr(obj.tool_name, attr)

        if isinstance(obj, Point2D):
            if attr in ("X", "x"):
                return obj.x
            if attr in ("Y", "y"):
                return obj.y
            raise FusionError("expression_error", f"Point has no attribute {attr!r}")

        if isinstance(obj, dict) and attr in obj:
            return obj[attr]

        raise FusionError(
            "expression_error", f"cannot access attribute {attr!r} on {type(obj).__name__}"
        )

    def _getitem(self, obj: Any, key: Any) -> Any:
        if isinstance(obj, Point2D):
            idx = int(_as_num(key))
            if idx == 1:
                return obj.x
            if idx == 2:
                return obj.y
            raise FusionError("expression_error", f"Point index {idx} out of range (1..2)")

        if isinstance(obj, (list, tuple)):
            idx = int(_as_num(key))
            # Fusion Lua arrays are 1-based!
            if 1 <= idx <= len(obj):
                return obj[idx - 1]
            raise FusionError(
                "expression_error",
                f"1-based index {idx} out of range for sequence of length {len(obj)}",
            )

        if isinstance(obj, dict):
            if key in obj:
                return obj[key]
            raise FusionError("expression_error", f"key {key!r} not found")

        raise FusionError("expression_error", f"cannot index {type(obj).__name__} with {key!r}")


def evaluate_expression(expr: str, ctx: ExprContext) -> Any:
    """Evaluate a Fusion Lua expression string in ``ctx``."""
    tokens = _tokenize_expr(expr)
    return _ExprParser(tokens, ctx).parse()
