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
MAX_EXPR_DEPTH = 64


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
            if not math.isfinite(val):
                raise FusionError("expression_error", f"non-finite number {raw!r}")
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
        f = float(val)
        if not math.isfinite(f):
            raise FusionError("expression_error", f"non-finite numeric value {val!r}")
        return f
    raise FusionError("expression_error", f"expected number, got {val!r}")


def _check_finite(val: float, op: str) -> float:
    if not math.isfinite(val):
        raise FusionError("expression_error", f"non-finite result from {op}")
    return val


def _lua_truthy(val: Any) -> bool:
    """Lua truthiness: only ``nil`` (``None``) and ``false`` are falsy; ``0`` is truthy."""
    return val is not None and val is not False


def _iif_truthy(val: Any) -> bool:
    """Fusion ``iif()`` truthiness: treats ``0`` as false in addition to ``nil``/``false``."""
    return val is not None and val is not False and val != 0


def _safe_math(fn: Callable[..., float], name: str) -> Callable[..., float]:
    def _wrapped(*args: Any) -> float:
        nums = [_as_num(a) for a in args]
        try:
            res = float(fn(*nums))
        except (ValueError, OverflowError, ZeroDivisionError, TypeError) as exc:
            raise FusionError("expression_error", f"{name}() failed: {exc}") from exc
        return _check_finite(res, name)

    return _wrapped


def _safe_pow(base: Any, exp: Any) -> float:
    b = _as_num(base)
    e = _as_num(exp)
    if b == 0.0 and e < 0.0:
        raise FusionError("expression_error", "zero cannot be raised to a negative power")
    if b < 0.0 and not e.is_integer():
        raise FusionError("expression_error", "negative number raised to non-integer power")
    try:
        res = math.pow(b, e)
    except (ValueError, OverflowError, ZeroDivisionError) as exc:
        raise FusionError("expression_error", f"power operation failed: {exc}") from exc
    return _check_finite(res, "^")


def _builtins() -> dict[str, Any]:
    def _clamp(x: Any, lo: Any, hi: Any) -> float:
        return max(_as_num(lo), min(_as_num(hi), _as_num(x)))

    def _iif(cond: Any, a: Any, b: Any) -> Any:
        return a if _iif_truthy(cond) else b

    def _point(x: Any, y: Any) -> Point2D:
        return Point2D(_as_num(x), _as_num(y))

    def _min(*args: Any) -> float:
        if not args:
            raise FusionError("expression_error", "min() requires at least one argument")
        return min(_as_num(a) for a in args)

    def _max(*args: Any) -> float:
        if not args:
            raise FusionError("expression_error", "max() requires at least one argument")
        return max(_as_num(a) for a in args)

    funcs: dict[str, Any] = {
        "Point": _point,
        "min": _min,
        "max": _max,
        "abs": _safe_math(abs, "abs"),
        "floor": _safe_math(math.floor, "floor"),
        "ceil": _safe_math(math.ceil, "ceil"),
        "round": _safe_math(round, "round"),
        "clamp": _clamp,
        "iif": _iif,
        "sin": _safe_math(math.sin, "sin"),
        "cos": _safe_math(math.cos, "cos"),
        "tan": _safe_math(math.tan, "tan"),
        "asin": _safe_math(math.asin, "asin"),
        "acos": _safe_math(math.acos, "acos"),
        "atan": _safe_math(math.atan, "atan"),
        "atan2": _safe_math(math.atan2, "atan2"),
        "sqrt": _safe_math(math.sqrt, "sqrt"),
        "pow": _safe_pow,
        "exp": _safe_math(math.exp, "exp"),
        "log": _safe_math(math.log, "log"),
        "rad": _safe_math(math.radians, "rad"),
        "deg": _safe_math(math.degrees, "deg"),
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


# -- Lightweight expression AST (enables true short-circuiting and full syntax checking) --------


@dataclass(frozen=True)
class _LitNode:
    value: Any


@dataclass(frozen=True)
class _IdentNode:
    name: str
    followed_by_dot: bool


@dataclass(frozen=True)
class _UnaryNode:
    op: str
    operand: Any


@dataclass(frozen=True)
class _BinaryNode:
    op: str
    left: Any
    right: Any


@dataclass(frozen=True)
class _LogicNode:
    op: str  # "AND" or "OR"
    left: Any
    right: Any


@dataclass(frozen=True)
class _GetAttrNode:
    obj: Any
    attr: str


@dataclass(frozen=True)
class _GetItemNode:
    obj: Any
    index: Any


@dataclass(frozen=True)
class _CallNode:
    callee: Any
    args: tuple[Any, ...]


class _ExprParser:
    def __init__(self, tokens: list[_ExprToken]) -> None:
        self._tokens = tokens
        self._pos = 0
        self._depth = 0

    def _enter(self) -> None:
        self._depth += 1
        if self._depth > MAX_EXPR_DEPTH:
            raise FusionError(
                "expression_error",
                f"expression nesting exceeds maximum depth of {MAX_EXPR_DEPTH}",
            )

    def _leave(self) -> None:
        self._depth -= 1

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
        node = self._parse_or()
        if self._peek().kind != "EOF":
            tok = self._peek()
            raise FusionError(
                "expression_error",
                f"unexpected trailing token {tok.kind!r} ({tok.value!r}) at offset {tok.pos}",
            )
        return node

    def _parse_or(self) -> Any:
        left = self._parse_and()
        while self._peek().kind == "OR":
            self._advance()
            right = self._parse_and()
            left = _LogicNode("OR", left, right)
        return left

    def _parse_and(self) -> Any:
        left = self._parse_compare()
        while self._peek().kind == "AND":
            self._advance()
            right = self._parse_compare()
            left = _LogicNode("AND", left, right)
        return left

    def _parse_compare(self) -> Any:
        left = self._parse_add()
        while self._peek().kind in ("==", "~=", "!=", "<", "<=", ">", ">="):
            op = self._advance().kind
            right = self._parse_add()
            left = _BinaryNode(op, left, right)
        return left

    def _parse_add(self) -> Any:
        left = self._parse_mul()
        while self._peek().kind in ("+", "-"):
            op = self._advance().kind
            right = self._parse_mul()
            left = _BinaryNode(op, left, right)
        return left

    def _parse_mul(self) -> Any:
        left = self._parse_unary()
        while self._peek().kind in ("*", "/", "%"):
            op = self._advance().kind
            right = self._parse_unary()
            left = _BinaryNode(op, left, right)
        return left

    def _parse_unary(self) -> Any:
        tok = self._peek()
        if tok.kind in ("-", "+", "NOT"):
            self._advance()
            self._enter()
            try:
                return _UnaryNode(tok.kind, self._parse_unary())
            finally:
                self._leave()
        return self._parse_power()

    def _parse_power(self) -> Any:
        base = self._parse_postfix()
        if self._peek().kind == "^":
            self._advance()
            self._enter()
            try:
                # Lua ^ is right-associative and tighter than unary on the RHS: -2^2 == -(2^2)
                exp = self._parse_unary()
            finally:
                self._leave()
            return _BinaryNode("^", base, exp)
        return base

    def _parse_postfix(self) -> Any:
        node = self._parse_primary()
        while True:
            tok = self._peek()
            if tok.kind == ".":
                self._advance()
                attr = self._expect("IDENT").value
                node = _GetAttrNode(node, attr)
            elif tok.kind == "[":
                self._advance()
                self._enter()
                try:
                    idx_node = self._parse_or()
                finally:
                    self._leave()
                self._expect("]")
                node = _GetItemNode(node, idx_node)
            elif tok.kind == "(":
                self._advance()
                self._enter()
                try:
                    args: list[Any] = []
                    while self._peek().kind != ")":
                        args.append(self._parse_or())
                        if self._peek().kind == ",":
                            self._advance()
                    self._expect(")")
                finally:
                    self._leave()
                node = _CallNode(node, tuple(args))
            else:
                break
        return node

    def _parse_primary(self) -> Any:
        tok = self._peek()
        if tok.kind == "NUMBER":
            self._advance()
            return _LitNode(tok.value)
        if tok.kind == "STRING":
            self._advance()
            return _LitNode(tok.value)
        if tok.kind == "TRUE":
            self._advance()
            return _LitNode(True)
        if tok.kind == "FALSE":
            self._advance()
            return _LitNode(False)
        if tok.kind == "NIL":
            self._advance()
            return _LitNode(None)
        if tok.kind == "(":
            self._advance()
            self._enter()
            try:
                val = self._parse_or()
                self._expect(")")
                return val
            finally:
                self._leave()
        if tok.kind == "IDENT":
            name = self._advance().value
            return _IdentNode(name, followed_by_dot=self._peek().kind == ".")

        raise FusionError(
            "expression_error",
            f"unexpected token {tok.kind!r} ({tok.value!r}) at offset {tok.pos}",
        )


def _eval_getattr(obj: Any, attr: str) -> Any:
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
        return obj  # handled below via ctx

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


def _eval_getitem(obj: Any, key: Any) -> Any:
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


def _eval_node(node: Any, ctx: ExprContext, depth: int = 0) -> Any:
    if depth > MAX_EXPR_DEPTH:
        raise FusionError(
            "expression_error",
            f"expression evaluation exceeds maximum depth of {MAX_EXPR_DEPTH}",
        )
    if isinstance(node, _LitNode):
        return node.value

    if isinstance(node, _IdentNode):
        name = node.name
        if name == "time":
            return ctx.time
        if name == "comp":
            return _CompProxy(ctx)
        if name == "math":
            return _MathProxy()
        if name == "self":
            return _ToolProxy(ctx.current_tool)
        if name in _BUILTINS:
            return _BUILTINS[name]
        if node.followed_by_dot:
            return _ToolProxy(name)
        return ctx.resolve_property(ctx.current_tool, name)

    next_depth = depth + 1

    if isinstance(node, _UnaryNode):
        val = _eval_node(node.operand, ctx, next_depth)
        if node.op == "-":
            return -_as_num(val)
        if node.op == "+":
            return _as_num(val)
        if node.op == "NOT":
            return not _lua_truthy(val)
        raise FusionError("expression_error", f"unknown unary op {node.op!r}")

    if isinstance(node, _LogicNode):
        left = _eval_node(node.left, ctx, next_depth)
        if node.op == "OR":
            return left if _lua_truthy(left) else _eval_node(node.right, ctx, next_depth)
        return _eval_node(node.right, ctx, next_depth) if _lua_truthy(left) else left

    if isinstance(node, _BinaryNode):
        left = _eval_node(node.left, ctx, next_depth)
        right = _eval_node(node.right, ctx, next_depth)
        op = node.op
        if op == "+":
            return _check_finite(_as_num(left) + _as_num(right), "+")
        if op == "-":
            return _check_finite(_as_num(left) - _as_num(right), "-")
        if op == "*":
            return _check_finite(_as_num(left) * _as_num(right), "*")
        if op == "/":
            denom = _as_num(right)
            if denom == 0.0:
                raise FusionError("expression_error", "division by zero in expression")
            return _check_finite(_as_num(left) / denom, "/")
        if op == "%":
            denom = _as_num(right)
            if denom == 0.0:
                raise FusionError("expression_error", "modulo by zero in expression")
            return _check_finite(_as_num(left) % denom, "%")
        if op == "^":
            return _safe_pow(left, right)
        if op == "==":
            return left == right
        if op in ("~=", "!="):
            return left != right
        if op == "<":
            return _as_num(left) < _as_num(right)
        if op == "<=":
            return _as_num(left) <= _as_num(right)
        if op == ">":
            return _as_num(left) > _as_num(right)
        if op == ">=":
            return _as_num(left) >= _as_num(right)
        raise FusionError("expression_error", f"unknown binary op {op!r}")

    if isinstance(node, _GetAttrNode):
        obj = _eval_node(node.obj, ctx, next_depth)
        if isinstance(obj, _ToolProxy):
            if node.attr == "Output":
                return _OutputProxy(obj.name)
            return ctx.resolve_property(obj.name, node.attr)
        if isinstance(obj, _OutputProxy):
            return ctx.resolve_output_attr(obj.tool_name, node.attr)
        return _eval_getattr(obj, node.attr)

    if isinstance(node, _GetItemNode):
        obj = _eval_node(node.obj, ctx, next_depth)
        key = _eval_node(node.index, ctx, next_depth)
        return _eval_getitem(obj, key)

    if isinstance(node, _CallNode):
        callee = _eval_node(node.callee, ctx, next_depth)
        if not callable(callee):
            raise FusionError("expression_error", f"object {callee!r} is not callable")
        args = [_eval_node(a, ctx, next_depth) for a in node.args]
        try:
            return callee(*args)
        except FusionError:
            raise
        except (TypeError, ValueError, OverflowError, ZeroDivisionError) as exc:
            raise FusionError("expression_error", f"function call failed: {exc}") from exc

    raise FusionError("expression_error", f"unexpected AST node {type(node).__name__}")


def evaluate_expression(expr: str, ctx: ExprContext) -> Any:
    """Evaluate a Fusion Lua expression string in ``ctx``."""
    tokens = _tokenize_expr(expr)
    try:
        ast = _ExprParser(tokens).parse()
        return _eval_node(ast, ctx)
    except RecursionError as exc:
        raise FusionError(
            "expression_error",
            f"expression nesting exceeds maximum depth of {MAX_EXPR_DEPTH}",
        ) from exc
