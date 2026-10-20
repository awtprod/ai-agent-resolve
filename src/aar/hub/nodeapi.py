"""The hub's node-facing HTTP API (S15a): ``POST /node/v1/<op>``.

A node on another host calls this to do everything ``LocalHubLink`` and ``EventLogTransport``
do in process: assert the hub is running, read control status, read and consume approval
tokens, record agent branches, read a project's branch records, and deliver lease/presence
feed events. Each op is a thin call into the very same hub functions, so remote behaviour
matches local behaviour exactly.

Identity is an enrolled bearer token (:mod:`aar.hub.nodes`), checked on every request; the
authenticated node id comes from the token, never from the request body. The hub is still
tailnet-only (DESIGN S15 decision 3): the token is a second factor, not the only gate, so the
bind is loopback unless ``--allow-remote`` is given.

Wire contract (matched on the node side by :mod:`aar.node.remote`):

- ``200 {"ok": true, "result": ..., "replayed": bool, "request_id": ...}`` on success.
- ``4xx {"ok": false, "error": {"type": <code>, "message": <bounded>}, "request_id": ...}``
  on a refusal; a traceback never appears, and an unexpected error is a generic 500.
- An emergency stop is ``423`` with type ``stopped``; a feed conflict is ``409`` with type
  ``feed_conflict``; a missing/unknown/revoked token is ``401`` with no detail.

This module is import-guarded behind the ``[hub]`` extra (Starlette); nothing in ``aar.node``
imports it.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from aar.hub import approvals, nodes
from aar.hub.common import Clock, InvalidInput, clean_text, utcnow
from aar.hub.control import Stopped
from aar.hub.events import Actor, EventLog
from aar.hub.leaseboard import FeedConflict, HubEventSpec, deliver_feed_event
from aar.node.hublink import LocalHubLink, UnknownHubProject
from aar.node.locator import ProjectLocator

logger = logging.getLogger("aar.hub.nodeapi")

MAX_BODY_BYTES = 1 << 20
"""Request body cap: 1 MiB. Over it is 413."""

MAX_REQUEST_ID = 64
_MAX_MESSAGE = 500
_PREFIX = "/node/v1/"

STOPPED_STATUS = 423
"""An emergency stop is 423 Locked: work is deliberately held until a human lifts the stop."""

FEED_CONFLICT_STATUS = 409

# Error type codes, shared verbatim with aar.node.remote.
ERR_INVALID_REQUEST = "invalid_request"
ERR_INVALID_INPUT = "invalid_input"
ERR_NODE_MISMATCH = "node_mismatch"
ERR_ACTOR_NOT_ALLOWED = "actor_not_allowed"
ERR_UNKNOWN_HUB_PROJECT = "unknown_hub_project"
ERR_STOPPED = "stopped"
ERR_FEED_CONFLICT = "feed_conflict"
ERR_UNAUTHORIZED = "unauthorized"
ERR_NOT_FOUND = "not_found"
ERR_METHOD_NOT_ALLOWED = "method_not_allowed"
ERR_PAYLOAD_TOO_LARGE = "payload_too_large"
ERR_INTERNAL = "internal_error"


class _ApiError(Exception):
    """A refusal to turn into a 4xx error envelope."""

    def __init__(self, status: int, type_: str, message: str, *, detail: Any = None) -> None:
        super().__init__(message)
        self.status = status
        self.type = type_
        self.message = message
        self.detail = detail


@dataclass(frozen=True, slots=True)
class _Ctx:
    log: EventLog
    link: LocalHubLink
    clock: Clock
    node_id: str
    body: dict[str, Any]
    request_id: str


def _now(ctx: _Ctx) -> datetime:
    return ctx.clock()


def _required(body: dict[str, Any], field: str) -> Any:
    if field not in body:
        raise _ApiError(400, ERR_INVALID_REQUEST, f"missing required field {field!r}")
    return body[field]


def _project(body: dict[str, Any], field: str = "project") -> ProjectLocator | None:
    raw = body.get(field)
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise _ApiError(400, ERR_INVALID_REQUEST, f"{field} must be an object")
    try:
        return ProjectLocator.from_dict(raw)
    except (ValueError, TypeError) as exc:
        raise _ApiError(400, ERR_INVALID_REQUEST, str(exc)) from exc


def _body_actor(ctx: _Ctx) -> Actor:
    """The job's on-behalf-of actor from the body: an agent, or this node. Never a human."""
    raw = _required(ctx.body, "actor")
    if not isinstance(raw, dict):
        raise _ApiError(400, ERR_INVALID_REQUEST, "actor must be an object")
    try:
        actor = Actor(raw.get("kind"), raw.get("id"))  # type: ignore[arg-type]
    except (ValueError, TypeError) as exc:
        raise _ApiError(400, ERR_INVALID_REQUEST, f"invalid actor: {exc}") from exc
    if actor.kind == "human":
        raise _ApiError(403, ERR_ACTOR_NOT_ALLOWED, "a human may not act through the node API")
    if actor.kind == "node" and actor.id != ctx.node_id:
        raise _ApiError(403, ERR_ACTOR_NOT_ALLOWED, "a node actor must be the authenticated node")
    return actor


# -- ops ---------------------------------------------------------------------------------


def _op_assert_running(ctx: _Ctx) -> tuple[Any, bool]:
    # Always check this node, plus 'all' and the project (the wide check LocalHubLink does).
    ctx.link.assert_running(project=_project(ctx.body), node_id=ctx.node_id)
    return None, False


def _op_control_status(ctx: _Ctx) -> tuple[Any, bool]:
    return ctx.link.control_status().to_dict(), False


def _op_approval(ctx: _Ctx) -> tuple[Any, bool]:
    token_id = _required(ctx.body, "token_id")
    approval = ctx.link.approval(token_id)
    return (approval.to_dict(_now(ctx)) if approval is not None else None), False


def _op_consume_approval(ctx: _Ctx) -> tuple[Any, bool]:
    approval, replayed = approvals.consume_for_node(
        ctx.log,
        _required(ctx.body, "token_id"),
        _required(ctx.body, "scope"),
        _required(ctx.body, "target"),
        _body_actor(ctx),
        via_node=ctx.node_id,
        request_id=ctx.request_id,
        clock=ctx.clock,
    )
    return approval.to_dict(_now(ctx)), replayed


def _op_record_branch(ctx: _Ctx) -> tuple[Any, bool]:
    body = ctx.body
    hub_id = ctx.link.record_branch(
        _project_required(ctx),
        _body_actor(ctx),
        source_timeline_id=_required(body, "source_timeline_id"),
        source_timeline_name=_required(body, "source_timeline_name"),
        agent_id=_required(body, "agent_id"),
        branch_timeline_id=_required(body, "branch_timeline_id"),
        branch_timeline_name=_required(body, "branch_timeline_name"),
        version=_required(body, "version"),
        handoff=body.get("handoff", ""),
    )
    return hub_id, False


def _op_branches(ctx: _Ctx) -> tuple[Any, bool]:
    found = ctx.link.branches(_project_required(ctx))
    return [branch.to_dict() for branch in found], False


def _op_feed_deliver(ctx: _Ctx) -> tuple[Any, bool]:
    spec = _spec(_required(ctx.body, "spec"))
    deliver_feed_event(ctx.log, spec, ctx.node_id, clock=ctx.clock)
    return None, False


def _project_required(ctx: _Ctx) -> ProjectLocator:
    project = _project(ctx.body)
    if project is None:
        raise _ApiError(400, ERR_INVALID_REQUEST, "missing required field 'project'")
    return project


def _spec(raw: Any) -> HubEventSpec:
    if not isinstance(raw, dict):
        raise _ApiError(400, ERR_INVALID_REQUEST, "spec must be an object")
    if not isinstance(raw.get("type"), str) or not isinstance(raw.get("payload"), dict):
        raise _ApiError(400, ERR_INVALID_REQUEST, "spec needs a string type and object payload")
    key = raw.get("idempotency_key")
    if key is not None and not isinstance(key, str):
        raise _ApiError(400, ERR_INVALID_REQUEST, "spec idempotency_key must be a string or null")
    return HubEventSpec(raw["type"], raw["payload"], key)


_OPS: dict[str, Callable[[_Ctx], tuple[Any, bool]]] = {
    "assert_running": _op_assert_running,
    "control_status": _op_control_status,
    "approval": _op_approval,
    "consume_approval": _op_consume_approval,
    "record_branch": _op_record_branch,
    "branches": _op_branches,
    "feed_deliver": _op_feed_deliver,
}


# -- translation and envelopes ------------------------------------------------------------


def _translate(exc: Exception) -> _ApiError:
    """Map a hub exception to its error envelope (or a 500 for an unexpected one)."""
    if isinstance(exc, _ApiError):
        return exc
    if isinstance(exc, Stopped):
        return _ApiError(STOPPED_STATUS, ERR_STOPPED, str(exc), detail=exc.record.to_dict())
    if isinstance(exc, FeedConflict):
        return _ApiError(FEED_CONFLICT_STATUS, ERR_FEED_CONFLICT, str(exc))
    if isinstance(exc, approvals.ApprovalError):
        return _ApiError(409, exc.code, str(exc))
    if isinstance(exc, UnknownHubProject):
        return _ApiError(409, ERR_UNKNOWN_HUB_PROJECT, str(exc))
    if isinstance(exc, InvalidInput):
        return _ApiError(400, ERR_INVALID_INPUT, str(exc))
    if isinstance(exc, ValueError | KeyError | TypeError):
        return _ApiError(400, ERR_INVALID_REQUEST, str(exc))
    raise exc  # unexpected: handled as a generic 500 above


def _ok(result: Any, replayed: bool, request_id: str) -> JSONResponse:
    return JSONResponse(
        {"ok": True, "result": result, "replayed": replayed, "request_id": request_id}
    )


def _error(
    status: int, type_: str, message: str, request_id: str | None, detail: Any = None
) -> JSONResponse:
    error: dict[str, Any] = {"type": type_, "message": message[:_MAX_MESSAGE]}
    if detail is not None:
        error["detail"] = detail
    return JSONResponse({"ok": False, "error": error, "request_id": request_id}, status_code=status)


async def _read_body(request: Request) -> bytes:
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > MAX_BODY_BYTES:
            raise _ApiError(413, ERR_PAYLOAD_TOO_LARGE, "request body over the 1 MiB cap")
    return bytes(body)


def _bearer(request: Request) -> str | None:
    header = request.headers.get("authorization")
    if header is None:
        return None
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token:
        return None
    return token


def build_app(log: EventLog, *, clock: Clock = utcnow) -> Starlette:
    """A Starlette app over ``log`` (opened read-write). Run it with uvicorn."""
    link = LocalHubLink(log, approval_clock=clock)

    async def entry(request: Request) -> Response:
        request_id: str | None = None
        op = ""
        node_id = ""
        try:
            op = _route(request)  # 404/405 before we touch auth or the body
            token = _bearer(request)
            node = nodes.authenticate(log, token)
            if node is None:
                # No detail: never say whether the token was missing, unknown or revoked.
                raise _ApiError(401, ERR_UNAUTHORIZED, "unauthorized")
            node_id = node
            body = _parse_body(await _read_body(request), request)
            request_id = _request_id(body)
            _check_body_node(body, node_id)
            ctx = _Ctx(log, link, clock, node_id, body, request_id)
            result, replayed = _OPS[op](ctx)
            logger.info("node API ok op=%s node=%s request_id=%s", op, node_id, request_id)
            return _ok(result, replayed, request_id)
        except _ApiError as exc:
            logger.info(
                "node API refused op=%s node=%s request_id=%s status=%s type=%s",
                op or "?",
                node_id or "?",
                request_id,
                exc.status,
                exc.type,
            )
            return _error(exc.status, exc.type, exc.message, request_id, exc.detail)
        except Exception as exc:
            try:
                api = _translate(exc)
            except Exception:
                # Truly unexpected: a generic 500, never a traceback in the body or a token.
                logger.exception("node API internal error op=%s node=%s", op or "?", node_id or "?")
                return _error(500, ERR_INTERNAL, "internal error", request_id)
            logger.info(
                "node API refused op=%s node=%s request_id=%s status=%s type=%s",
                op or "?",
                node_id or "?",
                request_id,
                api.status,
                api.type,
            )
            return _error(api.status, api.type, api.message, request_id, api.detail)

    return Starlette(
        routes=[
            Route(
                "/{rest:path}",
                entry,
                methods=["POST", "GET", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
            )
        ]
    )


def _route(request: Request) -> str:
    """The op name, or raise 404 (bad path / unknown op) or 405 (not POST)."""
    path = request.url.path
    if not path.startswith(_PREFIX):
        raise _ApiError(404, ERR_NOT_FOUND, "no such route")
    op = path[len(_PREFIX) :]
    if not op or "/" in op:
        raise _ApiError(404, ERR_NOT_FOUND, "no such route")
    if request.method != "POST":
        raise _ApiError(405, ERR_METHOD_NOT_ALLOWED, "only POST is allowed")
    if op not in _OPS:
        raise _ApiError(404, ERR_NOT_FOUND, f"unknown op {op!r}")
    return op


def _parse_body(raw: bytes, request: Request) -> dict[str, Any]:
    content_type = request.headers.get("content-type", "")
    if content_type.split(";", 1)[0].strip().lower() != "application/json":
        raise _ApiError(400, ERR_INVALID_REQUEST, "Content-Type must be application/json")
    try:
        value = json.loads(raw)
    except (ValueError, UnicodeDecodeError) as exc:
        raise _ApiError(400, ERR_INVALID_REQUEST, f"body is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise _ApiError(400, ERR_INVALID_REQUEST, "body must be a JSON object")
    return value


def _request_id(body: dict[str, Any]) -> str:
    raw = body.get("request_id")
    try:
        return clean_text(raw, "request_id", max_len=MAX_REQUEST_ID)
    except InvalidInput as exc:
        raise _ApiError(400, ERR_INVALID_REQUEST, str(exc)) from exc


def _check_body_node(body: dict[str, Any], node_id: str) -> None:
    if "node_id" in body and body["node_id"] != node_id:
        raise _ApiError(403, ERR_NODE_MISMATCH, "body node_id does not match the token's node")
