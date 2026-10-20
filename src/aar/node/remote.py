"""The node side of the remote hub link (S15a): ``RemoteHubLink`` and ``RemoteFeedTransport``.

These speak to ``aar hub nodeapi`` over HTTP with the stdlib only (``http.client``), so a node
on Windows needs none of the ``[hub]`` extra. They keep the exact error contract of
:class:`~aar.node.hublink.LocalHubLink` and :class:`~aar.node.feed.EventLogTransport`, so the
job runner and the feed cannot tell a remote hub from a local one:

- the hub is unreachable (a transport error, a timeout, a 5xx, 429, 503, an over-cap or a
  non-JSON 200) -> :class:`~aar.node.hublink.HubUnreachable` (an ``OSError``): fail closed,
  the job waits ``hub_unreachable`` and the feed backs off;
- a typed 4xx -> the same exception ``LocalHubLink`` raises (the ``ApprovalError`` family,
  :class:`~aar.hub.control.Stopped`, :class:`~aar.node.hublink.UnknownHubProject`,
  :class:`~aar.hub.common.InvalidInput`), and ``409 feed_conflict`` ->
  :class:`~aar.hub.leaseboard.FeedConflict`;
- a 4xx with an unknown type -> a generic :class:`HubRefused` (a ``ValueError``);
- ``401`` or ``403`` -> :class:`NodeNotEnrolled` (a ``ValueError``): the node's enrolment is
  broken and a human must fix it, so the runtime stops dispatching and stops the feed without
  retrying (the loop that reacts is S15b; here the exception is raised).

The bearer token is read from a file path, never from an env var, and never appears in a repr,
a log line or an exception message.
"""

from __future__ import annotations

import http.client
import json
import os
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from aar.hub.approvals import (
    Approval,
    ApprovalAlreadyConsumed,
    ApprovalError,
    ApprovalExpired,
    ApprovalNotFound,
    ApprovalRevoked,
    ApprovalScope,
    ApprovalScopeMismatch,
    ApprovalTargetMismatch,
)
from aar.hub.collab import Branch
from aar.hub.common import InvalidInput, actor_dict
from aar.hub.control import ControlStatus, Stopped, StopRecord, StopScope
from aar.hub.events import Actor
from aar.hub.leaseboard import FeedConflict, HubEventSpec
from aar.node.hublink import HubUnreachable, UnknownHubProject
from aar.node.locator import ProjectLocator

DEFAULT_TIMEOUT = 10.0
DEFAULT_RESPONSE_CAP = 4 << 20
"""Largest response the node will read (4 MiB); over it, the hub is treated as unreachable."""

MAX_REQUEST_ID = 64
_PREFIX = "/node/v1/"

# Status codes the hub uses (shared with aar.hub.nodeapi).
STOPPED_STATUS = 423
FEED_CONFLICT_STATUS = 409

_APPROVAL_BY_CODE: dict[str, type[ApprovalError]] = {
    cls.code: cls
    for cls in (
        ApprovalNotFound,
        ApprovalExpired,
        ApprovalRevoked,
        ApprovalAlreadyConsumed,
        ApprovalScopeMismatch,
        ApprovalTargetMismatch,
    )
}


class NodeNotEnrolled(ValueError):
    """The hub refused this node (401/403): its enrolment is missing, unknown or revoked.

    A refusal, not an outage: the node must stop claiming and dispatching and stop its feed,
    and a human must re-enrol it. The node never retries this by itself.
    """


class HubRefused(ValueError):
    """The hub refused a request with a type the node does not model. ``code`` is that type."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class RemoteConfig:
    """How to reach the hub. The token lives in ``token_file``, never here and never in env."""

    hub_url: str
    token_file: str | os.PathLike[str]
    connect_timeout: float = DEFAULT_TIMEOUT
    read_timeout: float = DEFAULT_TIMEOUT
    response_cap: int = DEFAULT_RESPONSE_CAP

    def __post_init__(self) -> None:
        parts = urlsplit(self.hub_url)
        if parts.scheme not in ("http", "https"):
            raise ValueError(f"hub_url must be http or https, got {parts.scheme!r}")
        if parts.username or parts.password:
            raise ValueError("hub_url must not carry credentials")
        if not parts.hostname:
            raise ValueError("hub_url must name a host")
        if self.connect_timeout <= 0 or self.read_timeout <= 0:
            raise ValueError("timeouts must be positive")
        if self.response_cap < 1:
            raise ValueError("response_cap must be at least 1 byte")
        object.__setattr__(self, "token_file", Path(self.token_file))


def _read_token(token_file: Path) -> str:
    """Read the bearer token from ``token_file``. Refuses a group/other-readable file on POSIX.

    The token is never read from the environment. Any error names only the path, never the
    token's bytes.
    """
    if os.name != "nt":
        # As ssh does for private keys: a token readable by group or others is refused.
        mode = token_file.stat().st_mode
        if mode & 0o077:
            raise ValueError(
                f"token file {token_file} is group- or other-readable (mode "
                f"{mode & 0o777:03o}); restrict it to the owner (chmod 600)"
            )
    # On Windows the POSIX mode bits are not meaningful, so the check is skipped (documented).
    text = token_file.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"token file {token_file} is empty")
    return text


class _NodeClient:
    """One HTTP client to the hub's node API, stdlib only. Shared by the link and the feed."""

    def __init__(self, config: RemoteConfig) -> None:
        self._config = config
        parts = urlsplit(config.hub_url)
        self._https = parts.scheme == "https"
        self._host = parts.hostname
        self._port = parts.port or (443 if self._https else 80)
        self._base_path = parts.path.rstrip("/")
        self._nonce = secrets.token_hex(8)
        self._counter = 0

    def _request_id(self) -> str:
        self._counter += 1
        return f"{self._nonce}-{self._counter}"[:MAX_REQUEST_ID]

    def call(self, op: str, body: dict[str, Any]) -> dict[str, Any]:
        """POST one op and return the ``200`` envelope, or raise the mapped exception."""
        request_id = self._request_id()
        payload = {**body, "request_id": request_id}
        status, raw = self._send(op, payload)
        return self._interpret(status, raw, body)

    def _send(self, op: str, payload: dict[str, Any]) -> tuple[int, bytes]:
        cfg = self._config
        token = _read_token(Path(cfg.token_file))
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
            "Content-Length": str(len(body)),
        }
        conn_cls = http.client.HTTPSConnection if self._https else http.client.HTTPConnection
        conn = conn_cls(self._host, self._port, timeout=cfg.connect_timeout)
        try:
            conn.connect()
            if conn.sock is not None:
                conn.sock.settimeout(cfg.read_timeout)  # reads use the read timeout
            conn.request("POST", f"{self._base_path}{_PREFIX}{op}", body=body, headers=headers)
            response = conn.getresponse()
            raw = response.read(cfg.response_cap + 1)
            if len(raw) > cfg.response_cap:
                raise HubUnreachable("the hub's response is over the 4 MiB cap")
            return response.status, raw
        except HubUnreachable:
            raise
        except (OSError, http.client.HTTPException) as exc:
            # Connection refused/reset, DNS failure, timeout, TLS failure, a dropped or garbled
            # reply: all "unreachable". The message names the host, never the token.
            raise HubUnreachable(
                f"cannot reach the hub at {self._host}:{self._port}: {type(exc).__name__}"
            ) from exc
        finally:
            conn.close()

    def _interpret(self, status: int, raw: bytes, body: dict[str, Any]) -> dict[str, Any]:
        if status >= 500 or status == 429:
            raise HubUnreachable(f"the hub returned {status}; treating it as unreachable")
        try:
            envelope = json.loads(raw)
        except (ValueError, UnicodeDecodeError) as exc:
            raise HubUnreachable(f"the hub returned {status} with a non-JSON body") from exc
        if status == 200:
            if isinstance(envelope, dict) and envelope.get("ok") is True:
                return envelope
            raise HubUnreachable("the hub returned a malformed 200 envelope")
        if status in (401, 403):
            # No detail is trusted here: any 401/403 means the enrolment must be fixed.
            raise NodeNotEnrolled("the hub refused this node (enrolment); a human must fix it")
        error = envelope.get("error") if isinstance(envelope, dict) else None
        error = error if isinstance(error, dict) else {}
        raise _map_refusal(status, error, body)


def _map_refusal(status: int, error: dict[str, Any], body: dict[str, Any]) -> Exception:
    type_ = error.get("type")
    message = error.get("message") or f"the hub refused with status {status}"
    if type_ == "feed_conflict":
        return FeedConflict(message)
    if type_ == "stopped":
        return _stopped(error, message)
    if isinstance(type_, str) and type_ in _APPROVAL_BY_CODE:
        token_id = body.get("token_id")
        return _APPROVAL_BY_CODE[type_](token_id if isinstance(token_id, str) else "", message)
    if type_ == "unknown_hub_project":
        return UnknownHubProject(message)
    if type_ == "invalid_input":
        field, _, rest = message.partition(": ")
        return InvalidInput(field or "request", rest or message)
    return HubRefused(type_ if isinstance(type_, str) else "unknown", message)


def _stopped(error: dict[str, Any], message: str) -> Stopped:
    detail = error.get("detail")
    if isinstance(detail, dict):
        try:
            record = StopRecord(
                scope=StopScope.parse(detail["scope"]),
                reason=detail["reason"],
                by=_actor(detail["by"]),
                at=detail["at"],
                event_id=detail["event_id"],
            )
            return Stopped(record)
        except (KeyError, TypeError, ValueError, InvalidInput):
            pass
    # Fall back to a synthetic record carrying the message, so the type is still Stopped.
    return Stopped(StopRecord(StopScope.all(), message, Actor("human", "unknown"), "", ""))


def _actor(data: Any) -> Actor:
    if not isinstance(data, dict):
        raise TypeError("actor must be an object")
    return Actor(data["kind"], data["id"])


def _approval_from_dict(data: dict[str, Any]) -> Approval:
    return Approval(
        token_id=data["token_id"],
        scope=ApprovalScope(data["scope"]),
        target=data["target"],
        reason=data["reason"],
        granted_by=_actor(data["granted_by"]),
        granted_at=data["granted_at"],
        expires_at=data["expires_at"],
        revoked_by=_actor(data["revoked_by"]) if data.get("revoked_by") else None,
        revoked_at=data.get("revoked_at"),
        revoke_reason=data.get("revoke_reason"),
        consumed_by=_actor(data["consumed_by"]) if data.get("consumed_by") else None,
        consumed_at=data.get("consumed_at"),
    )


def _control_status_from_dict(data: dict[str, Any]) -> ControlStatus:
    return ControlStatus(tuple(_stop_record(stop) for stop in data["stops"]))


def _stop_record(data: dict[str, Any]) -> StopRecord:
    return StopRecord(
        scope=StopScope.parse(data["scope"]),
        reason=data["reason"],
        by=_actor(data["by"]),
        at=data["at"],
        event_id=data["event_id"],
    )


def _branch_from_dict(data: dict[str, Any]) -> Branch:
    return Branch(
        project_id=data["project_id"],
        source_timeline_id=data["source"]["id"],
        source_timeline_name=data["source"]["name"],
        agent_id=data["agent_id"],
        branch_timeline_id=data["branch"]["id"],
        branch_timeline_name=data["branch"]["name"],
        version=data["version"],
        handoff=data["handoff"],
        recorded_by=_actor(data["recorded_by"]),
        ts=data["ts"],
    )


def _scope_value(scope: ApprovalScope | str) -> str:
    return scope.value if isinstance(scope, ApprovalScope) else str(scope)


class RemoteHubLink:
    """A :class:`~aar.node.hublink.HubLink` that calls ``aar hub nodeapi`` over HTTP."""

    def __init__(self, client: _NodeClient) -> None:
        self._client = client

    def assert_running(
        self, *, project: ProjectLocator | None = None, node_id: str | None = None
    ) -> None:
        # The hub always checks this node (from the token) plus 'all' and the project; the
        # node_id argument is advisory, never sent, so the body can never claim another node.
        body: dict[str, Any] = {}
        if project is not None:
            body["project"] = project.to_dict()
        self._client.call("assert_running", body)

    def control_status(self) -> ControlStatus:
        envelope = self._client.call("control_status", {})
        return _control_status_from_dict(envelope["result"])

    def consume_approval(
        self, token_id: str, scope: ApprovalScope | str, target: str, actor: Actor
    ) -> Approval:
        envelope = self._client.call(
            "consume_approval",
            {
                "token_id": token_id,
                "scope": _scope_value(scope),
                "target": target,
                "actor": actor_dict(actor),
            },
        )
        return _approval_from_dict(envelope["result"])

    def approval(self, token_id: str) -> Approval | None:
        envelope = self._client.call("approval", {"token_id": token_id})
        result = envelope["result"]
        return _approval_from_dict(result) if result is not None else None

    def record_branch(
        self,
        project: ProjectLocator,
        actor: Actor,
        *,
        source_timeline_id: str,
        source_timeline_name: str,
        agent_id: str,
        branch_timeline_id: str,
        branch_timeline_name: str,
        version: int,
        handoff: str = "",
    ) -> str:
        envelope = self._client.call(
            "record_branch",
            {
                "project": project.to_dict(),
                "actor": actor_dict(actor),
                "source_timeline_id": source_timeline_id,
                "source_timeline_name": source_timeline_name,
                "agent_id": agent_id,
                "branch_timeline_id": branch_timeline_id,
                "branch_timeline_name": branch_timeline_name,
                "version": version,
                "handoff": handoff,
            },
        )
        return envelope["result"]

    def branches(self, project: ProjectLocator) -> list[Branch]:
        envelope = self._client.call("branches", {"project": project.to_dict()})
        return [_branch_from_dict(item) for item in envelope["result"]]


class RemoteFeedTransport:
    """A :class:`~aar.node.feed.FeedTransport` that delivers feed specs to the hub over HTTP.

    Keeps the spec's idempotency key on every resend, so a lost ack never adds a second event
    (the hub dedupes on it). Maps the same error families as :class:`RemoteHubLink`:
    ``OSError`` unreachable, :class:`~aar.hub.leaseboard.FeedConflict` as itself, other refusals
    as ``ValueError`` (the feed quarantines them).
    """

    def __init__(self, client: _NodeClient) -> None:
        self._client = client

    def deliver(self, spec: HubEventSpec, *, node_id: str) -> None:
        self._client.call(
            "feed_deliver",
            {
                "spec": {
                    "type": spec.type,
                    "payload": spec.payload,
                    "idempotency_key": spec.idempotency_key,
                }
            },
        )
