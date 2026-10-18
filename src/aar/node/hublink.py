"""What a node needs from the hub: emergency-stop status and approval-token consume.

The node and the hub may run on different hosts later (DESIGN §3: the node "pulls" hot jobs
from the hub over the tailnet). The job runner therefore depends on this narrow ``HubLink``
protocol rather than on the hub modules directly, so a remote transport can be dropped in
without touching the runner.

``LocalHubLink`` is the in-process implementation over the hub's ``EventLog``: when the node
and hub share one process (and one SQLite log), control status and approval consume are plain
calls into ``aar.hub.control`` and ``aar.hub.approvals``.

The link is deliberately only these two reads/writes. Lease and presence feeds to the hub are
NOT here; S11 owns those in a separate module.

Follow-up F16 (remote transport): a node on another host needs a ``HubLink`` that speaks to
the hub over the network (the hub's MCP/HTTP surface), with the same two operations. Its
errors must map back to :class:`~aar.hub.control.Stopped` and the
:class:`~aar.hub.approvals.ApprovalError` subclasses so the runner behaves identically; the
runner never inspects anything hub-internal beyond those.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from aar.hub.approvals import Approval, ApprovalScope, consume
from aar.hub.common import Clock, utcnow
from aar.hub.control import ControlStatus, assert_running, status
from aar.hub.events import Actor, EventLog


@runtime_checkable
class HubLink(Protocol):
    """The hub operations the node's job runner depends on."""

    def assert_running(self, *, project_id: str | None = None, node_id: str | None = None) -> None:
        """Raise :class:`~aar.hub.control.Stopped` if ``all``, this project or this node is
        stopped. Called before every job step."""
        ...

    def control_status(self) -> ControlStatus:
        """Every active emergency stop (read-only), for reporting to agents."""
        ...

    def consume_approval(
        self, token_id: str, scope: ApprovalScope | str, target: str, actor: Actor
    ) -> Approval:
        """Use a human approval token once for ``scope`` on ``target``.

        Raises an :class:`~aar.hub.approvals.ApprovalError` subclass on any refusal; a refused
        attempt is itself recorded in the hub's event log.
        """
        ...


class LocalHubLink:
    """In-process :class:`HubLink` over a shared hub :class:`~aar.hub.events.EventLog`."""

    def __init__(self, log: EventLog, *, approval_clock: Clock = utcnow) -> None:
        self._log = log
        self._approval_clock = approval_clock

    def assert_running(self, *, project_id: str | None = None, node_id: str | None = None) -> None:
        assert_running(self._log, project_id=project_id, node_id=node_id)

    def control_status(self) -> ControlStatus:
        return status(self._log)

    def consume_approval(
        self, token_id: str, scope: ApprovalScope | str, target: str, actor: Actor
    ) -> Approval:
        return consume(self._log, token_id, scope, target, actor, clock=self._approval_clock)
