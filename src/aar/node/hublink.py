"""What a node needs from the hub: emergency-stop status and approval-token consume.

The node and the hub may run on different hosts later (DESIGN §3: the node "pulls" hot jobs
from the hub over the tailnet). The job runner therefore depends on this narrow ``HubLink``
protocol rather than on the hub modules directly, so a remote transport can be dropped in
without touching the runner.

``LocalHubLink`` is the in-process implementation over the hub's ``EventLog``. It also bridges
the two project identities that meet at the node: the console issues an emergency stop against
a project's **registry id** (``prj_...``), while a node job carries the project's **Resolve
unique id**. ``LocalHubLink`` maps between them through the hub's project registry (by display
name), so a "Stop Show" from the console actually reaches the job (H4).

The link is deliberately only these operations. Lease and presence feeds to the hub are NOT
here; S11 owns those in a separate module.

Follow-up F16 (remote transport): a node on another host needs a ``HubLink`` that speaks to
the hub over the network, with the same operations, mapping errors back to
:class:`~aar.hub.control.Stopped` and the :class:`~aar.hub.approvals.ApprovalError` subclasses.
Follow-up F17: cache the registry id ↔ Resolve uid map instead of rebuilding it per check.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from aar.hub.approvals import Approval, ApprovalScope, consume, get_approval
from aar.hub.common import Clock, utcnow
from aar.hub.control import ControlStatus, assert_running, status
from aar.hub.events import Actor, EventLog
from aar.hub.registry import Registry
from aar.node.locator import ProjectLocator


@runtime_checkable
class HubLink(Protocol):
    """The hub operations the node's job runner depends on."""

    def assert_running(
        self, *, project: ProjectLocator | None = None, node_id: str | None = None
    ) -> None:
        """Raise :class:`~aar.hub.control.Stopped` if ``all``, this project (by either its
        Resolve uid or its registry id) or this node is stopped. Called before every step."""
        ...

    def control_status(self) -> ControlStatus:
        """Every active emergency stop (read-only), for reporting to agents."""
        ...

    def consume_approval(
        self, token_id: str, scope: ApprovalScope | str, target: str, actor: Actor
    ) -> Approval:
        """Use a human approval token once for ``scope`` on ``target``. Raises an
        :class:`~aar.hub.approvals.ApprovalError` subclass on refusal; refusals are logged."""
        ...

    def approval_state(self, token_id: str) -> tuple[str, Actor | None] | None:
        """Return ``(state, consumed_by)`` for a token, or ``None`` if it was never granted."""
        ...


class LocalHubLink:
    """In-process :class:`HubLink` over a shared hub :class:`~aar.hub.events.EventLog`."""

    def __init__(self, log: EventLog, *, approval_clock: Clock = utcnow) -> None:
        self._log = log
        self._approval_clock = approval_clock

    def _registry_id(self, project: ProjectLocator) -> str | None:
        found = Registry(self._log).find_by_name(project.name)
        return found.id if found is not None else None

    def assert_running(
        self, *, project: ProjectLocator | None = None, node_id: str | None = None
    ) -> None:
        if project is None:
            assert_running(self._log, node_id=node_id)
            return
        # Check both the Resolve uid the job carries and the registry id the console uses.
        assert_running(self._log, project_id=project.project_id, node_id=node_id)
        registry_id = self._registry_id(project)
        if registry_id is not None and registry_id != project.project_id:
            assert_running(self._log, project_id=registry_id, node_id=node_id)

    def control_status(self) -> ControlStatus:
        return status(self._log)

    def consume_approval(
        self, token_id: str, scope: ApprovalScope | str, target: str, actor: Actor
    ) -> Approval:
        return consume(self._log, token_id, scope, target, actor, clock=self._approval_clock)

    def approval_state(self, token_id: str) -> tuple[str, Actor | None] | None:
        approval = get_approval(self._log, token_id)
        if approval is None:
            return None
        return approval.state(self._approval_clock()), approval.consumed_by
