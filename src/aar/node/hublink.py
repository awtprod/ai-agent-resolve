"""What a node needs from the hub: emergency-stop status and approval-token consume.

The node and the hub may run on different hosts later (DESIGN §3: the node "pulls" hot jobs
from the hub over the tailnet). The job runner therefore depends on this narrow ``HubLink``
protocol rather than on the hub modules directly, so a remote transport can be dropped in
without touching the runner.

``LocalHubLink`` is the in-process implementation over the hub's ``EventLog``. It also bridges
the two project identities that meet at the node: the console issues an emergency stop against
a project's **registry id** (``prj_...``), while a node job carries the project's **Resolve
unique id**. ``LocalHubLink`` maps between them through the hub's project registry, so a "Stop
Show" from the console actually reaches the job (H4). The registry does not record the Resolve
uid, so the map is deliberately wide (fail closed, P18b/P18c): a job counts as stopped when ANY
registry project that may be its project is stopped -- one whose display name or an alias is the
job's project name, or one with a source whose locator names it (by ``name``, or by carrying
the uid). A rename keeps the source locator, so it still matches; two projects with the same
name both match, so stopping either pauses the job (an over-stop, never a missed stop).

It also records each agent branch a job's ``timeline.branch`` made (S11's
:func:`~aar.hub.collab.record_branch`), under the hub's ``prj_`` id for the job's project.

The link is deliberately only these operations. Lease and presence feeds to the hub are NOT
here; S11 owns those in :mod:`aar.node.feed`.

**Error contract** (the same as :class:`aar.node.feed.FeedTransport`'s, so a remote link and a
remote feed transport behave alike): every operation raises an ``OSError`` (``ConnectionError``,
``TimeoutError`` ...) when the hub cannot be reached, and a ``ValueError`` when the hub refuses
the request (for ``consume_approval``, an :class:`~aar.hub.approvals.ApprovalError` subclass;
for ``assert_running``, :class:`~aar.hub.control.Stopped`). The job runner treats "unreachable"
as fail-closed and temporary: the job pauses (``hub_unreachable``), nothing runs, no token is
marked refused, and it carries on once the hub answers again. ``LocalHubLink`` maps a SQLite
``OperationalError`` (a locked or unopenable hub database) to :class:`HubUnreachable`.

Follow-up F16 (remote transport): a node on another host needs a ``HubLink`` that speaks to
the hub over the network, with the same operations and the error contract above.
Follow-up F17: cache the registry id ↔ Resolve uid map instead of rebuilding it per check.
"""

from __future__ import annotations

import functools
import sqlite3
from collections.abc import Callable
from typing import Any, Protocol, TypeVar, runtime_checkable

from aar.hub.approvals import Approval, ApprovalScope, consume, get_approval
from aar.hub.collab import Branch, branches, record_branch
from aar.hub.common import Clock, utcnow
from aar.hub.control import ControlStatus, assert_running, status
from aar.hub.events import Actor, EventLog
from aar.hub.registry import Registry
from aar.node.locator import ProjectLocator

_T = TypeVar("_T")


class HubUnreachable(ConnectionError):
    """The hub could not be reached (an ``OSError``, per the link's error contract)."""


class UnknownHubProject(ValueError):
    """No single hub registry project is this Resolve project (a refusal, not a hub outage)."""


def _reachable(fn: Callable[..., _T]) -> Callable[..., _T]:
    """Map a SQLite outage (locked, unopenable) to :class:`HubUnreachable` (an ``OSError``)."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> _T:
        try:
            return fn(*args, **kwargs)
        except sqlite3.OperationalError as exc:
            raise HubUnreachable(f"the hub database is unavailable: {exc}") from exc

    return wrapper


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

    def approval(self, token_id: str) -> Approval | None:
        """The token's record (scope, target, who consumed it), or ``None`` if never granted."""
        ...

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
        """Record an agent branch under the hub's ``prj_`` id for ``project``; return that id.
        Idempotent: a record the hub already holds exactly so (same branch, agent, version,
        node) is not appended again.

        Raises :class:`UnknownHubProject` (a ``ValueError``) when no single registry project
        is ``project``, and the hub's ``InvalidInput`` when it refuses the record."""
        ...

    def branches(self, project: ProjectLocator) -> list[Branch]:
        """The hub's recorded agent branches for ``project``, oldest first (read-only).

        S15b's branch-ownership rebuild (F-S11-2) reads this on node start-up. Raises
        :class:`UnknownHubProject` when no single registry project is ``project``."""
        ...


class LocalHubLink:
    """In-process :class:`HubLink` over a shared hub :class:`~aar.hub.events.EventLog`."""

    def __init__(self, log: EventLog, *, approval_clock: Clock = utcnow) -> None:
        self._log = log
        self._approval_clock = approval_clock

    def registry_ids(self, project: ProjectLocator) -> list[str]:
        """Every registry project id that may be ``project`` (see the module docstring)."""
        ids = {
            found.id
            for found in Registry(self._log).list()
            if found.name == project.name
            or project.name in found.aliases
            or any(_source_names(source.locator, project) for source in found.sources)
        }
        return sorted(ids)

    @_reachable
    def assert_running(
        self, *, project: ProjectLocator | None = None, node_id: str | None = None
    ) -> None:
        if project is None:
            assert_running(self._log, node_id=node_id)
            return
        # Check both the Resolve uid the job carries and every registry id it may be under.
        assert_running(self._log, project_id=project.project_id, node_id=node_id)
        for registry_id in self.registry_ids(project):
            if registry_id != project.project_id:
                assert_running(self._log, project_id=registry_id, node_id=node_id)

    @_reachable
    def control_status(self) -> ControlStatus:
        return status(self._log)

    @_reachable
    def consume_approval(
        self, token_id: str, scope: ApprovalScope | str, target: str, actor: Actor
    ) -> Approval:
        return consume(self._log, token_id, scope, target, actor, clock=self._approval_clock)

    @_reachable
    def approval(self, token_id: str) -> Approval | None:
        return get_approval(self._log, token_id)

    def hub_project_id(self, project: ProjectLocator) -> str:
        """The ONE registry project that is ``project``, for writing a record under it.

        Unlike the (deliberately wide) stop check, a record must land on exactly one project:
        a project whose source carries the Resolve uid wins; else the single project whose
        display name or alias is the project's name. None or several: refused.
        """
        projects = Registry(self._log).list()
        by_uid = [
            p.id
            for p in projects
            if project.project_id is not None
            and any(
                isinstance(src.locator, dict) and project.project_id in src.locator.values()
                for src in p.sources
            )
        ]
        candidates = by_uid or [
            p.id for p in projects if p.name == project.name or project.name in p.aliases
        ]
        if len(candidates) != 1:
            raise UnknownHubProject(
                f"{len(candidates)} hub registry projects match Resolve project "
                f"{project.name!r} [{project.project_id}]; a branch record needs exactly one "
                f"({sorted(candidates)})"
            )
        return candidates[0]

    @_reachable
    def branches(self, project: ProjectLocator) -> list[Branch]:
        return branches(self._log, self.hub_project_id(project))

    @_reachable
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
        hub_id = self.hub_project_id(project)
        for known in branches(self._log, hub_id):
            if (
                known.branch_timeline_id == branch_timeline_id
                and known.agent_id == agent_id
                and known.version == version
                and known.recorded_by == actor
            ):
                return hub_id  # already recorded exactly so: a resend adds nothing
        record_branch(
            self._log,
            hub_id,
            actor,
            source_timeline_id=source_timeline_id,
            source_timeline_name=source_timeline_name,
            agent_id=agent_id,
            branch_timeline_id=branch_timeline_id,
            branch_timeline_name=branch_timeline_name,
            version=version,
            handoff=handoff,
        )
        return hub_id


def _source_names(locator: object, project: ProjectLocator) -> bool:
    """Whether a registry source locator names ``project`` (by name, or by its Resolve uid)."""
    if not isinstance(locator, dict):
        return False
    if locator.get("name") == project.name:
        return True
    return project.project_id is not None and project.project_id in locator.values()
