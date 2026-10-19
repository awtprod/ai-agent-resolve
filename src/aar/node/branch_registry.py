"""The node-local owner registry for agent branch timelines.

The branch-edit guard must decide access by *owner*, not by name: a human timeline that
merely looks like a branch, or a copy of a branch (which gets a new unique id), must not
be editable by an agent (DESIGN R3). So ``timeline.branch`` is the only writer here, and
it records, keyed by ``(project unique id, branch timeline unique id)``, who owns it. The
guard allows an edit only when the target is registered to the calling agent in its
project. The key carries the project id because a timeline unique id is only unique within
a project; a duplicated or re-imported project can carry the same id elsewhere.

The registry is per node (per broker), held by the runner and injected into each op's
environment. It is in-memory and starts empty: a restart does not rebuild it from the hub,
so the guard is fail-closed until ``timeline.branch`` re-establishes ownership this
session. Rebuilding a node's ownership from the hub's branch records on start-up is a
follow-up (F-S11-2).
"""

from __future__ import annotations

from dataclasses import dataclass


class OwnershipConflict(Exception):
    """Raised when a branch unique id is re-registered to a different agent."""


@dataclass(frozen=True, slots=True)
class BranchOwnership:
    """Who owns a branch timeline, and what it was branched from."""

    source_timeline_id: str
    agent_id: str
    version: int


class BranchRegistry:
    """A node-local map of ``(project id, branch timeline unique id)`` to its owner."""

    def __init__(self) -> None:
        self._owners: dict[tuple[str, str], BranchOwnership] = {}

    def register(
        self,
        project_id: str,
        branch_timeline_id: str,
        *,
        source_timeline_id: str,
        agent_id: str,
        version: int,
    ) -> BranchOwnership:
        """Record the owner of a branch. Only ``timeline.branch`` calls this.

        Re-registering the same ``(project, branch)`` to the same agent is idempotent;
        re-registering it to a *different* agent raises :class:`OwnershipConflict` so one
        agent can never silently take over another's branch.
        """
        if not project_id or not branch_timeline_id or not source_timeline_id or not agent_id:
            raise ValueError("project, branch, source and agent ids must be non-empty")
        if version < 1:
            raise ValueError(f"branch version must be at least 1, got {version}")
        key = (project_id, branch_timeline_id)
        existing = self._owners.get(key)
        if existing is not None and existing.agent_id != agent_id:
            raise OwnershipConflict(
                f"branch {branch_timeline_id!r} in project {project_id!r} is owned by "
                f"{existing.agent_id!r}, not {agent_id!r}"
            )
        ownership = BranchOwnership(source_timeline_id, agent_id, version)
        self._owners[key] = ownership
        return ownership

    def owner(self, project_id: str, branch_timeline_id: str) -> BranchOwnership | None:
        """The owner of the branch with this id in this project, or ``None``."""
        return self._owners.get((project_id, branch_timeline_id))

    def owned_by(self, project_id: str, branch_timeline_id: str, agent_id: str) -> bool:
        """Whether ``agent_id`` owns the branch with this id in this project."""
        ownership = self._owners.get((project_id, branch_timeline_id))
        return ownership is not None and ownership.agent_id == agent_id
