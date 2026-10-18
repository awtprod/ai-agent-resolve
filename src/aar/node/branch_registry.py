"""The node-local owner registry for agent branch timelines.

The branch-edit guard must decide access by *owner*, not by name: a human timeline that
merely looks like a branch, or a copy of a branch (which gets a new unique id), must not
be editable by an agent (DESIGN R3). So ``timeline.branch`` is the only writer here, and
it records, keyed by the branch timeline's **unique id**, who owns it. The guard allows an
edit only when the target's unique id is registered to the calling agent.

The registry is per node (per broker), held by the runner and injected into each op's
environment. It is in-memory: it mirrors the branches this node made this session, and a
restart rebuilds it from the hub's branch records.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BranchOwnership:
    """Who owns a branch timeline, and what it was branched from."""

    source_timeline_id: str
    agent_id: str
    version: int


class BranchRegistry:
    """A node-local map of branch timeline unique id to its owner."""

    def __init__(self) -> None:
        self._owners: dict[str, BranchOwnership] = {}

    def register(
        self, branch_timeline_id: str, *, source_timeline_id: str, agent_id: str, version: int
    ) -> BranchOwnership:
        """Record (or refresh) the owner of a branch. Only ``timeline.branch`` calls this."""
        if not branch_timeline_id or not source_timeline_id or not agent_id:
            raise ValueError("branch, source and agent ids must be non-empty")
        if version < 1:
            raise ValueError(f"branch version must be at least 1, got {version}")
        ownership = BranchOwnership(source_timeline_id, agent_id, version)
        self._owners[branch_timeline_id] = ownership
        return ownership

    def owner(self, branch_timeline_id: str) -> BranchOwnership | None:
        """The owner of the branch with this unique id, or ``None`` if it is not a branch."""
        return self._owners.get(branch_timeline_id)

    def owned_by(self, branch_timeline_id: str, agent_id: str) -> bool:
        """Whether ``agent_id`` owns the branch with this unique id."""
        ownership = self._owners.get(branch_timeline_id)
        return ownership is not None and ownership.agent_id == agent_id
