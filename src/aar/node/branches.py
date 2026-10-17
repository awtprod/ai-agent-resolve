"""Agent branch timelines and the review-note markers that live on them.

An agent never edits a human's timeline (DESIGN R3). It duplicates it to an agent
branch named ``<timeline> [agent:<id> v<n>]``, edits the copy, renders a review and
hands the merge back to a human (Compare Timelines, 21.1, has no scripting API). This
module holds the small, pure helpers for that convention: naming and parsing branch
timelines, recognising one, and the ``customData`` string that ties a timeline marker
to a hub review note so a mirror can update rather than duplicate it.

Nothing here touches Resolve or the hub; it is string work shared by the node op
library (:mod:`aar.node.ops.collab`), the op framework's branch guard and the hub.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ``<source> [agent:<agent id> v<n>]`` at the end of a timeline name. The agent id may
# hold anything but a closing bracket; the version is a positive integer.
_BRANCH_SUFFIX = re.compile(r" \[agent:(?P<agent>[^\]]+) v(?P<version>\d+)\]\Z")

# ``aar:note:<review id>:<note id>``: a timeline marker that mirrors a review note.
_NOTE_MARKER = re.compile(r"\Aaar:note:(?P<review>[^:]+):(?P<note>.+)\Z")
_NOTE_MARKER_PREFIX = "aar:note:"


@dataclass(frozen=True, slots=True)
class BranchName:
    """A parsed agent-branch timeline name."""

    source: str
    agent_id: str
    version: int


def branch_name(source_name: str, agent_id: str, version: int) -> str:
    """Return the branch timeline name ``<source> [agent:<id> v<n>]``."""
    if not source_name:
        raise ValueError("source_name must be non-empty")
    if not agent_id or "]" in agent_id:
        raise ValueError("agent_id must be non-empty and hold no ']'")
    if version < 1:
        raise ValueError(f"branch version must be at least 1, got {version}")
    return f"{source_name} [agent:{agent_id} v{version}]"


def parse_branch(name: str) -> BranchName | None:
    """Parse any agent-branch name, or ``None`` when it is not one."""
    match = _BRANCH_SUFFIX.search(name)
    if match is None:
        return None
    return BranchName(
        source=name[: match.start()],
        agent_id=match.group("agent"),
        version=int(match.group("version")),
    )


def is_agent_branch(name: str) -> bool:
    """Whether ``name`` is an agent branch of any source and any agent."""
    return _BRANCH_SUFFIX.search(name) is not None


def branch_version_for(name: str, source_name: str, agent_id: str) -> int | None:
    """The version if ``name`` is a branch of ``source_name`` for ``agent_id``, else ``None``.

    The source and agent are matched exactly, so a branch of another source or another
    agent (``Main [agent:other v1]`` for agent ``mine``) does not count.
    """
    parsed = parse_branch(name)
    if parsed is None or parsed.source != source_name or parsed.agent_id != agent_id:
        return None
    return parsed.version


def note_marker_custom_data(review_id: str, note_id: str) -> str:
    """The ``customData`` for a marker that mirrors note ``note_id`` of ``review_id``."""
    if ":" in review_id:
        raise ValueError("review_id must not contain ':' (it delimits the marker custom data)")
    if not review_id or not note_id:
        raise ValueError("review_id and note_id must be non-empty")
    return f"{_NOTE_MARKER_PREFIX}{review_id}:{note_id}"


def parse_note_marker(custom_data: str) -> tuple[str, str] | None:
    """Return ``(review_id, note_id)`` for a note marker's ``customData``, else ``None``.

    Returns ``None`` for a human's own marker (empty or unrelated ``customData``) and for
    a malformed ``aar:note:`` value, so a reader never crashes on bad data.
    """
    match = _NOTE_MARKER.match(custom_data)
    if match is None:
        return None
    return match.group("review"), match.group("note")


def is_note_marker_for(custom_data: str, review_id: str) -> bool:
    """Whether ``custom_data`` is a note marker belonging to ``review_id``."""
    parsed = parse_note_marker(custom_data)
    return parsed is not None and parsed[0] == review_id
