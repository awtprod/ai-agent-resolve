"""Agent branch timelines and the review-note markers that live on them.

An agent never edits a human's timeline (DESIGN R3). It duplicates it to an agent
branch named ``<timeline> [agent:<id> v<n>]``, edits the copy, renders a review and
hands the merge back to a human (Compare Timelines, 21.1, has no scripting API). This
module holds the small, pure helpers for that convention: naming and parsing branch
timelines, and the ``customData`` string that ties a timeline marker to a hub review
note so a mirror can update rather than duplicate it.

Names are for display and version numbering only; access is decided by the owner
registry (:mod:`aar.node.branch_registry`), keyed by timeline unique id. The grammar
here is strict and anchored so a human timeline that merely looks like a branch is
never mistaken for one by a reader: a restricted agent-id charset, ASCII digits only,
versions from 1 with no leading zero, a non-empty source, and the whole string matched.

Nothing here touches Resolve or the hub; it is string work.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Agent id and the id parts of a note marker: ASCII letters, digits, dot, underscore,
# dash. No ':' (it delimits), no ']' (it closes the branch suffix), no control or newline.
_ID = r"[A-Za-z0-9._-]+"
_VERSION = r"[1-9][0-9]*"  # 1 and up, ASCII digits, no leading zero

# ``<source> [agent:<agent id> v<n>]`` at the very end of a timeline name.
_BRANCH_SUFFIX = re.compile(rf" \[agent:({_ID}) v({_VERSION})\]\Z")
_AGENT_ID = re.compile(rf"\A{_ID}\Z")

# ``aar:note:<review id>:<note id>``: a timeline marker mirroring a review note.
_NOTE_MARKER_PREFIX = "aar:note:"
_NOTE_MARKER = re.compile(rf"\Aaar:note:({_ID}):({_ID})\Z")
_NOTE_ID = re.compile(rf"\A{_ID}\Z")


@dataclass(frozen=True, slots=True)
class BranchName:
    """A parsed agent-branch timeline name."""

    source: str
    agent_id: str
    version: int


MAX_ID_LEN = 128  # matches the hub's node/id bound


def valid_agent_id(agent_id: str) -> bool:
    """Whether ``agent_id`` is usable in a branch name and the owner registry."""
    return (
        isinstance(agent_id, str)
        and len(agent_id) <= MAX_ID_LEN
        and _AGENT_ID.fullmatch(agent_id) is not None
    )


def valid_note_id(value: str) -> bool:
    """Whether ``value`` is a usable review id or note id for a marker's custom data."""
    return (
        isinstance(value, str)
        and len(value) <= MAX_ID_LEN
        and _NOTE_ID.fullmatch(value) is not None
    )


def branch_name(source_name: str, agent_id: str, version: int) -> str:
    """Return the branch timeline name ``<source> [agent:<id> v<n>]``."""
    if not source_name:
        raise ValueError("source_name must be non-empty")
    if not valid_agent_id(agent_id):
        raise ValueError(f"agent_id must match {_ID!r}, got {agent_id!r}")
    if version < 1:
        raise ValueError(f"branch version must be at least 1, got {version}")
    return f"{source_name} [agent:{agent_id} v{version}]"


def parse_branch(name: str) -> BranchName | None:
    """Parse an agent-branch name, or ``None`` when it is not one (strict, anchored)."""
    if not isinstance(name, str):
        return None
    match = _BRANCH_SUFFIX.search(name)
    if match is None:
        return None
    source = name[: match.start()]
    if not source:  # a branch needs a non-empty source timeline name
        return None
    if not valid_agent_id(match.group(1)):  # only what branch_name could have made
        return None
    return BranchName(source=source, agent_id=match.group(1), version=int(match.group(2)))


def is_agent_branch(name: str) -> bool:
    """Whether ``name`` is a well-formed agent branch of any source and any agent."""
    return parse_branch(name) is not None


def branch_version_for(name: str, source_name: str, agent_id: str) -> int | None:
    """The version if ``name`` is a branch of ``source_name`` for ``agent_id``, else ``None``."""
    parsed = parse_branch(name)
    if parsed is None or parsed.source != source_name or parsed.agent_id != agent_id:
        return None
    return parsed.version


def note_marker_custom_data(review_id: str, note_id: str) -> str:
    """The ``customData`` for a marker that mirrors note ``note_id`` of ``review_id``."""
    if not valid_note_id(review_id) or not valid_note_id(note_id):
        raise ValueError(f"review_id and note_id must match {_ID!r}: {review_id!r}, {note_id!r}")
    return f"{_NOTE_MARKER_PREFIX}{review_id}:{note_id}"


def parse_note_marker(custom_data: str) -> tuple[str, str] | None:
    """Return ``(review_id, note_id)`` for a note marker's ``customData``, else ``None``.

    ``None`` for a human's own marker (empty or unrelated ``customData``) and for a
    malformed ``aar:note:`` value, so a reader never crashes on bad data.
    """
    if not isinstance(custom_data, str):
        return None
    match = _NOTE_MARKER.match(custom_data)
    if match is None:
        return None
    return match.group(1), match.group(2)


def is_note_marker_for(custom_data: str, review_id: str) -> bool:
    """Whether ``custom_data`` is a note marker belonging to ``review_id``."""
    parsed = parse_note_marker(custom_data)
    return parsed is not None and parsed[0] == review_id


def has_note_prefix(custom_data: str) -> bool:
    """Whether ``custom_data`` claims to be a note marker (well-formed or not)."""
    return isinstance(custom_data, str) and custom_data.startswith(_NOTE_MARKER_PREFIX)
