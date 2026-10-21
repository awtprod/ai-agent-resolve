"""MCP surface over the hub, for AI agents: read-only unless agent writes are enabled.

Built with the official MCP Python SDK. In mcp 2.x the ergonomic server class that
was ``FastMCP`` in v1 is ``MCPServer`` (``mcp.server.mcpserver``); the old
``mcp.server.fastmcp`` module is a tombstone that raises on import. This server
exposes only read tools; writing to the hub happens elsewhere. Tool names use
underscores rather than dots for the broadest client compatibility, and every tool
carries ``readOnlyHint`` so agents know it never mutates.

The hub is tailnet-only for v1 (DESIGN D2): there is no auth here, so the server
must be bound to a trusted interface by the caller and the database is opened
read-only.

``build_server(..., agent_writes=True)`` (``aar hub serve --agent-writes``) also registers
the agent write tools of :mod:`aar.hub.mcp_writes` (briefs, recipes, plans, own status,
facts) over a writable log. Without it the tool list is exactly the read-only tools above.
"""

from __future__ import annotations

import base64
import binascii
import json
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import ToolAnnotations

from aar.hub import approvals, control, jobs, reviews
from aar.hub.artifacts import ArtifactStore
from aar.hub.collab import branches as collab_branches
from aar.hub.collab import project_board as collab_project_board
from aar.hub.common import Clock, skipped_summary, utcnow
from aar.hub.events import Event, EventLog
from aar.hub.graph import WorkGraph
from aar.hub.leaseboard import lease_board as project_lease_board
from aar.hub.registry import Registry
from aar.hub.writes import ParamsValidator, accept_any_params

SERVER_INSTRUCTIONS = (
    "Shared memory for humans and AI agents working on DaVinci Resolve projects. "
    "Use it to discover which projects exist and where, what has happened (the event "
    "log), what work is planned or in flight (the work graph), and what has been "
    "produced (artifacts). Call control_status before hot work: if it reports a stop, "
    "do not touch Resolve. Results carry skipped: {count, events}, the malformed log "
    "events that were ignored. This surface is read-only."
)

WRITE_INSTRUCTIONS = (
    "Shared memory for humans and AI agents working on DaVinci Resolve projects. "
    "Use it to discover which projects exist and where, what has happened (the event "
    "log), what work is planned or in flight (the work graph), and what has been "
    "produced (artifacts). Call control_status before hot work: if it reports a stop, "
    "do not touch Resolve. Results carry skipped: {count, events}, the malformed log "
    "events that were ignored. Agent writes are enabled: create briefs (brief_create), "
    "recipe runs (recipe_create) and plans (plan_submit), move your own nodes' status "
    "(node_set_status) and record project facts (facts_record, facts_retract). Every "
    "write needs your agent_id (self-declared, for attribution only) and an "
    "idempotency_key; retrying with the same key is safe. Deciding reviews, approvals "
    "and resuming after a stop stay with humans."
)

# Every tool here only reads; advertise that to clients (SEP tool annotations).
_READ_ONLY = ToolAnnotations(read_only_hint=True)

_MAX_TAIL = 1000
# Deeper values are elided from tool results: the MCP serializer fails on nesting
# past about 250 levels, and no agent needs a payload that deep.
MAX_RESULT_DEPTH = 64
ELIDED = "<elided: nested deeper than 64 levels>"

_MAX_PROJECT_ID = 256
_MAX_BOARD_ROWS = 500
_MAX_BOARD_CURSOR = 2048
_MAX_BRANCHES_PAGE = 50
_MAX_BRANCHES_LIMIT = 100
_MAX_BRANCHES_BYTES = 256 * 1024
_MAX_JOBS_PAGE = 100
_MAX_JOBS_LIMIT = 500


def _bounded(value: Any, depth: int = 0) -> Any:
    """``value`` with anything nested deeper than ``MAX_RESULT_DEPTH`` replaced by a marker.

    It recurses, but never past the cap, so it cannot itself hit the recursion limit.
    """
    if isinstance(value, dict):
        if depth >= MAX_RESULT_DEPTH:
            return ELIDED
        return {key: _bounded(item, depth + 1) for key, item in value.items()}
    if isinstance(value, list | tuple):
        if depth >= MAX_RESULT_DEPTH:
            return ELIDED
        return [_bounded(item, depth + 1) for item in value]
    return value


def _board_cursor(key: tuple[str, str, str]) -> str:
    """An opaque ``project_board`` cursor: the sort key of the last lease on a page."""
    raw = json.dumps(list(key), separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).decode("ascii")


def _board_after(cursor: str) -> tuple[str, str, str]:
    """The sort key a ``project_board`` cursor names; a ``ToolError`` if it is not one."""
    bad = ToolError("cursor must be a next_cursor value returned by project_board")
    if not isinstance(cursor, str) or not 0 < len(cursor) <= _MAX_BOARD_CURSOR:
        raise bad
    try:
        key = json.loads(base64.urlsafe_b64decode(cursor.encode("ascii")))
    except (UnicodeError, binascii.Error, ValueError):
        raise bad from None
    if not (isinstance(key, list) and len(key) == 3 and all(isinstance(k, str) for k in key)):
        raise bad
    return (key[0], key[1], key[2])


def _event_dict(event: Event) -> dict[str, Any]:
    return {
        "seq": event.seq,
        "id": event.id,
        "project_id": event.project_id,
        "type": event.type,
        "actor": {"kind": event.actor.kind, "id": event.actor.id},
        "ts": event.ts,
        "payload": event.payload,
        "correlation_id": event.correlation_id,
        "causation_id": event.causation_id,
    }


def build_server(
    log: EventLog,
    registry: Registry,
    artifacts: ArtifactStore,
    *,
    name: str = "aar-hub",
    clock: Clock = utcnow,
    agent_writes: bool = False,
    params_validator: ParamsValidator = accept_any_params,
) -> MCPServer:
    """Build the hub MCP server over the given hub components.

    ``clock`` decides what has expired (leases, presence, approval tokens). With
    ``agent_writes`` the agent write tools are registered too (the log must be
    writable); ``params_validator`` checks recipe params for ``recipe_create``.
    """
    instructions = WRITE_INSTRUCTIONS if agent_writes else SERVER_INSTRUCTIONS
    server: MCPServer = MCPServer(name=name, instructions=instructions)

    @server.tool(
        name="projects_list",
        description=(
            'List every project the hub knows about. Returns {"projects": [...]}, each '
            "with its stable id (prj_...), display name, aliases, and sources (where the "
            "project lives: a cloud library, a Local Database, or a .drp/.drt file on a "
            "named host). Use this first to find a project id for the other tools."
        ),
        annotations=_READ_ONLY,
    )
    def projects_list() -> dict[str, Any]:
        registry.refresh()
        return _bounded(
            {
                "projects": [project.to_dict() for project in registry.list()],
                "skipped": skipped_summary(registry.skipped()),
            }
        )

    @server.tool(
        name="project_get",
        description=(
            'Get one project by its stable id. Returns {"found": true, "project": {...}} '
            'with the project\'s name, aliases and sources, or {"found": false} if the id is '
            "unknown. Use projects_list to discover ids."
        ),
        annotations=_READ_ONLY,
    )
    def project_get(project_id: str) -> dict[str, Any]:
        registry.refresh()
        project = registry.get(project_id)
        skipped = skipped_summary(registry.skipped())
        if project is None:
            return {"found": False, "project_id": project_id, "skipped": skipped}
        return _bounded({"found": True, "project": project.to_dict(), "skipped": skipped})

    @server.tool(
        name="events_tail",
        description=(
            'Read the append-only event log in order. Returns {"events": [...], '
            '"next_seq": N}; pass next_seq back as since_seq to page forward and resume '
            "where you left off. Omit project_id for all events, or pass one to see only a "
            "project's history. limit is clamped to 1..1000. Each event has seq, id, type, "
            "actor, ts and payload. Use this to replay what happened or catch up."
        ),
        annotations=_READ_ONLY,
    )
    def events_tail(
        project_id: str | None = None, since_seq: int = 0, limit: int = 100
    ) -> dict[str, Any]:
        # Raise ToolError (not a bare exception) so the model sees the real reason
        # rather than a generic "Error executing tool".
        if limit < 1:
            raise ToolError(f"limit must be between 1 and {_MAX_TAIL}")
        if since_seq < 0:
            raise ToolError("since_seq must not be negative")
        limit = min(_MAX_TAIL, limit)
        events = log.read(since_seq, project_id=project_id, limit=limit)
        next_seq = events[-1].seq if events else since_seq
        return _bounded({"events": [_event_dict(event) for event in events], "next_seq": next_seq})

    @server.tool(
        name="work_graph",
        description=(
            "Get a project's work graph: what is planned and in flight. Returns "
            '{"found": true, "nodes": [...], "edges": [...]} or {"found": false} if the '
            "project id is unknown. Nodes are units of work (brief, plan, op, artifact, "
            "review) each with a lifecycle status (pending, active, blocked, done, failed, "
            "cancelled). Edges are typed (derives_from, produces, reviews, blocks)."
        ),
        annotations=_READ_ONLY,
    )
    def work_graph(project_id: str) -> dict[str, Any]:
        registry.refresh()
        if registry.get(project_id) is None:
            return {"found": False, "project_id": project_id}
        graph = WorkGraph(log, project_id)
        return _bounded(
            {"found": True, **graph.to_dict(), "skipped": skipped_summary(graph.skipped())}
        )

    @server.tool(
        name="artifact_meta",
        description=(
            "Get the metadata for an artifact by its content id (a SHA-256 hex string). "
            'Returns {"found": true, "meta": {...}} with size, mime, project, producing '
            'op and created time, or {"found": false} if the id is unknown. This returns '
            "metadata only, never the bytes."
        ),
        annotations=_READ_ONLY,
    )
    def artifact_meta(artifact_id: str) -> dict[str, Any]:
        meta = artifacts.get_meta(artifact_id)
        if meta is None:
            return {"found": False, "artifact_id": artifact_id}
        return _bounded({"found": True, "meta": meta.to_dict()})

    @server.tool(
        name="reviews_pending",
        description=(
            "List open review requests (renders waiting for a human to approve or reject). "
            'Returns {"reviews": [...]}, oldest first, each with id, project_id, title, '
            "status, subject (the work-graph node under review), renders (artifact ids), "
            "timeline, fps, notes at their frames/timecodes, and decision (null while "
            "open). Omit project_id for every project."
        ),
        annotations=_READ_ONLY,
    )
    def reviews_pending(project_id: str | None = None) -> dict[str, Any]:
        scan = reviews.scan_reviews(log, project_id, pending_only=True)
        return {
            "reviews": [review.to_dict() for review in scan.reviews],
            "skipped": skipped_summary(scan.skipped),
        }

    @server.tool(
        name="review_get",
        description=(
            'Get one review by project id and review id. Returns {"found": true, '
            '"review": {...}} with its status, renders, notes (frame, timecode, text, '
            "author) and decision ({decision: approve|reject, reason, by, ts} once a "
            'human decided, else null), or {"found": false}. A review is approved only '
            "when decision.decision is 'approve'."
        ),
        annotations=_READ_ONLY,
    )
    def review_get(project_id: str, review_id: str) -> dict[str, Any]:
        review, skipped = reviews.review_view(log, project_id, review_id)
        summary = skipped_summary(skipped)
        if review is None:
            return {
                "found": False,
                "project_id": project_id,
                "review_id": review_id,
                "skipped": summary,
            }
        return {"found": True, "review": review.to_dict(), "skipped": summary}

    @server.tool(
        name="control_status",
        description=(
            'Emergency-stop status. Returns {"running": bool, "stops": [...]}; each stop '
            "has scope ('all', 'project:<id>' or 'node:<id>'), reason, by and at. If any "
            "stop covers your project or node, do no hot work until a human resumes it."
        ),
        annotations=_READ_ONLY,
    )
    def control_status() -> dict[str, Any]:
        state = control.status(log)
        return {**state.to_dict(), "skipped": skipped_summary(state.skipped)}

    @server.tool(
        name="lease_board",
        description=(
            "Who holds what on each Resolve node, and where a human is working. Returns "
            '{"as_of": ts, "stale": bool, "leases": [...], "presence": [...], "gaps": [...], '
            '"superseded": [...]}. A lease has node_id, scope (instance, project, timeline), '
            "key, holder {kind: human|agent, id}, since and expires_at; expired leases, and "
            "leases a node no longer holds by its latest snapshot, are already left out. "
            "Presence entries mean a human is active on that node since the given time. "
            "stale means a node's feed dropped events (gaps) or a second feed process for "
            "one node id was refused (superseded), and the node has not reconciled since."
        ),
        annotations=_READ_ONLY,
    )
    def lease_board() -> dict[str, Any]:
        board = project_lease_board(log, clock=clock)
        return {**board.to_dict(), "skipped": skipped_summary(board.skipped)}

    @server.tool(
        name="project_board",
        description=(
            "The lease board narrowed to one project: who holds the project and which of "
            "its timelines, and which humans are active. project_id is the Resolve project's "
            'unique id (the first part of a lease key). Returns {"project_id": id, "as_of": '
            'ts, "stale": bool, "leases": [...], "presence": [...], "gaps": [...], '
            '"next_cursor": str|null, "truncated": bool}; stale means a node\'s feed '
            "dropped events and has not reconciled yet. Leases come a page at a time, at "
            "most 500, in a fixed order: pass next_cursor back as cursor for the next page "
            "(a lease released between pages never shifts the others). Instance leases are "
            "left out; presence is node-level. Use this when a hot op is refused or delayed "
            "to see who holds what."
        ),
        annotations=_READ_ONLY,
    )
    def project_board(
        project_id: str, cursor: str | None = None, limit: int = _MAX_BOARD_ROWS
    ) -> dict[str, Any]:
        if not isinstance(project_id, str) or not 0 < len(project_id) <= _MAX_PROJECT_ID:
            raise ToolError(f"project_id must be 1 to {_MAX_PROJECT_ID} characters")
        after = None if cursor is None else _board_after(cursor)
        limit = max(1, min(_MAX_BOARD_ROWS, limit))
        board = collab_project_board(log, project_id, clock=clock)
        # A keyset page: the leases after the cursor's sort key, not an offset, so a lease
        # released (or expired) between pages never makes the next page skip a live one.
        rest = [lease for lease in board.leases if after is None or lease.sort_key > after]
        page = rest[:limit]
        next_cursor = _board_cursor(page[-1].sort_key) if len(rest) > len(page) else None
        data = board.to_dict()
        data["leases"] = [lease.to_dict() for lease in page]
        presence = data["presence"]
        data["presence"] = presence[:_MAX_BOARD_ROWS]
        return {
            **data,
            "next_cursor": next_cursor,
            "truncated": next_cursor is not None or len(presence) > _MAX_BOARD_ROWS,
            "skipped": skipped_summary(board.skipped),
        }

    @server.tool(
        name="agent_branches",
        description=(
            "List the agent branch timelines recorded for a project. Each entry has the "
            "source timeline (id, name), the owning agent_id, the branch timeline (id, name), "
            "its version, and a handoff note (merging back is a human step: Compare "
            'Timelines). Returns {"branches": [...], "next_cursor": N|null}; pass next_cursor '
            "back as cursor to page (a page also ends early to stay small). Omit project_id "
            "for every project."
        ),
        annotations=_READ_ONLY,
    )
    def agent_branches(
        project_id: str | None = None, cursor: int = 0, limit: int = _MAX_BRANCHES_PAGE
    ) -> dict[str, Any]:
        if project_id is not None and (
            not isinstance(project_id, str) or not 0 < len(project_id) <= _MAX_PROJECT_ID
        ):
            raise ToolError(f"project_id must be 1 to {_MAX_PROJECT_ID} characters")
        if cursor < 0:
            raise ToolError("cursor must not be negative")
        limit = max(1, min(_MAX_BRANCHES_LIMIT, limit))
        found = collab_branches(log, project_id)
        page: list[dict[str, Any]] = []
        size = 0
        for branch in found[cursor : cursor + limit]:
            row = branch.to_dict()
            size += len(json.dumps(row, ensure_ascii=False).encode())
            if page and size > _MAX_BRANCHES_BYTES:
                break
            page.append(row)
        end = cursor + len(page)
        return {"branches": page, "next_cursor": end if end < len(found) else None}

    @server.tool(
        name="approvals_list",
        description=(
            "List human approval tokens (metadata only; agents cannot grant). Returns "
            '{"approvals": [...]}, newest first, each with token_id, scope (db.switch, '
            "project.load, project.write, cloud.create, cloud.restore, switch.force), "
            "target, reason, state (active, expired, revoked, consumed), who granted it "
            "and when it expires. Set include_inactive to also list used or dead tokens."
        ),
        annotations=_READ_ONLY,
    )
    def approvals_list(include_inactive: bool = False) -> dict[str, Any]:
        now = clock()
        view = approvals.approvals_view(log)
        listed = [
            token.to_dict(now)
            for token in view.tokens
            if include_inactive or token.state(now) == "active"
        ]
        return {"approvals": listed, "skipped": skipped_summary(view.skipped)}

    @server.tool(
        name="jobs_list",
        description=(
            "List durable Resolve jobs, including waiting_for_node when its configured node "
            "has not claimed it and node_offline when a sticky claim's lease expired. "
            'Returns {"jobs": [...], "next_cursor": N|null}; payload params and approval '
            "references are omitted. Optional project_id and state filters are exact."
        ),
        annotations=_READ_ONLY,
    )
    def jobs_list(
        project_id: str | None = None,
        state: str | None = None,
        cursor: int = 0,
        limit: int = _MAX_JOBS_PAGE,
    ) -> dict[str, Any]:
        if project_id is not None and (
            not isinstance(project_id, str) or not 0 < len(project_id) <= _MAX_PROJECT_ID
        ):
            raise ToolError(f"project_id must be 1 to {_MAX_PROJECT_ID} characters")
        if not isinstance(cursor, int) or isinstance(cursor, bool) or cursor < 0:
            raise ToolError("cursor must be a non-negative integer")
        if not isinstance(limit, int) or isinstance(limit, bool):
            raise ToolError("limit must be an integer")
        limit = max(1, min(_MAX_JOBS_LIMIT, limit))
        now = clock()
        found = [
            job
            for job in jobs.list_jobs(log)
            if (project_id is None or job.target.project_id == project_id)
            and (state is None or job.state(now) == state)
        ]
        page = found[cursor : cursor + limit]
        end = cursor + len(page)
        return {
            "jobs": [job.to_dict(now, include_payload=False) for job in page],
            "next_cursor": end if end < len(found) else None,
        }

    if agent_writes:
        from aar.hub.mcp_writes import register_write_tools

        register_write_tools(server, log, registry, artifacts, params_validator=params_validator)
    return server
