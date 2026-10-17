"""Read-only MCP surface over the hub, for AI agents.

Built with the official MCP Python SDK. In mcp 2.x the ergonomic server class that
was ``FastMCP`` in v1 is ``MCPServer`` (``mcp.server.mcpserver``); the old
``mcp.server.fastmcp`` module is a tombstone that raises on import. This server
exposes only read tools; writing to the hub happens elsewhere. Tool names use
underscores rather than dots for the broadest client compatibility, and every tool
carries ``readOnlyHint`` so agents know it never mutates.

The hub is tailnet-only for v1 (DESIGN D2): there is no auth here, so the server
must be bound to a trusted interface by the caller and the database is opened
read-only.
"""

from __future__ import annotations

from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import ToolAnnotations

from aar.hub.artifacts import ArtifactStore
from aar.hub.events import Event, EventLog
from aar.hub.graph import WorkGraph
from aar.hub.registry import Registry

SERVER_INSTRUCTIONS = (
    "Shared memory for humans and AI agents working on DaVinci Resolve projects. "
    "Use it to discover which projects exist and where, what has happened (the event "
    "log), what work is planned or in flight (the work graph), and what has been "
    "produced (artifacts). This surface is read-only."
)

# Every tool here only reads; advertise that to clients (SEP tool annotations).
_READ_ONLY = ToolAnnotations(read_only_hint=True)

_MAX_TAIL = 1000


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
) -> MCPServer:
    """Build the read-only hub MCP server over the given hub components."""
    server: MCPServer = MCPServer(name=name, instructions=SERVER_INSTRUCTIONS)

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
        return {"projects": [project.to_dict() for project in registry.list()]}

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
        if project is None:
            return {"found": False, "project_id": project_id}
        return {"found": True, "project": project.to_dict()}

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
        return {"events": [_event_dict(event) for event in events], "next_seq": next_seq}

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
        return {"found": True, **graph.to_dict()}

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
        return {"found": True, "meta": meta.to_dict()}

    return server
