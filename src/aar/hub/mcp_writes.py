"""Opt-in MCP write tools for agents (``aar hub serve --agent-writes``).

Registered on the hub's MCP server only when it is built with ``agent_writes=True``;
without it the server keeps exactly its read-only tools. Each write tool takes the
caller's ``agent_id`` and records the write as ``Actor("agent", agent_id)``. In v1 that id
is self-declared: it is attribution, not authentication (DESIGN decision 6). The hub is
tailnet-only and anyone who can reach it can claim any agent id.

Human-only actions are never tools here: deciding a review, granting or revoking an
approval, and resuming after an emergency stop stay in the console.

A refused write returns an error result (``isError``) whose structured content is
``{"ok": false, "error": {code, message, fix_hint, details}}``; a successful one returns
``{"ok": true, "op", "replayed", "project_id", ...}``.
"""

from __future__ import annotations

import json
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp_types import CallToolResult, TextContent, ToolAnnotations
from pydantic import StrictInt

from aar.hub import facts as facts_mod
from aar.hub import jobs, writes
from aar.hub.artifacts import ArtifactStore
from aar.hub.common import HumanRequired, InvalidInput, skipped_summary, utcnow
from aar.hub.events import Actor, EventLog, IdempotencyConflict
from aar.hub.graph import WorkGraph
from aar.hub.mcp_server import _bounded
from aar.hub.registry import Registry
from aar.node.branches import valid_agent_id
from aar.node.locator import ProjectLocator

# Writes append to an append-only log (never destructive) and are idempotent per key.
_WRITE = ToolAnnotations(
    read_only_hint=False, destructive_hint=False, idempotent_hint=True, open_world_hint=False
)
_READ_ONLY = ToolAnnotations(read_only_hint=True)

WRITE_TOOL_NAMES = (
    "brief_create",
    "recipe_create",
    "plan_submit",
    "node_set_status",
    "facts_record",
    "facts_retract",
    "job_submit",
    "job_cancel",
)
READ_TOOL_NAMES = ("facts_get", "brief_get", "plan_get")

MAX_FACTS_PAGE = 200
DEFAULT_FACTS_PAGE = 100
MAX_FACTS_BYTES = 256 * 1024
MAX_PREFIX = 200

_IDENTITY = (
    "agent_id is your own agent id (1-128 of A-Z a-z 0-9 . _ -). It is self-declared and "
    "recorded for attribution only; the hub does not authenticate it (v1). "
    "idempotency_key (1-128 of A-Z a-z 0-9 . _ : / -) names this one request: a retry with "
    "the same key and the same content returns the first result with replayed=true and "
    "writes nothing; the same key with different content is refused with "
    "idempotency_conflict. Use a fresh key for every new request. "
)
_ERRORS = (
    "On refusal the tool result is an error whose content is {ok: false, error: {code, "
    "message, fix_hint, details}}; fix what fix_hint says and call again."
)


def _result(body: dict[str, Any], *, is_error: bool = False) -> CallToolResult:
    """A tool result carrying ``body`` both as structured content and as JSON text."""
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(body, sort_keys=True))],
        structured_content=body,
        is_error=is_error,
    )


def _refused(error: writes.WriteError) -> CallToolResult:
    return _result({"ok": False, "error": error.to_dict()}, is_error=True)


def _agent(agent_id: Any) -> Actor:
    if not isinstance(agent_id, str) or not valid_agent_id(agent_id):
        raise writes.WriteError(
            "invalid_input",
            f"agent_id {agent_id!r:.80} is not a valid agent id",
            "Use 1-128 characters of A-Z a-z 0-9 . _ - (the same id you use for branches).",
            field="agent_id",
        )
    return Actor("agent", agent_id)


def register_write_tools(
    server: MCPServer,
    log: EventLog,
    registry: Registry,
    artifacts: ArtifactStore,
    *,
    params_validator: writes.ParamsValidator = writes.accept_any_params,
) -> None:
    """Add the agent write tools, and their typed read companions, to ``server``."""
    if log.read_only:
        raise ValueError("agent write tools need a writable event log")

    def run(call: Any) -> CallToolResult:
        # Every write tool returns a CallToolResult (the SDK allows no union with it), so a
        # refusal can be an error result that still carries structured content.
        try:
            return _result(call().to_dict())
        except writes.WriteError as exc:
            return _refused(exc)

    def run_job(op: str, call: Any) -> CallToolResult:
        try:
            job, replayed = call()
            return _result(
                writes.WriteResult(
                    op,
                    replayed,
                    job.target.project_id,
                    {"job": _bounded(job.to_dict(utcnow(), include_payload=False))},
                ).to_dict()
            )
        except writes.WriteError as exc:
            return _refused(exc)
        except InvalidInput as exc:
            # Queue validators may wrap Pydantic/build errors containing input values.
            # Keep those internals out of the MCP result, including approval references.
            return _refused(
                writes.WriteError(
                    "invalid_input",
                    f"Invalid job input: {exc.field}.",
                    "Check the named field against the tool description and job kind's "
                    "params schema; use bounded JSON with finite numbers and an exact "
                    "project locator containing project_id.",
                    field=exc.field,
                )
            )
        except IdempotencyConflict:
            return _refused(
                writes.WriteError(
                    "idempotency_conflict",
                    "This key already names a different job request.",
                    "Retry the original request unchanged, or use a fresh idempotency_key.",
                )
            )
        except HumanRequired:
            return _refused(
                writes.WriteError(
                    "not_creator",
                    "Only the job's creator may cancel it through agent MCP.",
                    "Use your own job_id; another actor's job needs a human using "
                    "`aar hub job cancel`.",
                )
            )
        except jobs.JobError as exc:
            return _result(
                {
                    "ok": False,
                    "error": {
                        "code": exc.code,
                        "message": "The job queue refused this request.",
                        "fix_hint": (
                            "Refresh jobs_list and check the job_id and current state. "
                            "If the queue history is corrupt or full, ask a human to inspect it."
                        ),
                        "details": {},
                    },
                },
                is_error=True,
            )

    @server.tool(
        name="job_submit",
        description=(
            "Enqueue durable Resolve work for an assigned node, even while it is absent. "
            "Submission does not execute work or grant approval: node safety gates still apply. "
            + _IDENTITY
            + "The execution agent_id is always your same agent_id. kind names a registered "
            "job kind (ops.sequence or render); params is its JSON object (at most 256 KiB, "
            "32 levels deep, finite numbers), validated by the queue and op schemas. project "
            "is an exact locator {folder_path: [folder names], name, project_id}; project_id "
            "is required and unknown keys are refused. assigned_node is a configured node id "
            "(1-128 characters); priority is an integer from -1000000 to 1000000 (default 0). "
            "approval_token optionally references an existing human approval, never grants one. "
            "Example: kind 'ops.sequence', params {steps: [{op: 'timeline.list', params: {}}]}, "
            "project {folder_path: ['Shows'], name: 'Show', project_id: 'resolve-show'}, "
            "assigned_node 'edit-pc'. Returns {ok: true, op: 'job.submit', project_id, replayed, "
            "job: bounded metadata without params or approval references}. " + _ERRORS
        ),
        annotations=_WRITE,
    )
    def job_submit(
        agent_id: str,
        idempotency_key: str,
        kind: str,
        params: dict[str, Any],
        project: dict[str, Any],
        assigned_node: str,
        priority: StrictInt = 0,
        approval_token: str | None = None,
    ) -> CallToolResult:
        def submit() -> tuple[jobs.HubJob, bool]:
            actor = _agent(agent_id)
            writes.write_key(actor, idempotency_key)
            # ProjectLocator.from_dict accepts iterables for internal callers. MCP's
            # JSON boundary requires an array, never a string interpreted as characters.
            if "folder_path" in project and not isinstance(project["folder_path"], list):
                raise InvalidInput("project.folder_path", "must be an array of folder names")
            try:
                target = ProjectLocator.from_dict(project)
            except (KeyError, TypeError, ValueError) as exc:
                raise InvalidInput("project", "must be an exact project locator") from exc
            return jobs.submit(
                log,
                actor,
                kind=kind,
                params=params,
                project=target,
                agent_id=actor.id,
                assigned_node=assigned_node,
                idempotency_key=idempotency_key,
                priority=priority,
                approval_token=approval_token,
            )

        return run_job("job.submit", submit)

    @server.tool(
        name="job_cancel",
        description=(
            "Request cancellation of a job created by your agent_id. The node observes "
            "the request; this does not release or reassign its claim. "
            + _IDENTITY
            + "job_id is from job_submit or jobs_list; reason is 1-1000 characters. "
            "Other actors' jobs require a human using `aar hub job cancel`. Example: job_id "
            "'job_0123456789abcdef', reason 'superseded by a new cut'. Returns {ok: true, "
            "op: 'job.cancel', project_id, replayed, job: bounded metadata without params "
            "or approval references}. " + _ERRORS
        ),
        annotations=_WRITE,
    )
    def job_cancel(agent_id: str, idempotency_key: str, job_id: str, reason: str) -> CallToolResult:
        def cancel() -> tuple[jobs.HubJob, bool]:
            actor = _agent(agent_id)
            writes.write_key(actor, idempotency_key)
            return jobs.cancel(log, actor, job_id, reason, idempotency_key=idempotency_key)

        return run_job("job.cancel", cancel)

    @server.tool(
        name="brief_create",
        description=(
            "Create a brief: the typed description of one job (a long-form, a short or a "
            "thumbnail) that recipes and plans derive from. Call it first, before "
            "recipe_create and plan_submit. "
            + _IDENTITY
            + "brief is an object: project_id (prj_..., from projects_list), show, kind "
            "(longform | short | thumbnail), recipe (a recipe name, e.g. 'ccn-longform'), "
            "rule_profile (a built-in rule profile name, e.g. 'ccn'; plans for this brief "
            "are linted under it), deliverables (1-32 of {name, kind: master | review | "
            "short | thumbnail | captions}, names unique, 1-64 of A-Z a-z 0-9 _ -), "
            "sources (up to 64 of {kind: 'cloud_project', library, project} | {kind: "
            "'timeline', unique_id, name?} | {kind: 'media', path, host?, sha256?}, each "
            "with an optional label), and optional hook (300), handle (100), title (200), "
            'notes (8000). Unknown keys are refused. Example: {"project_id": "prj_1a2b", '
            '"show": "Acme Comedy", "kind": "short", "recipe": "ccn-short", '
            '"rule_profile": "ccn", "deliverables": [{"name": "short-1", "kind": "short"}], '
            '"sources": [{"kind": "timeline", "unique_id": "5f0c-77aa"}], "hook": '
            '"He never parks there again"}. Returns {ok: true, op: "brief.create", '
            "replayed, project_id, node: {id, kind: 'brief', status: 'pending', title, "
            "payload: the stored brief}}. " + _ERRORS
        ),
        annotations=_WRITE,
    )
    def brief_create(agent_id: str, idempotency_key: str, brief: dict[str, Any]) -> CallToolResult:
        return run(
            lambda: writes.create_brief(
                log, brief, _agent(agent_id), idempotency_key=idempotency_key
            )
        )

    @server.tool(
        name="recipe_create",
        description=(
            "Start a recipe run for a brief: adds a 'recipe' node (status pending) with a "
            "derives_from edge to the brief. Use it after brief_create, once per recipe run. "
            + _IDENTITY
            + "recipe is an object: recipe (must equal the brief's recipe name), brief_id "
            "(the brief node's id), params (optional JSON object, at most 64 KiB, nested at "
            "most 8 deep, finite numbers; checked by the recipe's own validator), "
            "max_review_rounds (optional, 1-10, default 3). Example: project_id 'prj_1a2b', "
            'recipe {"recipe": "ccn-short", "brief_id": "nd_0123456789abcdef", "params": '
            '{"max_seconds": 37}}. Returns {ok: true, op: "recipe.create", replayed, '
            "project_id, node: {id, kind: 'recipe', status, payload}, edge: {id, type: "
            "'derives_from', src: recipe id, dst: brief id}}. " + _ERRORS
        ),
        annotations=_WRITE,
    )
    def recipe_create(
        agent_id: str, idempotency_key: str, project_id: str, recipe: dict[str, Any]
    ) -> CallToolResult:
        return run(
            lambda: writes.create_recipe(
                log,
                project_id,
                recipe,
                _agent(agent_id),
                idempotency_key=idempotency_key,
                params_validator=params_validator,
            )
        )

    @server.tool(
        name="plan_submit",
        description=(
            "Submit an EditPlan for a brief. The hub validates it, lints it under its rule "
            "profile, stores it as a content-addressed artifact and adds a 'plan' node with "
            "a derives_from edge to derives_from (a brief, recipe, earlier plan or review "
            "node of the project; never an op or artifact). "
            + _IDENTITY
            + "plan is the EditPlan object (the same JSON `aar plan lint` reads, at most "
            "1 MiB). Its profile must be a built-in profile name and, when the plan derives "
            "from a brief (directly or through a recipe, plan or review), equal the brief's "
            "rule_profile. A plan that fails validation is refused with invalid_plan and "
            "details.errors naming each field location (loc, msg). A plan that validates but "
            "has lint errors is still stored, with status 'blocked'; with no errors it is "
            "'pending'. The hub lints without a transcript or laugh map, so the rules that "
            "need one are listed in node.payload.lint.skipped_rules: they did not run. "
            "title is optional. Example: project_id 'prj_1a2b', derives_from "
            '\'nd_0123456789abcdef\' (the recipe node), plan {"project": "Acme S01", '
            '"timeline": "Short 03", "kind": "short", "fps": "24000/1001", "resolution": '
            '{"width": 1080, "height": 1920}, "profile": "ccn", "sources": [{"id": "cam-a", '
            '"media": "media/camA.mov", "shot_size": "close", "recording_id": "take-1"}], '
            '"segments": [{"source": "cam-a", "src_in": 1000, "src_out": 1240}]}. '
            "Returns {ok: true, op: 'plan.submit', replayed, project_id, "
            "node: {id, kind: 'plan', status, payload: {schema: 'aar.plan-node/1', "
            "plan_sha256, plan_version, rule_profile, derives_from, brief_id, lint: {ok, "
            "errors, warnings, infos, skipped_rules, findings (errors first, at most 20), "
            "omitted}}}, edge, artifact_id}. " + _ERRORS
        ),
        annotations=_WRITE,
    )
    def plan_submit(
        agent_id: str,
        idempotency_key: str,
        project_id: str,
        derives_from: str,
        plan: dict[str, Any],
        title: str | None = None,
    ) -> CallToolResult:
        return run(
            lambda: writes.submit_plan(
                log,
                artifacts,
                project_id,
                plan,
                _agent(agent_id),
                derives_from=derives_from,
                idempotency_key=idempotency_key,
                title=title,
            )
        )

    @server.tool(
        name="node_set_status",
        description=(
            "Move a work-graph node you created to a new status. You may move only brief, "
            "recipe, plan and op nodes that this agent_id created; never a review (a human "
            "decides reviews) and never another agent's node. Allowed moves: pending -> "
            "active | blocked | cancelled; active -> blocked | done | failed | cancelled; "
            "blocked -> active | cancelled; failed -> active | cancelled; done and "
            "cancelled are final. "
            + _IDENTITY
            + "Example: project_id 'prj_1a2b', node_id 'nd_0123456789abcdef', status "
            "'active'. Returns {ok: true, op: 'node.set_status', replayed, project_id, "
            "node_id, kind, from, to}. Refusal codes include not_creator, kind_not_allowed, "
            "protected_review and illegal_transition (details.allowed lists the legal "
            "moves). " + _ERRORS
        ),
        annotations=_WRITE,
    )
    def node_set_status(
        agent_id: str, idempotency_key: str, project_id: str, node_id: str, status: str
    ) -> CallToolResult:
        return run(
            lambda: writes.set_own_status(
                log, project_id, node_id, status, _agent(agent_id), idempotency_key=idempotency_key
            )
        )

    shapes = "; ".join(
        f"{namespace.shape}: {_value_hint(namespace.name)}" for namespace in facts_mod.NAMESPACES
    )

    @server.tool(
        name="facts_record",
        description=(
            "Record a per-project fact that later work should reuse: a sync offset, a "
            "track gain, a template trap or id, a path map, a render hash or a QA number. "
            "The value becomes the key's current value (the last record wins; earlier "
            "values stay in its history). "
            + _IDENTITY
            + "key is '<namespace>.<part>' where every <part> is 1-64 of A-Z a-z 0-9 _ - "
            "(no dots), and value is an object that fits the namespace: "
            + shapes
            + ". source is optional {recipe_node: a recipe node id, artifact: a sha256 the "
            'hub stored}. Example: key \'sync.offset.wide\', value {"frames": 12, "fps": '
            '{"num": 24000, "den": 1001}, "relative_to": "close camera"}. Returns {ok: '
            "true, op: 'facts.record', replayed, project_id, fact: {key, namespace, value, "
            "source, by, at, event_id}}. Unknown namespaces are refused "
            "(unknown_namespace), values that do not fit too (invalid_fact). " + _ERRORS
        ),
        annotations=_WRITE,
    )
    def facts_record(
        agent_id: str,
        idempotency_key: str,
        project_id: str,
        key: str,
        value: dict[str, Any],
        source: dict[str, Any] | None = None,
    ) -> CallToolResult:
        return run(
            lambda: facts_mod.record_fact(
                log,
                project_id,
                key,
                value,
                _agent(agent_id),
                idempotency_key=idempotency_key,
                source=source,
            )
        )

    @server.tool(
        name="facts_retract",
        description=(
            "Retract a per-project fact that no longer holds (for example a path map after "
            "a drive moved). The key stops having a current value; its history is kept. "
            + _IDENTITY
            + "reason (1-1000 characters) says why. Example: key 'path.map.edit-pc', reason "
            "'the media drive moved'. Returns {ok: true, op: 'facts.retract', replayed, "
            "project_id, retracted: {key, namespace, reason, by, at, event_id}}. A key that "
            "is not set is refused with unknown_fact. " + _ERRORS
        ),
        annotations=_WRITE,
    )
    def facts_retract(
        agent_id: str, idempotency_key: str, project_id: str, key: str, reason: str
    ) -> CallToolResult:
        return run(
            lambda: facts_mod.retract_fact(
                log, project_id, key, reason, _agent(agent_id), idempotency_key=idempotency_key
            )
        )

    @server.tool(
        name="facts_get",
        description=(
            "Read a project's facts (offsets, gains, template traps and ids, path maps, "
            "render hashes, QA numbers). Returns {found: true, project_id, facts: [{key, "
            "namespace, value, source, set_by, set_at, history: [the last 10 changes]}], "
            "next_cursor, skipped} sorted by key, or {found: false} for an unknown project. "
            "prefix keeps keys that start with it (e.g. 'sync.offset.'). Retracted keys are "
            "left out unless include_retracted is true (then they appear with retracted: "
            "true). A page holds at most limit facts (1-200, default 100) and ends early to "
            "stay under 256 KiB: pass next_cursor back as cursor for the next page."
        ),
        annotations=_READ_ONLY,
    )
    def facts_get(
        project_id: str,
        prefix: str | None = None,
        include_retracted: bool = False,
        cursor: str | None = None,
        limit: int = DEFAULT_FACTS_PAGE,
    ) -> CallToolResult:
        try:
            writes.check_id(project_id, "project_id")
            for name, value in (("prefix", prefix), ("cursor", cursor)):
                if value is not None and (not isinstance(value, str) or len(value) > MAX_PREFIX):
                    raise writes.WriteError(
                        "invalid_input",
                        f"{name} must be at most {MAX_PREFIX} characters",
                        f"Pass a shorter {name}.",
                        field=name,
                    )
        except writes.WriteError as exc:
            return _refused(exc)
        registry.refresh()
        if registry.get(project_id) is None:
            return _result({"found": False, "project_id": project_id})
        view = facts_mod.facts_view(log, project_id)
        rows: list[dict[str, Any]] = []
        for fact in sorted(view.facts.values(), key=lambda item: item.key):
            if prefix is not None and not fact.key.startswith(prefix):
                continue
            if cursor is not None and fact.key <= cursor:
                continue
            if fact.is_set:
                rows.append(fact.to_dict())
            elif include_retracted:
                rows.append({**fact.to_dict(), "value": None, "retracted": True})
        limit = max(1, min(MAX_FACTS_PAGE, limit))
        page: list[dict[str, Any]] = []
        size = 0
        for row in rows[:limit]:
            size += len(json.dumps(row, ensure_ascii=False).encode())
            if page and size > MAX_FACTS_BYTES:
                break
            page.append(row)
        next_cursor = page[-1]["key"] if len(page) < len(rows) else None
        return _result(
            {
                "found": True,
                "project_id": project_id,
                "facts": page,
                "next_cursor": next_cursor,
                "skipped": skipped_summary(view.skipped),
            }
        )

    @server.tool(
        name="brief_get",
        description=(
            "Get one brief node with its typed data. Returns {found: true, node: {id, kind, "
            "status, title, payload}, typed: bool, brief: the aar.brief/1 data (null if the "
            "node holds untyped data), creator: {kind, id}, derived: [{node_id, kind, "
            "status}] (recipes and plans that derive from it, at most 100), skipped} or "
            "{found: false}. Use it to pick up a job after losing context."
        ),
        annotations=_READ_ONLY,
    )
    def brief_get(project_id: str, node_id: str) -> dict[str, Any]:
        view = writes.brief_view(log, project_id, node_id)
        if view is None:
            return {"found": False, "project_id": project_id, "node_id": node_id}
        skipped = skipped_summary(WorkGraph(log, project_id).skipped())
        return _bounded({"found": True, **view, "skipped": skipped})

    @server.tool(
        name="plan_get",
        description=(
            "Get one plan node: its status, its aar.plan-node/1 data (plan_sha256, "
            "plan_version, rule_profile, derives_from, brief_id, lint summary), its creator, "
            "what derives from it, and (unless include_plan is false) the EditPlan JSON "
            "itself, read from the artifact store and checked against its hash. Returns "
            "{found: true, node, typed, creator, derived, plan, plan_error, skipped} or "
            "{found: false}. plan_error says why the plan could not be served (missing or "
            "corrupt artifact)."
        ),
        annotations=_READ_ONLY,
    )
    def plan_get(project_id: str, node_id: str, include_plan: bool = True) -> dict[str, Any]:
        view = writes.plan_view(log, artifacts, project_id, node_id, include_plan=include_plan)
        if view is None:
            return {"found": False, "project_id": project_id, "node_id": node_id}
        skipped = skipped_summary(WorkGraph(log, project_id).skipped())
        return _bounded({"found": True, **view, "skipped": skipped})


_VALUE_HINTS = {
    "sync.offset": "{frames: int, fps: {num, den}, relative_to: text}",
    "audio.gain": "{db: -96..24}",
    "template.trap": "{text: up to 2000 characters}",
    "template.id": "{unique_id: a Resolve unique id}",
    "path.map": "{from: path, to: path}",
    "render.hash": "{sha256: 64 hex, bytes: int}",
    "qa": "{value: finite number, unit: e.g. LUFS, dBTP, s, MB}",
}


def _value_hint(namespace: str) -> str:
    return _VALUE_HINTS[namespace]
