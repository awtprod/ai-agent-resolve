"""S7 x S11 integration: the coordinator's policy P1-P5 and the validator's items 1-8.

Every test here fails on the pre-integration tree (S7 round 3 rebased onto S11f, before the
wiring). The worlds use ``own_timelines=False`` unless a test says otherwise, so "Main" is a
human's timeline, exactly as in production.
"""

from __future__ import annotations

import ast
import re
import sqlite3
import threading
from pathlib import Path

import pytest

from aar.hub import control
from aar.hub.approvals import ApprovalScope, grant
from aar.hub.collab import branches
from aar.hub.events import Actor
from aar.hub.registry import ProjectSource, Registry
from aar.node.broker import BrokerConfig, OpStatus
from aar.node.hublink import HubUnreachable, LocalHubLink, UnknownHubProject
from aar.node.jobs import ApprovalPolicy, JobStore
from aar.node.jobs.runner import op_approval_target
from aar.node.leases import Holder
from aar.node.ops import list_ops, submit_op
from aar.node.runtime import NodeIdInUse, NodeRuntime

from .fake_resolve import FakeRenderJob
from .jobs_support import AGENT_ID, MAIN_REF, JobWorld
from .ops_support import content_writes

HUMAN = Actor("human", "andrew")
SRC = Path(__file__).resolve().parents[2] / "src"


def _marker_job(*frames: int, timeline: dict | None = None) -> dict:
    ref = timeline or MAIN_REF
    return {
        "steps": [
            {"op": "markers.add", "params": {"timeline": ref, "markers": [{"frame": f}]}}
            for f in frames
        ]
    }


def _render_params(world: JobWorld, **over: object) -> dict:
    params = {
        "timeline": world.ref("Main"),
        "target_dir": "D:/Out",
        "custom_name": "show_v1",
        "preset": "YouTube 1080p",
        "render_format": "mov",
        "render_codec": "ProRes",
        "render_mode": 1,
        "mark_in": 0,
        "mark_out": 99,
    }
    params.update(over)
    return params


def _human_world(tmp_path: Path, **kwargs: object) -> JobWorld:
    return JobWorld(tmp_path, own_timelines=False, **kwargs)  # type: ignore[arg-type]


def _op_grant(world: JobWorld, op: str, project: str = "Show") -> str:
    target = op_approval_target(world.pid(project), op)
    return grant(
        world.log, ApprovalScope.PROJECT_WRITE, target, 3600, f"let {op} run", HUMAN
    ).token_id


# == P1: the branch guard applies in every project ================================================


def test_p1_human_timeline_edit_waits_for_an_approval_in_a_sandbox_project(tmp_path: Path) -> None:
    """The project is not production (``live=False``), yet an edit to the human's "Main" waits
    for an approval naming the op; nothing is written and the broker never runs it."""
    world = _human_world(tmp_path, approval=ApprovalPolicy(live=False))
    job_id = world.submit("ops.sequence", _marker_job(100), world.locator())
    world.runner.run_until_settled()
    view = world.runner.get(job_id)
    assert view.state == "waiting" and view.waiting["kind"] == "needs_approval"
    detail = view.waiting["detail"]
    assert detail["target"] == f"{world.pid()}#op:markers.add"
    assert detail["op"] == "markers.add" and detail["guard"]["code"] == "not_an_agent_branch"
    assert world.main.markers == {} and content_writes(world.fake) == []
    assert world.broker.pending() == []  # never queued


def test_p1_shared_state_write_waits_for_an_approval_in_a_sandbox_project(tmp_path: Path) -> None:
    """``clip.set_input_color_space`` changes state every timeline shares: approval needed."""
    world = _human_world(tmp_path, approval=ApprovalPolicy(live=False))
    clip = world.project.media_pool.import_rec709(("Footage",), "lower third")
    job = {
        "steps": [
            {
                "op": "clip.set_input_color_space",
                "params": {
                    "items": [{"unique_id": clip.GetUniqueId()}],
                    "color_space": "Rec.709 (Scene)",
                },
            }
        ]
    }
    job_id = world.submit("ops.sequence", job, world.locator())
    world.runner.run_until_settled()
    view = world.runner.get(job_id)
    assert view.state == "waiting"
    assert view.waiting["detail"]["guard"]["code"] == "requires_approval"
    assert content_writes(world.fake) == []


def test_p1_the_agents_own_branch_needs_no_approval_outside_production(tmp_path: Path) -> None:
    world = _human_world(tmp_path, approval=ApprovalPolicy(live=False))
    world.own("Main [agent]")
    job_id = world.submit(
        "ops.sequence", _marker_job(7, timeline=world.ref("Main [agent]")), world.locator()
    )
    world.runner.run_until_settled()
    assert world.runner.get(job_id).state == "succeeded"
    assert sorted(world.project.timeline("Main [agent]").markers) == [7]
    assert world.log.read(types=["approval.consumed"]) == []


# == P4: a consumed approval naming the op lets that op alone past the guard ======================


def test_p4_an_approval_naming_the_op_lets_it_run_once_and_is_audited(tmp_path: Path) -> None:
    world = _human_world(tmp_path)
    job_id = world.submit("ops.sequence", _marker_job(100), world.locator())
    world.runner.run_until_settled()
    token = _op_grant(world, "markers.add")
    world.runner.approve(job_id, token, HUMAN)
    world.runner.run_until_settled()
    view = world.runner.get(job_id)
    assert view.state == "succeeded" and sorted(world.main.markers) == [100]
    consumed = world.log.read(types=["approval.consumed"])
    assert len(consumed) == 1  # the hub audit: one consume, by the job's agent, for this job
    assert consumed[0].actor == Actor("agent", f"{AGENT_ID}#{job_id}")
    assert consumed[0].payload["target"] == f"{world.pid()}#op:markers.add"
    overrides = [e for e in view.events if e["kind"] == "guard_override"]
    assert len(overrides) == 1 and overrides[0]["data"]["token_id"] == token
    assert view.approvals == {"op:0/0:markers.add/markers.add": token}


@pytest.mark.parametrize("which", ["project", "other-op", "other-project"])
def test_p4_an_approval_for_anything_else_is_refused(tmp_path: Path, which: str) -> None:
    """Only a token on exactly ``<project>#op:<op>`` lifts the guard: a plain project.write
    (the F15 production target), one naming another op, or another project's, is refused."""
    world = _human_world(tmp_path)
    job_id = world.submit("ops.sequence", _marker_job(100), world.locator())
    world.runner.run_until_settled()
    if which == "project":
        target = world.pid()
    elif which == "other-op":
        target = op_approval_target(world.pid(), "timeline.duplicate")
    else:
        target = op_approval_target(world.pid("Other"), "markers.add")
    token = grant(world.log, ApprovalScope.PROJECT_WRITE, target, 3600, "wrong", HUMAN)
    world.runner.approve(job_id, token.token_id, HUMAN)
    world.runner.run_until_settled()
    view = world.runner.get(job_id)
    assert view.state == "waiting" and view.waiting["detail"]["refused_code"] == "wrong_target"
    assert world.main.markers == {} and view.approvals == {}


def test_p4_each_guarded_op_needs_its_own_approval(tmp_path: Path) -> None:
    """Single use, per op: after the first approved write, the second waits for another."""
    world = _human_world(tmp_path)
    job_id = world.submit("ops.sequence", _marker_job(100, 200), world.locator())
    world.runner.run_until_settled()
    world.runner.approve(job_id, _op_grant(world, "markers.add"), HUMAN)
    world.runner.run_until_settled()
    view = world.runner.get(job_id)
    assert sorted(world.main.markers) == [100]
    assert view.state == "waiting" and view.current_step == 1
    world.runner.approve(job_id, _op_grant(world, "markers.add"), HUMAN)
    world.runner.run_until_settled()
    assert world.runner.get(job_id).state == "succeeded"
    assert sorted(world.main.markers) == [100, 200]
    assert len(world.log.read(types=["approval.consumed"])) == 2


def test_p4_a_consumed_override_survives_a_restart_and_is_not_consumed_again(
    tmp_path: Path,
) -> None:
    world = _human_world(tmp_path)
    job_id = world.submit("ops.sequence", _marker_job(100), world.locator())
    world.runner.run_until_settled()
    world.runner.approve(job_id, _op_grant(world, "markers.add"), HUMAN)
    world.runner.pump()  # consumes, submits the op (queued), then the node dies
    assert world.store.get(job_id).approvals
    assert world.main.markers == {}
    runner = world.restart()
    runner.run_until_settled()
    assert runner.get(job_id).state == "succeeded" and sorted(world.main.markers) == [100]
    assert len(world.log.read(types=["approval.consumed"])) == 1


def test_p4_a_forged_override_is_not_trusted(tmp_path: Path) -> None:
    """An ``approvals`` entry the hub does not confirm (here: a token another job consumed)
    is dropped on load; the op waits for its own approval."""
    world = _human_world(tmp_path)
    a = world.submit("ops.sequence", _marker_job(100), world.locator())
    world.runner.run_until_settled()
    token = _op_grant(world, "markers.add")
    world.runner.approve(a, token, HUMAN)
    world.runner.run_until_settled()
    b = world.submit("ops.sequence", _marker_job(200), world.locator())
    world.runner.run_until_settled()
    world.runtime.close()
    world.store.close()
    conn = sqlite3.connect(world.db_path)
    key = "op:0/0:markers.add/markers.add"
    conn.execute("UPDATE jobs SET approvals=? WHERE job_id=?", (f'{{"{key}": "{token}"}}', b))
    conn.commit()
    conn.close()
    world.store = JobStore(world.db_path)
    runner = world.restart()
    runner.run_until_settled()
    view = runner.get(b)
    assert view.state == "waiting" and view.approvals == {}
    assert "approval_unverified" in [e["kind"] for e in view.events]
    assert sorted(world.main.markers) == [100]


def _code_lines(path: Path) -> list[str]:
    """``path``'s source with every comment and string (docstrings too) blanked out."""
    import tokenize

    lines = path.read_text().splitlines()
    out = [list(line) for line in lines]
    with path.open("rb") as fh:
        for tok in tokenize.tokenize(fh.readline):
            if tok.type not in (tokenize.COMMENT, tokenize.STRING):
                continue
            (r0, c0), (r1, c1) = tok.start, tok.end
            for r in range(r0, r1 + 1):
                row = out[r - 1]
                lo = c0 if r == r0 else 0
                hi = c1 if r == r1 else len(row)
                row[lo:hi] = [" "] * (hi - lo)
    return ["".join(row) for row in out]


def _require_branch_keywords() -> list[tuple[str, str, str]]:
    """Every ``require_branch=<expr>`` keyword argument in ``src``: (file, function, expr)."""
    found: list[tuple[str, str, str]] = []
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for func in ast.walk(tree):
            if not isinstance(func, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            for call in ast.walk(func):
                if isinstance(call, ast.Call):
                    for kw in call.keywords:
                        if kw.arg == "require_branch":
                            rel = path.relative_to(SRC).as_posix()
                            found.append((rel, func.name, ast.unparse(kw.value)))
    return sorted(set(found))


def test_p4_the_only_production_path_to_require_branch_false_is_the_approved_op() -> None:
    """P4: no production code reaches ``require_branch=False`` except the runner's override,
    built only after a consumed approval naming the op. A grep (text) and an AST walk (any
    expression passed as ``require_branch``) must agree."""
    text_hits = [
        (path.relative_to(SRC).as_posix(), line)
        for path in sorted(SRC.rglob("*.py"))
        for line in _code_lines(path)
        if re.search(r"require_branch\s*=\s*False", line)
    ]
    assert [f for f, _ in text_hits] == ["aar/node/jobs/runner.py"], text_hits
    assert _require_branch_keywords() == [
        ("aar/node/jobs/runner.py", "_override_op", "False"),
        ("aar/node/ops/framework.py", "submit_op", "require_branch"),  # forwards its argument
    ]
    runner_src = (SRC / "aar/node/jobs/runner.py").read_text()
    body = runner_src[runner_src.index("    def _override_op(") :]
    body = body[: body.index("\n    def ", 10)]
    assert "not in job.approvals" in body  # it refuses without the consumed approval
    callers = [line for line in runner_src.splitlines() if "self._override_op(" in line]
    assert len(callers) == 1  # one call site, right after _consume_approval succeeded


# == P2: every op runs as the job's own agent =====================================================


@pytest.mark.parametrize(
    "agent_id",
    ["", "job-runner:node-shows", "has space", "x" * 129, "tab\tid", None, 7],
)
def test_p2_a_job_with_no_valid_agent_id_is_refused_at_submit(tmp_path: Path, agent_id) -> None:
    world = _human_world(tmp_path)
    with pytest.raises(ValueError, match="agent"):
        world.runner.submit("ops.sequence", _marker_job(100), world.locator(), agent_id=agent_id)
    assert world.runner.list() == [] and world.store.load_all() == []


def test_p2_a_job_with_no_agent_id_at_all_is_refused(tmp_path: Path) -> None:
    world = _human_world(tmp_path)
    with pytest.raises(TypeError):
        world.runner.submit("ops.sequence", _marker_job(100), world.locator())  # type: ignore[call-arg]
    assert world.runner.list() == [] and world.store.load_all() == []


def test_p2_ops_run_and_branches_are_owned_as_the_jobs_agent(tmp_path: Path) -> None:
    """The broker sees the job's agent as the op holder; a branch the job cuts is owned by
    that agent; another agent's job may not edit it."""
    world = _human_world(tmp_path)
    seen: list[str] = []
    real = world.broker._emit

    def spy(kind: str, **data: object) -> None:
        if kind == "op_started":
            seen.append(str(data["holder"]))
        real(kind, **data)

    world.broker._emit = spy  # type: ignore[method-assign]
    job = {"steps": [{"op": "timeline.branch", "params": {"source": world.ref("Main")}}]}
    job_id = world.submit("ops.sequence", job, world.locator())
    world.runner.run_until_settled()
    view = world.runner.get(job_id)
    assert view.state == "succeeded" and view.agent_id == AGENT_ID
    assert seen == [str(Holder.agent(AGENT_ID))]
    branch = view.results[0]["timeline"]
    assert world.branch_registry.owned_by(world.pid(), branch["unique_id"], AGENT_ID)
    other = world.submit(
        "ops.sequence", _marker_job(5, timeline=branch), world.locator(), agent_id="other-agent"
    )
    world.runner.run_until_settled()
    assert world.runner.get(other).state == "waiting"  # not its branch: guarded
    mine = world.submit("ops.sequence", _marker_job(5, timeline=branch), world.locator())
    world.runner.run_until_settled()
    assert world.runner.get(mine).state == "succeeded"  # the owner edits it freely


# == P3: render ops have a named guard exemption and never change timeline content ================


def test_p3_exactly_the_render_ops_carry_a_guard_exemption() -> None:
    exempt = {s.name: s.guard_exemption for s in list_ops() if s.guard_exemption is not None}
    assert set(exempt) == {"render.build", "render.delete_job"}
    assert all(reason and "render" in reason for reason in exempt.values())
    # The other guard pass is S11's creates_timeline, on exactly the two copy-making ops.
    assert {s.name for s in list_ops() if s.creates_timeline} == {
        "timeline.duplicate",
        "timeline.branch",
    }
    assert all(s.mutates for s in list_ops() if s.guard_exemption is not None)


def test_p3_a_render_of_a_human_timeline_never_changes_its_content(tmp_path: Path) -> None:
    """Rendering the human's own "Main" is allowed (CCN delivery needs it), with no approval
    outside production, and makes no content write: the timeline, its markers and every
    other timeline read back exactly as before; Deliver is restored; only our job is touched."""
    world = _human_world(tmp_path)
    human = FakeRenderJob("rj-human", {"JobId": "rj-human", "TimelineName": "Main"})
    world.project.render_jobs.append(human)
    world.main.AddMarker(10, "Red", "human note", "", 1, "")
    before = {
        t.GetUniqueId(): (t.GetName(), dict(t.markers), dict(t.GetSettings() or {}))
        for t in world.project.timelines
    }
    original = world.project.deliver.copy()
    world.fake.calls.clear()
    job_id = world.submit("render", _render_params(world), world.locator())
    world.runner.run_until_settled()
    view = world.runner.get(job_id)
    assert view.state == "succeeded", view.error
    assert content_writes(world.fake) == []
    after = {
        t.GetUniqueId(): (t.GetName(), dict(t.markers), dict(t.GetSettings() or {}))
        for t in world.project.timelines
    }
    assert after == before
    assert world.project.deliver.settings == original.settings
    assert [j.job_id for j in world.project.render_jobs] == ["rj-human"]
    assert world.log.read(types=["approval.consumed"]) == []


def test_p3_render_delete_job_cannot_delete_a_render_job_it_did_not_create(tmp_path: Path) -> None:
    """The exemption is safe only because delete runs inside a job whose journal proves the
    JobId is its own: submitted directly, it is refused and the human's job survives."""
    world = _human_world(tmp_path)
    world.project.render_jobs.append(FakeRenderJob("rj-human", {"JobId": "rj-human"}))
    op_id = submit_op(
        world.broker,
        "render.delete_job",
        {"render_job_id": "rj-human"},
        target=world.locator(),
        holder=Holder.agent(AGENT_ID),
        branch_registry=world.branch_registry,
    )
    world.broker.run_until_idle()
    result = world.broker.result(op_id)
    assert result.status is OpStatus.FAILED and result.error.code == "render_needs_job"
    assert [j.job_id for j in world.project.render_jobs] == ["rj-human"]


# == P5: node wiring ==============================================================================


def test_p5_one_branch_registry_is_shared_by_the_runtime_runner_and_ops(tmp_path: Path) -> None:
    world = _human_world(tmp_path)
    rt = world.runtime
    assert rt.runner.branch_registry is rt.branch_registry is world.branch_registry
    job = {"steps": [{"op": "timeline.branch", "params": {"source": world.ref("Main")}}]}
    job_id = world.submit("ops.sequence", job, world.locator())
    world.runner.run_until_settled()
    branch = world.runner.get(job_id).results[0]["timeline"]["unique_id"]
    assert rt.branch_registry.owned_by(world.pid(), branch, AGENT_ID)


def test_p5_the_feed_and_the_lease_table_see_each_other_on_one_clock(tmp_path: Path) -> None:
    world = _human_world(tmp_path)
    rt = world.runtime
    lease = rt.leases.acquire(
        Holder.agent(AGENT_ID),
        __import__("aar.node.leases", fromlist=["Scope"]).Scope.PROJECT,
        (world.pid(),),
        60.0,
    )
    assert [x.id for x in rt.feed._lease_source()] == [lease.id]  # lease_source = table.active
    assert rt.feed._clock is rt.leases._clock is world.clock  # one clock
    world.runner.pump()  # the pump flushes: the grant reaches the hub as this node
    granted = world.log.read(types=["lease.granted"])
    assert granted and all(e.actor == Actor("node", world.node_id) for e in granted)
    assert rt.feed.reconciled  # the first flush reconciled (lease_source was wired)


def test_p5_every_pump_syncs_presence_then_flushes_the_feed(tmp_path: Path) -> None:
    world = _human_world(tmp_path)
    rt = world.runtime
    calls: list[str] = []
    real_sync, real_flush = rt.presence_reporter.sync, rt.feed.flush
    rt.presence_reporter.sync = lambda now, wall: (calls.append("sync"), real_sync(now, wall))[1]  # type: ignore[method-assign]
    rt.feed.flush = lambda: (calls.append("flush"), real_flush())[1]  # type: ignore[method-assign]
    for _ in range(3):
        world.runner.pump()
    assert calls == ["sync", "flush"] * 3
    # A human starts working: the next pump puts them on the board.
    world.project.current_timeline = world.project.timeline("Main [agent]")
    world.presence.observe(world.gw.get_context(), world.clock())
    world.runner.pump()
    changed = world.log.read(types=["presence.changed"])
    assert changed and changed[-1].actor == Actor("node", world.node_id)
    assert changed[-1].payload["active"] is True


def _register(world: JobWorld, *, sources: list[ProjectSource] | None = None, name: str = "Show"):
    return Registry(world.log).create_project(
        name,
        HUMAN,
        sources=sources
        if sources is not None
        else [ProjectSource("local_db", {"name": "Show", "uid": world.pid()})],
    )


def _branch_job(world: JobWorld) -> dict:
    return {"steps": [{"op": "timeline.branch", "params": {"source": world.ref("Main")}}]}


def test_p5_a_branch_a_job_cuts_is_recorded_on_the_hub_as_the_node(tmp_path: Path) -> None:
    world = _human_world(tmp_path)
    show = _register(world)
    job_id = world.submit("ops.sequence", _branch_job(world), world.locator())
    world.runner.run_until_settled()
    view = world.runner.get(job_id)
    result = view.results[0]
    records = branches(world.log, show.id)
    assert len(records) == 1
    rec = records[0]
    assert rec.project_id == show.id  # the hub's prj_ id, not the Resolve uid
    assert rec.recorded_by == Actor("node", world.node_id) == Actor("node", world.feed.node_id)
    assert rec.agent_id == AGENT_ID and rec.version == result["version"]
    assert rec.branch_timeline_id == result["timeline"]["unique_id"]
    assert rec.source_timeline_id == world.main.GetUniqueId()
    assert [e["kind"] for e in view.events].count("branch_recorded") == 1
    # A second job reusing the branch records nothing new on the hub (idempotent).
    world.submit("ops.sequence", _branch_job(world), world.locator())
    world.runner.run_until_settled()
    assert len(world.log.read(types=["collab.branch_recorded"])) == 1


@pytest.mark.parametrize("registry", ["none", "two"])
def test_p5_a_branch_with_no_single_hub_project_is_alerted_not_lost(
    tmp_path: Path, registry: str
) -> None:
    world = _human_world(tmp_path)
    if registry == "two":
        _register(world, sources=[ProjectSource("cloud_library", {"library": "A", "name": "Show"})])
        _register(world, sources=[ProjectSource("cloud_library", {"library": "B", "name": "Show"})])
    job_id = world.submit("ops.sequence", _branch_job(world), world.locator())
    world.runner.run_until_settled()
    view = world.runner.get(job_id)
    assert view.state == "succeeded"  # the branch exists in Resolve; the record is the alert
    assert [a["code"] for a in view.alerts] == ["branch_not_recorded"]
    assert world.log.read(types=["collab.branch_recorded"]) == []


def test_p5_a_branch_record_the_hub_could_not_take_is_sent_again_after_a_restart(
    tmp_path: Path,
) -> None:
    world = _human_world(tmp_path)
    show = _register(world)
    real = LocalHubLink.record_branch

    def down(self, *a, **k):
        raise HubUnreachable("hub down")

    LocalHubLink.record_branch = down  # type: ignore[method-assign]
    try:
        job_id = world.submit("ops.sequence", _branch_job(world), world.locator())
        world.runner.run_until_settled()
        for _ in range(5):
            world.runner.pump()
    finally:
        LocalHubLink.record_branch = real  # type: ignore[method-assign]
    view = world.runner.get(job_id)
    assert [e["kind"] for e in view.events].count("branch_record_pending") == 1  # bounded
    assert branches(world.log, show.id) == []
    runner = world.restart()  # the node restarts before the hub is back up
    runner.pump()
    assert len(branches(world.log, show.id)) == 1
    assert "branch_recorded" in [e["kind"] for e in runner.get(job_id).events]


def test_p5_one_live_runtime_per_node_id(tmp_path: Path) -> None:
    world = _human_world(tmp_path)
    other_store = JobStore(tmp_path / "other.db")
    try:
        with pytest.raises(NodeIdInUse):
            NodeRuntime.local(world.gw, other_store, world.log, node_id=world.node_id)
        world.runtime.close()
        again = NodeRuntime.local(world.gw, other_store, world.log, node_id=world.node_id)
        again.close()
    finally:
        other_store.close()


def test_p5_the_runner_refuses_a_feed_for_another_node(tmp_path: Path) -> None:
    from aar.node.jobs import JobRunner

    world = _human_world(tmp_path)
    with pytest.raises(ValueError, match="node"):
        JobRunner(
            world.broker,
            world.store,
            world.runtime.hublink,
            node_id="another-node",
            branch_registry=world.branch_registry,
            feed=world.feed,
            presence_reporter=world.runtime.presence_reporter,
        )


# == item 8: the hub's error contract on the node side ============================================


def test_item8_an_unreachable_hub_pauses_the_job_fail_closed_then_it_carries_on(
    tmp_path: Path,
) -> None:
    world = JobWorld(tmp_path)  # owned timelines: only the hub link matters here
    real = LocalHubLink.assert_running
    LocalHubLink.assert_running = lambda self, **k: (_ for _ in ()).throw(  # type: ignore[method-assign]
        ConnectionError("no route to hub")
    )
    try:
        job_id = world.submit("ops.sequence", _marker_job(100), world.locator())
        for _ in range(5):
            world.runner.pump()
        view = world.runner.get(job_id)
        assert view.state == "waiting" and view.waiting["kind"] == "hub_unreachable"
        assert world.main.markers == {} and world.broker.pending() == []
    finally:
        LocalHubLink.assert_running = real  # type: ignore[method-assign]
    world.runner.run_until_settled()
    assert world.runner.get(job_id).state == "succeeded" and sorted(world.main.markers) == [100]


def test_item8_an_unreachable_consume_burns_nothing_and_a_lost_answer_is_recovered(
    tmp_path: Path,
) -> None:
    """The consume commits on the hub but its answer is lost (TimeoutError): the job neither
    marks the token refused nor consumes again; it recovers the consume and runs once."""
    world = _human_world(tmp_path)
    job_id = world.submit("ops.sequence", _marker_job(100), world.locator())
    world.runner.run_until_settled()
    world.runner.approve(job_id, _op_grant(world, "markers.add"), HUMAN)
    real = LocalHubLink.consume_approval
    state = {"lost": 0}

    def lost_answer(self, *a, **k):
        real(self, *a, **k)  # the consume commits on the hub ...
        state["lost"] += 1
        raise TimeoutError("the hub did not answer in time")

    LocalHubLink.consume_approval = lost_answer  # type: ignore[method-assign]
    try:
        world.runner.pump()
        view = world.runner.get(job_id)
        assert view.state == "waiting" and view.waiting["kind"] == "hub_unreachable"
        assert "refused_token" not in view.waiting["detail"]
    finally:
        LocalHubLink.consume_approval = real  # type: ignore[method-assign]
    world.runner.run_until_settled()
    assert world.runner.get(job_id).state == "succeeded" and sorted(world.main.markers) == [100]
    assert state["lost"] == 1 and len(world.log.read(types=["approval.consumed"])) == 1


def test_item8_the_local_link_reports_a_locked_hub_as_unreachable(tmp_path: Path) -> None:
    """A locked hub database is ``HubUnreachable`` (an ``OSError``); a refusal stays a
    ``ValueError`` (an unknown hub project here)."""
    import aar.node.hublink as hublink_module

    world = _human_world(tmp_path)
    link = LocalHubLink(world.log)

    def locked(*a, **k):
        raise sqlite3.OperationalError("database is locked")

    for name in ("assert_running", "get_approval", "consume", "branches"):
        real = getattr(hublink_module, name)
        setattr(hublink_module, name, locked)
        try:
            with pytest.raises(HubUnreachable):
                if name == "assert_running":
                    link.assert_running(project=world.locator(), node_id=world.node_id)
                elif name == "get_approval":
                    link.approval("apv_x")
                elif name == "consume":
                    link.consume_approval("apv_x", "project.write", "t", Actor("agent", "a"))
                else:
                    _register(world)
                    link.record_branch(
                        world.locator(),
                        Actor("node", world.node_id),
                        source_timeline_id="a",
                        source_timeline_name="Main",
                        agent_id=AGENT_ID,
                        branch_timeline_id="b",
                        branch_timeline_name="Main [agent:test-agent v1]",
                        version=1,
                    )
        finally:
            setattr(hublink_module, name, real)
    assert issubclass(HubUnreachable, OSError) and issubclass(UnknownHubProject, ValueError)
    with pytest.raises(ValueError):
        link.record_branch(
            world.locator("Other"),  # registered nowhere: refused, not unreachable
            Actor("node", world.node_id),
            source_timeline_id="a",
            source_timeline_name="Main",
            agent_id=AGENT_ID,
            branch_timeline_id="b",
            branch_timeline_name="Main [agent:test-agent v1]",
            version=1,
        )


# == the coordinator's addendum ===================================================================


def test_the_runner_submits_only_through_the_framework() -> None:
    """Addendum §2: the runner never calls a private op entry point (``_execute``,
    ``_branch_refusal``, ...), so the guard, leases and audit always apply."""
    tree = ast.parse((SRC / "aar/node/jobs/runner.py").read_text())
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module
        and node.module.startswith("aar.node.ops")
        for alias in node.names
    }
    assert not {name for name in imported if name.startswith("_")}
    assert "make_hot_op" in imported
    names = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert "_execute" not in names and "_branch_refusal" not in names


def test_a_production_project_still_needs_its_f15_approval_for_the_agents_own_branch(
    tmp_path: Path,
) -> None:
    """P1 adds to F15, it does not replace it: on a production project an agent's own branch
    edit still waits for the project.write approval on the project."""
    world = _human_world(tmp_path, approval=ApprovalPolicy(live=True))
    world.own("Main [agent]")
    job_id = world.submit(
        "ops.sequence", _marker_job(7, timeline=world.ref("Main [agent]")), world.locator()
    )
    world.runner.run_until_settled()
    view = world.runner.get(job_id)
    assert view.state == "waiting" and view.waiting["detail"]["target"] == world.pid()
    token = grant(world.log, ApprovalScope.PROJECT_WRITE, world.pid(), 3600, "ok", HUMAN)
    world.runner.approve(job_id, token.token_id, HUMAN)
    world.runner.run_until_settled()
    assert world.runner.get(job_id).state == "succeeded"


def test_a_production_override_counts_as_the_f15_approval(tmp_path: Path) -> None:
    """On a production project a guarded op needs ONE human approval, the one naming it."""
    world = _human_world(tmp_path, approval=ApprovalPolicy(live=True))
    job_id = world.submit("ops.sequence", _marker_job(100), world.locator())
    world.runner.run_until_settled()
    world.runner.approve(job_id, _op_grant(world, "markers.add"), HUMAN)
    world.runner.run_until_settled()
    assert world.runner.get(job_id).state == "succeeded"
    assert len(world.log.read(types=["approval.consumed"])) == 1


def test_a_stop_still_pauses_a_job_waiting_on_the_guard(tmp_path: Path) -> None:
    world = _human_world(tmp_path)
    job_id = world.submit("ops.sequence", _marker_job(100), world.locator())
    world.runner.run_until_settled()
    control.stop(world.log, control.StopScope.all(), "stop", HUMAN)
    world.runner.approve(job_id, _op_grant(world, "markers.add"), HUMAN)
    for _ in range(3):
        world.runner.pump()
    assert world.runner.get(job_id).waiting["kind"] == "stop"
    assert world.main.markers == {} and world.log.read(types=["approval.consumed"]) == []
    control.resume(world.log, control.StopScope.all(), "go", HUMAN)
    world.runner.run_until_settled()
    assert sorted(world.main.markers) == [100]


def test_production_broker_config_is_honoured_by_the_runtime(tmp_path: Path) -> None:
    world = _human_world(tmp_path, broker_config=BrokerConfig(restore_delay_seconds=3.0))
    assert world.broker.config.restore_delay_seconds == 3.0


def test_a_hung_hub_check_does_not_block_reads(tmp_path: Path) -> None:
    """The runner reads stay available while a pump waits on the hub (no lock held)."""
    world = JobWorld(tmp_path, drain_budget=0.5)
    job_id = world.submit("ops.sequence", _marker_job(100), world.locator())
    got: list = []
    t = threading.Thread(target=lambda: got.append(world.runner.get(job_id)))
    t.start()
    t.join(30)
    assert got and got[0].job_id == job_id
