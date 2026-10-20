"""S7 integration addendum §3 (the S7 round-3 validator's LOW-1..4) and §5 (the S11f round-2
validator's two LOWs). The §3 tests fail on round-3 a6e8150; the §5 tests on S11f's dev tree.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from aar.node.ops.write import POSSIBLE_COPIES

from .fake_resolve import FakeRenderJob
from .jobs_support import JobWorld
from .ops_support import AGENT, Show

WAIT = 30.0


class _Crash(BaseException):
    """A process death inside a Resolve call: no ``except Exception`` clean-up runs."""


def _render_params(world: JobWorld, **over: object) -> dict:
    params = {
        "timeline": world.ref("Main"),
        "target_dir": "D:/Out",
        "custom_name": "show_v1",
        "mark_in": 0,
        "mark_out": 99,
    }
    params.update(over)
    return params


def _calls(world: JobWorld, name: str) -> list:
    return [c for c in world.fake.calls if c[0] == name]


def _human_job(world: JobWorld, job_id: str = "rj-human-1", status: str = "Ready") -> str:
    info = {"JobId": job_id, "TimelineName": "Main", "TargetDir": "E:/Human"}
    world.project.render_jobs.append(FakeRenderJob(job_id, info, status=status))
    return job_id


# == §3 LOW-1: the queue snapshot before AddRenderJob ============================================


def test_low1_a_crash_inside_add_render_job_that_added_nothing_resumes_cleanly(
    tmp_path: Path,
) -> None:
    """A human job is already queued. The node dies INSIDE AddRenderJob before Resolve added
    anything. The journaled queue snapshot (``pre_ids``, which holds the human's job) shows
    nothing new, so the resumed build adds its job and renders, instead of declaring the
    human's job an ambiguous candidate (mutant ``pre_ids = []``)."""
    world = JobWorld(tmp_path)
    human = _human_job(world)
    job_id = world.submit("render", _render_params(world), world.locator())
    world.runner.pump()
    real_add = world.project.AddRenderJob

    def die_before_add() -> str:
        raise _Crash("died inside AddRenderJob before Resolve added anything")

    world.project.AddRenderJob = die_before_add  # type: ignore[method-assign]
    with pytest.raises(_Crash):
        world.broker.run_until_idle()
    world.project.AddRenderJob = real_add  # type: ignore[method-assign]
    assert world.store.get(job_id).cursor["journal"]["pre_ids"] == [human]
    runner = world.restart()
    runner.run_until_settled()
    view = runner.get(job_id)
    assert view.state == "succeeded", view.error
    assert view.alerts == []
    assert [j.job_id for j in world.project.render_jobs] == [human]  # ours cleaned up after
    assert len(_calls(world, "AddRenderJob")) == 1


# == §3 LOW-2: render.delete_job's ``deleted`` flag is observable ================================


def test_low2_the_render_reports_whether_cleanup_deleted_its_job(tmp_path: Path) -> None:
    world = JobWorld(tmp_path)
    job_id = world.submit("render", _render_params(world), world.locator())
    world.runner.run_until_settled()
    view = world.runner.get(job_id)
    assert view.measured["cleanup"] == "deleted" and world.project.render_jobs == []


def test_low2_a_job_a_human_already_removed_is_reported_as_already_gone(tmp_path: Path) -> None:
    world = JobWorld(tmp_path)
    world.project.render_complete_after = 3
    real_status = world.project.GetRenderJobStatus

    def complete_then_removed(job: str) -> dict[str, Any]:
        out = real_status(job)
        if out.get("JobStatus") == "Complete":
            world.project.render_jobs = [j for j in world.project.render_jobs if j.job_id != job]
        return out

    world.project.GetRenderJobStatus = complete_then_removed  # type: ignore[method-assign]
    job_id = world.submit("render", _render_params(world), world.locator())
    world.runner.run_until_settled()
    view = world.runner.get(job_id)
    assert view.state == "succeeded"
    assert view.measured["cleanup"] == "already_gone"
    assert _calls(world, "DeleteRenderJob") == []


# == §3 LOW-3: a JobId listed twice is not trusted ================================================


def test_low3_a_render_job_listed_twice_is_refused_and_no_human_job_is_touched(
    tmp_path: Path,
) -> None:
    world = JobWorld(tmp_path)
    human = _human_job(world)
    real_list = world.project.GetRenderJobList

    def doubled() -> list[dict[str, Any]]:
        jobs = real_list()
        ours = [j for j in jobs if j["JobId"] != human]
        return jobs + ours  # Resolve reports our new JobId twice

    world.project.GetRenderJobList = doubled  # type: ignore[method-assign]
    job_id = world.submit("render", _render_params(world), world.locator())
    world.runner.run_until_settled()
    view = world.runner.get(job_id)
    assert view.state == "failed" and view.error["code"] == "add_render_job_unverified"
    assert "2 times" in view.error["message"]
    assert _calls(world, "StartRendering") == []
    assert ("DeleteRenderJob", human) not in world.fake.calls
    assert human in [j.job_id for j in world.project.render_jobs]


# == §3 LOW-4: restarts during a healthy long render are not a crash loop ========================


def _restart_during_render(world: JobWorld, job_id: str, restarts: int) -> list[str]:
    states = []
    for _ in range(restarts):
        runner = world.restart()
        runner.pump()  # the poll after the restart
        runner.pump()
        states.append(runner.get(job_id).state)
    return states


def test_low4_restarts_while_the_render_keeps_progressing_never_stall_it(tmp_path: Path) -> None:
    world = JobWorld(tmp_path)
    world.project.render_complete_after = 100  # completion creeps up 1% on every poll
    job_id = world.submit("render", _render_params(world), world.locator())
    for _ in range(6):
        world.runner.pump()
    assert world.store.get(job_id).cursor.get("phase") == "poll"
    states = _restart_during_render(world, job_id, 8)
    assert "stalled" not in states, states
    assert world.store.get(job_id).restarts == 0
    kinds = [e["kind"] for e in world.runner.get(job_id).events]
    assert kinds.count("progress_after_restart") == 8  # one event per forgiven restart


def test_low4_restarts_with_no_render_progress_still_stall_it(tmp_path: Path) -> None:
    world = JobWorld(tmp_path)
    world.project.render_complete_after = 1000
    job_id = world.submit("render", _render_params(world), world.locator())
    for _ in range(6):
        world.runner.pump()
    rj = world.project.render_jobs[0]
    real_status = world.project.GetRenderJobStatus

    def stuck(job: str) -> dict[str, Any]:  # still "Rendering", never moves on
        return {"JobStatus": "Rendering", "CompletionPercentage": 42}

    world.project.GetRenderJobStatus = stuck  # type: ignore[method-assign]
    world.runner.pump()
    world.runner.pump()  # the stuck mark is recorded once
    states = _restart_during_render(world, job_id, 8)
    world.project.GetRenderJobStatus = real_status  # type: ignore[method-assign]
    assert "stalled" in states, states
    view = world.runner.get(job_id)
    assert view.stall["code"] == "crash_loop"
    assert rj.status == "Rendering"  # the watchdog reports, never kills


# == §5 S11f r2 LOW: a programming error after the copy is not "transient" ========================


def test_s11f_low_a_non_gateway_error_after_the_copy_keeps_its_traceback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aar.node.ops import write

    show = Show()

    def broken(*a: Any, **k: Any) -> Any:
        raise TypeError("unsupported operand type(s)")

    monkeypatch.setattr(write, "_set_resolution", broken)
    error = show.fails(
        "timeline.duplicate",
        {
            "source": show.ref("Main"),
            "name": "Main 9x16",
            "resolution": {"width": 1080, "height": 1920},
        },
    )
    assert error.code == "op_raised"  # never the transient-sounding resolve_call_failed
    assert error.detail["exception"] == "TypeError"
    assert "unsupported operand" in error.detail["traceback"]
    assert "Traceback" in error.detail["traceback"]
    assert "do not retry" in error.detail["fix_hint"]
    assert error.detail["created_timeline_id"]  # the copy is still named for the undo


def test_s11f_low_the_runner_never_retries_a_failed_op(tmp_path: Path, monkeypatch) -> None:
    """A job whose op fails (op_raised, or resolve_call_failed) fails: the op is submitted
    exactly once, never blind-retried."""
    from aar.node.ops import framework

    world = JobWorld(tmp_path)
    real = framework.OpEnv.timelines

    def flaky(self: Any) -> Any:
        raise RuntimeError("Resolve went away mid-call")

    job = {
        "steps": [
            {
                "op": "markers.mirror_review",
                "params": {
                    "timeline": world.ref("Main"),
                    "review_id": "rev1",
                    "notes": [{"note_id": "n1", "frame": 10, "text": "tighten"}],
                },
            }
        ]
    }
    monkeypatch.setattr(framework.OpEnv, "timelines", flaky)
    job_id = world.submit("ops.sequence", job, world.locator())
    world.runner.run_until_settled()
    monkeypatch.setattr(framework.OpEnv, "timelines", real)
    view = world.runner.get(job_id)
    assert view.state == "failed"
    submitted = [e for e in view.events if e["kind"] == "op_submitted"]
    assert len(submitted) == 1, submitted
    for _ in range(5):
        world.runner.pump()
    assert len([e for e in world.runner.get(job_id).events if e["kind"] == "op_submitted"]) == 1


# == §5 S11f r2 LOW: a copy Resolve made but did not hand back is named, never deleted ===========


@pytest.mark.parametrize("returns", ["none", "source"])
def test_s11f_low_an_unclaimed_copy_is_named_and_never_deleted(
    monkeypatch: pytest.MonkeyPatch, returns: str
) -> None:
    show = Show()
    timeline_cls = type(show.main)
    real = timeline_cls.DuplicateTimeline

    def made_but_lost(self: Any, name: str) -> Any:
        real(self, name)
        return None if returns == "none" else self  # Resolve made it, then lied about it

    monkeypatch.setattr(timeline_cls, "DuplicateTimeline", made_but_lost)
    before = {t.GetUniqueId() for t in show.project.timelines}
    error = show.fails("timeline.duplicate", {"source": show.ref("Main"), "name": "Main copy"})
    assert error.code in ("duplicate_failed", "duplicate_unverified")
    left = error.detail[POSSIBLE_COPIES]
    new = [t for t in show.project.timelines if t.GetUniqueId() not in before]
    assert left == [{"name": "Main copy", "unique_id": new[0].GetUniqueId()}]
    assert "created_timeline_id" not in error.detail  # not provably ours: never undone
    assert [c for c in show.fake.calls if c[0] == "DeleteTimelines"] == []
    assert any(t.GetName() == "Main copy" for t in show.project.timelines)  # still there


def test_s11f_low_a_branch_whose_copy_was_lost_names_it_and_deletes_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    show = Show()
    timeline_cls = type(show.main)
    real = timeline_cls.DuplicateTimeline
    monkeypatch.setattr(
        timeline_cls, "DuplicateTimeline", lambda self, name: (real(self, name), None)[1]
    )
    error = show.fails("timeline.branch", {"source": show.ref("Main")})
    assert error.code == "duplicate_failed"
    assert error.detail[POSSIBLE_COPIES][0]["name"] == f"Main [agent:{AGENT.id} v1]"
    assert [c for c in show.fake.calls if c[0] == "DeleteTimelines"] == []


def test_s11f_low_only_same_named_new_timelines_are_possible_copies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A new timeline under ANOTHER name (a human making one meanwhile) is not a possible
    copy: only a new one carrying the requested name is named (mutant S04)."""
    from .fake_resolve import FakeTimeline

    show = Show()
    timeline_cls = type(show.main)
    real = timeline_cls.DuplicateTimeline

    def made_lost_and_a_human_timeline_appeared(self: Any, name: str) -> Any:
        real(self, name)
        show.project.timelines.append(
            FakeTimeline(show.fake, "Human cut", "tl-HUMAN", show.project, start_frame=0)
        )
        return None

    monkeypatch.setattr(timeline_cls, "DuplicateTimeline", made_lost_and_a_human_timeline_appeared)
    error = show.fails("timeline.duplicate", {"source": show.ref("Main"), "name": "Main copy"})
    names = [c["name"] for c in error.detail[POSSIBLE_COPIES]]
    assert names == ["Main copy"]
