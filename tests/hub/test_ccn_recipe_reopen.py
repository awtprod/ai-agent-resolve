"""``ccn-longform`` after a render the cut made fail (S19f), end to end and offline: a cut whose
review render fails QA, the reopen run exactly as the guide prints it, the same plan refused, a
changed cut, render, export, review, approve and the report. Plus the classification table of
:mod:`aar.recipes.ccn.faults`, code by code, through the step's own conversion."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from ccn_recipe_support import (
    GUIDE_BLOCK,
    CcnEnv,
    agent_cut,
    latest,
    make_run,
    outputs,
    read_guide,
    sh,
    start,
    work,
)
from write_support import AGENT, HUMAN

from aar.hub.reviews import decide
from aar.hub.writes import WriteError
from aar.plan.export import ExportError
from aar.plan.export.errors import EXPORT_CODES
from aar.plan.render import RenderError
from aar.plan.render.errors import RENDER_ONLY_CODES
from aar.recipes.ccn import CCN_LONGFORM
from aar.recipes.ccn.faults import PLAN_LAYOUT_CODES, RENDER_PLAN_CODES, upstream_failure
from aar.recipes.ccn.steps import _library_failure
from aar.recipes.context import StepFailure
from aar.recipes.report import build_report_text
from aar.recipes.run import step_complete
from aar.recipes.state import load_state
from aar.recipes.status import status_view
from ccn_support import WORDS, needs_ffmpeg

#: Only the third sentence is heard; the first is digital silence (the transcript still has
#: every word), so a cut of the first sentence fails QA on loudness and one of the third passes.
SILENT_FIRST = {"ranges": [{"start_s": 0.9, "end_s": 4.1}]}
HEARD_LAST = {"ranges": [{"start_s": 8.9, "end_s": 11.1}]}


@needs_ffmpeg
def test_a_cut_that_fails_qa_is_reopened_changed_and_shipped(ccn_env: CcnEnv) -> None:
    env = ccn_env
    show = env.show("cam-a.mov", words=WORDS[13:], room=False)
    run = make_run(env, env.params(show), key="reopen")
    start(env, run, "reopen")
    assert work(env) == 5
    first_plan = agent_cut(env, run, keep=SILENT_FIRST, key="r1a1")
    work(env)
    render = latest(env, run, "render")
    assert render.status == "failed" and render.result_error["code"] == "qa_failed"
    assert render.upstream and "loudness" in render.result_error["message"]
    assert latest(env, run, "export") is None

    # The status view names the reopen; the agent runs the guide's own command for it.
    view = status_view(env.hub.log, run)
    offer = next(a for a in view["next_actions"] if a["step_id"] == "render")
    assert offer["tool"] == "recipe_reopen" and offer["args"]["step_id"] == "cut"
    assert offer["what"].startswith("recipe_reopen cut")
    guide = read_guide(env, run)
    reopen_line = GUIDE_BLOCK.findall(guide)[-1].strip()
    assert " recipe reopen " in reopen_line and reopen_line.endswith(" cut")
    assert "3 attempts in a round" in re.sub(r"\s+", " ", guide)
    reopen_line = reopen_line.replace("<you>", "agent-1").replace("<new key>", "reopen-1")
    body = json.loads(sh(env, reopen_line))
    assert body["op"] == "recipe.reopen" and body["round"] == 1
    cut = next(s for s in body["steps"] if s["step_id"] == "cut")
    assert cut["state"] == "needs_agent" and cut["attempt"] == 2
    assert cut["reopened_for"]["step"] == "render" and cut["reopened_for"]["attempt"] == 1
    assert cut["reopened_for"]["code"] == "qa_failed"
    assert cut["reopened_for"]["op_node"] == render.op_id

    # The same plan again is refused; a changed one goes through render, export and review.
    with pytest.raises(WriteError) as same:
        step_complete(
            env.hub.log,
            env.hub.store,
            run,
            "cut",
            None,
            {"plan_node": first_plan},
            AGENT,
            idempotency_key="cut-same",
        )
    assert same.value.code == "plan_unchanged"
    second_plan = agent_cut(env, run, keep=HEARD_LAST, key="r1a2")
    assert work(env) == 2  # render, export
    rendered = latest(env, run, "render")
    assert rendered.attempt == 2 and rendered.status == "done"
    assert outputs(env, run, "render")["subject_node"] == second_plan
    copy = outputs(env, run, "export")["copy"]
    assert copy["round"] == 1 and copy["fcpxml"].endswith("Synthetic Long v1.r1.fcpxml")
    review = latest(env, run, "review")
    assert review.status == "active" and review.round == 1
    decide(env.hub.log, env.hub.project_id, review.review_id, "approve", "ship it", HUMAN)
    assert work(env) == 1
    state = load_state(env.hub.log, run)
    assert state.recipe_status == "done" and state.current_round() == 1
    report = build_report_text(env.hub.log, run)
    assert "- review rounds: used 1 of" in report
    assert "- fix attempts: round 1: 2" in report.split("### cut (agent)")[1].split("###")[0]
    assert len(env.calls) == 1  # the transcript was made once


@needs_ffmpeg
def test_the_cut_has_three_attempts_then_the_status_names_cancel(ccn_env: CcnEnv) -> None:
    """Every attempt of the cut in a round counts, the reopened ones too: after the third cut
    fails QA the status says the cut has no attempts left and that what remains is a cancel and a
    new run; recipe_reopen is refused no_attempts_left."""
    from aar.recipes.run import recipe_reopen

    env = ccn_env
    show = env.show("cam-a.mov", words=WORDS[13:], room=False)
    run = make_run(env, env.params(show), key="three")
    start(env, run, "three")
    work(env)
    silent = [
        {"ranges": [{"start_s": 0.9, "end_s": 4.1}]},
        {"ranges": [{"start_s": 0.9, "end_s": 3.3}]},
        {"ranges": [{"start_s": 0.9, "end_s": 2.1}]},
    ]
    for n, keep in enumerate(silent, start=1):
        agent_cut(env, run, keep=keep, key=f"a{n}")
        work(env)
        assert latest(env, run, "render").result_error["code"] == "qa_failed"
        if n < 3:
            recipe_reopen(env.hub.log, run, "cut", AGENT, idempotency_key=f"re{n}")
    with pytest.raises(WriteError) as refused:
        recipe_reopen(env.hub.log, run, "cut", AGENT, idempotency_key="re3")
    assert refused.value.code == "no_attempts_left"
    offer = next(
        a for a in status_view(env.hub.log, run)["next_actions"] if a["step_id"] == "render"
    )
    assert offer["who"] == "human" and "tool" not in offer
    assert "no attempts left (3 of 3" in offer["what"] and "recipe_cancel" in offer["what"]


# -- the classification table ------------------------------------------------------------------


def test_the_ccn_render_and_export_declare_on_fail_cut() -> None:
    assert CCN_LONGFORM.step("render").on_fail == "cut"
    assert CCN_LONGFORM.step("export").on_fail == "cut"
    assert all(s.on_fail is None for s in CCN_LONGFORM.steps if s.id not in ("render", "export"))


@pytest.mark.parametrize("code", sorted(PLAN_LAYOUT_CODES))
def test_every_plan_layout_code_is_marked_at_render_and_export(code: str) -> None:
    assert _library_failure("render", RenderError(code, "m", "h")).upstream is True
    failure = _library_failure("export", ExportError(code, "m", "h"))
    assert failure.upstream is True and failure.code == f"export_{code}"
    assert "recipe_reopen cut" in failure.fix_hint and failure.fix_hint.startswith("h")
    assert failure.to_error()["upstream"] is True


@pytest.mark.parametrize("code", sorted(RENDER_PLAN_CODES))
def test_every_render_plan_code_is_marked(code: str) -> None:
    assert _library_failure("render", RenderError(code, "m", "h")).upstream is True


def test_render_too_large_is_marked_only_for_overlapping_pictures() -> None:
    def marked(**details: object) -> StepFailure:
        return _library_failure("render", RenderError("render_too_large", "m", "h", **details))

    assert marked(reason="pictures").upstream is True
    assert marked(reason="encoder_base").upstream is False  # the stub's resolution, not the cut
    assert marked().upstream is False  # the budget holds no sound of the mix
    assert "upstream" not in marked(reason="encoder_base").to_error()


UNMARKED_RENDER = sorted(RENDER_ONLY_CODES - RENDER_PLAN_CODES - {"render_too_large"})
UNMARKED_EXPORT = sorted(EXPORT_CODES - PLAN_LAYOUT_CODES)


@pytest.mark.parametrize("code", UNMARKED_RENDER)
def test_environment_render_codes_stay_unmarked(code: str) -> None:
    failure = _library_failure("render", RenderError(code, "m", "h"))
    assert failure.upstream is False and failure.fix_hint == "h"
    assert "upstream" not in failure.to_error()


@pytest.mark.parametrize("code", UNMARKED_EXPORT)
def test_environment_export_codes_stay_unmarked(code: str) -> None:
    assert _library_failure("export", ExportError(code, "m", "h")).upstream is False


def test_the_step_codes() -> None:
    assert upstream_failure("plan_not_from_stub") and upstream_failure("qa_failed")
    for code in (
        "media_changed",
        "invalid_params",
        "no_profile",
        "missing_input",
        "work_dir_unset",
        "export_copy_failed",
        "export_path_symlink",
        "timeout",
        "error",
        "lint_errors",  # unprefixed: only the steps' own render_/export_ codes count
    ):
        assert upstream_failure(code) is False, code


def test_the_unmarked_set_is_what_the_brief_lists() -> None:
    """The environment codes the brief names stay unmarked: media missing, changed or unreadable,
    a missing tool, timeouts, the export directory, params, the exporter's self-check."""
    for code in (
        "render_media_not_on_hub",
        "render_media_unreadable",
        "render_tool_missing",
        "render_render_timeout",
        "render_render_failed",
        "render_bad_output_path",
        "export_bad_media_map",
        "export_media_not_on_edit_pc",
        "export_self_check_failed",
        "export_output_exists",
    ):
        assert upstream_failure(code) is False, code


def test_the_design_doc_lists_every_marked_code() -> None:
    design = (Path(__file__).parents[2] / "docs" / "DESIGN.md").read_text(encoding="utf-8")
    section = design.split("Which render and export failures are the cut's")[1].split("\n\n**")[0]
    for code in sorted(PLAN_LAYOUT_CODES | RENDER_PLAN_CODES | {"plan_not_from_stub"}):
        assert f"`{code}`" in section, code
