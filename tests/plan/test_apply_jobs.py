"""``phase_result_from_job``: the trust boundary between a finished node job and the compiler
(round 4: N1 binding + by-name mapping, N2 assemble from a real phase-2 job, N3 completeness and
typed refusals, N4 through the builder, N5 whole-pool find arity, N7 shared files, R9 arity)."""

from __future__ import annotations

import contextlib
import copy
from typing import Any

import pytest
from apply_builders import longform, plan, seg, source
from apply_support import (
    BRANCH_UID,
    assemble_job,
    find_value,
    job_step,
    media_info,
    prepare_job,
    synth_assemble,
    synth_prepare,
    template_describe,
)

from aar.node.ops.framework import MediaRef
from aar.plan.apply import (
    AssembleResult,
    CompileError,
    PrepareResult,
    build_submission,
    compile_assemble,
    compile_finish,
    phase_result_from_job,
    plan_hash,
)
from aar.plan.model import EditPlan

LF = longform()
LF_TD = template_describe(10, name=LF.template)
TD = template_describe(5, name="Tpl")
B = "B001 multicam"
TITLE = {"kind": "title", "lines": ["x"], "record_in": 0, "record_out": 10, "template": "LT"}


def _pool_plan(
    order: tuple[str, ...] = ("a", "b", "c"), b_media: str = "B001 multicam"
) -> EditPlan:
    """One file source and two pool-name (multicam) sources, plus one title template."""
    srcs = {
        "a": source("a", media="E:/X/a.mov", camera="A"),
        "b": source("b", media=b_media, camera="B"),
        "c": source("c", media="C001 multicam", camera="C"),
    }
    return plan(
        sources=[srcs[k] for k in order],
        segments=[seg(src="a", a=0, b=24), seg(src="b", a=24, b=48), seg(src="c", a=48, b=72)],
        titles=[TITLE],
    )


P = _pool_plan()


def _prepare(job: Any, p: EditPlan = P, td=TD) -> PrepareResult:
    result = phase_result_from_job(job, p, "prepare", template=td)
    assert isinstance(result, PrepareResult)
    return result


def _prepare_code(job: Any, p: EditPlan = P, td=TD) -> str:
    with pytest.raises(CompileError) as ei:
        phase_result_from_job(job, p, "prepare", template=td)
    return ei.value.code


def _assemble_code(job: Any, prep: PrepareResult, p: EditPlan = P, **kw) -> str:
    with pytest.raises(CompileError) as ei:
        phase_result_from_job(job, p, "assemble", prepare=prep, **kw)
    return ei.value.code


def _mi(name: str, mid: str, bin_path: Any = ("Night 8", "CCN")) -> dict:
    return media_info(name, mid, bin_path=bin_path).model_dump(mode="json")


def _set_find(job: dict, name: str, items: list[dict], **kw: Any) -> dict:
    job = copy.deepcopy(job)
    value = find_value([], **kw)
    value["media"] = items
    value["scanned"] = len(items)
    job_step(job, name)["value"] = value
    return job


def _ids(prep: PrepareResult) -> dict[str, str]:
    return {s.source_id: s.media.unique_id for s in prep.sources}


# -- the honest path -----------------------------------------------------------------------------


def test_honest_prepare_maps_every_result_by_step_name() -> None:
    prep = _prepare(prepare_job(P, TD))
    assert prep.plan_sha256 == plan_hash(P)
    assert prep.job_id == "job_prepare_test"
    assert prep.branch.unique_id == BRANCH_UID
    assert _ids(prep) == {"a": "pool-a", "b": "pool-b", "c": "pool-c"}
    pool_b = next(s for s in prep.sources if s.source_id == "b")
    assert pool_b.media.bin_path == ("Night 8", "CCN")  # found anywhere in the pool (N5)
    assert [(t.template, t.media.unique_id) for t in prep.titles] == [("LT", "title-0")]


@pytest.mark.parametrize("which", ["pool", "ccn"])
def test_honest_jobs_build_all_three_phases(which: str) -> None:
    p, td = (P, TD) if which == "pool" else (LF, LF_TD)
    prep = _prepare(prepare_job(p, td), p, td)
    # N2: a real phase-2 job is track.ensure + appends; it has no timeline.branch step.
    job = assemble_job(p, prep)
    assert {s["op"] for s in job["steps"]} == {"track.ensure", "timeline.append_ranges"}
    asm = phase_result_from_job(job, p, "assemble", prepare=prep)
    assert isinstance(asm, AssembleResult)
    assert asm.branch == prep.branch
    assert asm.job_id == "job_assemble_test"
    finish = compile_finish(p, prepare=prep, assemble=asm)
    assert finish.unresolved == 0
    build_submission(p, "finish", finish.steps)  # submittable


# -- N1: the job is bound to this plan, phase and inputs; results map by name -------------------


def test_reorder_swap_probe_stale_job_for_a_reordered_plan_is_refused() -> None:
    # The validator's probe: a phase-1 job from plan v1 (pool sources b, c) wrapped for plan v2
    # that lists the two sources the other way round. r3 mapped by position: c -> uid-B, b -> uid-C.
    v1 = _pool_plan(("a", "b", "c"))
    v2 = _pool_plan(("a", "c", "b"))
    job = prepare_job(v1, TD)
    assert _ids(_prepare(job, v1)) == {"a": "pool-a", "b": "pool-b", "c": "pool-c"}
    assert _prepare_code(job, v2) == "job_not_from_plan"


def test_reordered_steps_never_swap_clips() -> None:
    # Even with its steps rearranged into v2's order, each result maps to its own source by step
    # name, so b keeps b's clip and c keeps c's.
    v1 = _pool_plan(("a", "b", "c"))
    v2 = _pool_plan(("a", "c", "b"))
    job = prepare_job(v1, TD)
    fb, fc = job_step(job, "find:source:b"), job_step(job, "find:source:c")
    i, j = job["steps"].index(fb), job["steps"].index(fc)
    job["steps"][i], job["steps"][j] = fc, fb
    assert _ids(_prepare(job, v2)) == {"a": "pool-a", "b": "pool-b", "c": "pool-c"}


def test_swapped_find_results_are_refused() -> None:
    job = prepare_job(P, TD)
    fb, fc = job_step(job, "find:source:b"), job_step(job, "find:source:c")
    fb["value"], fc["value"] = fc["value"], fb["value"]
    assert _prepare_code(job) == "forged_phase_result"


def test_swapped_step_names_are_refused() -> None:
    job = prepare_job(P, TD)
    fb, fc = job_step(job, "find:source:b"), job_step(job, "find:source:c")
    fb["name"], fc["name"] = fc["name"], fb["name"]
    assert _prepare_code(job) == "job_not_from_plan"


def test_job_of_a_changed_plan_is_refused() -> None:
    job = prepare_job(_pool_plan(b_media="B001 multicam"), TD)
    assert _prepare_code(job, _pool_plan(b_media="B002 multicam")) == "job_not_from_plan"


def test_job_of_another_template_describe_is_refused() -> None:
    job = prepare_job(P, TD)
    other = template_describe(5, name="Tpl", unique_id="tl-template-OTHER")
    assert _prepare_code(job, P, other) == "job_not_from_plan"


@pytest.mark.parametrize(
    "edit",
    ["extra_step", "duplicate_step", "renamed_step", "param_changed", "op_changed"],
)
def test_a_job_that_is_not_the_compiled_step_list_is_refused(edit: str) -> None:
    job = prepare_job(P, TD)
    steps = job["steps"]
    if edit == "extra_step":
        steps.append({**copy.deepcopy(steps[-1]), "name": "find:title:Other"})
    elif edit == "duplicate_step":
        steps.append(copy.deepcopy(steps[-1]))
    elif edit == "renamed_step":
        job_step(job, "find:source:b")["name"] = "find-source:0"
    elif edit == "param_changed":
        job_step(job, "find:source:b")["params"]["recursive"] = False
    else:
        job_step(job, "find:source:b")["op"] = "media.import"
    assert _prepare_code(job) == "job_not_from_plan"


def test_a_phase_1_job_is_not_a_phase_2_job() -> None:
    prep = _prepare(prepare_job(P, TD))
    assert _assemble_code(prepare_job(P, TD), prep) == "job_not_from_plan"


def test_an_assemble_job_with_a_branch_step_prepended_is_refused() -> None:
    prep = _prepare(prepare_job(P, TD))
    job = assemble_job(P, prep)
    job["steps"].insert(0, copy.deepcopy(job_step(prepare_job(P, TD), "branch")))
    assert _assemble_code(job, prep) == "job_not_from_plan"


def test_assemble_binds_the_batch_size() -> None:
    prep = _prepare(prepare_job(P, TD))
    job = assemble_job(P, prep, batch_size=2)
    assert _assemble_code(job, prep) == "job_not_from_plan"  # compiled at the default 10
    asm = phase_result_from_job(job, P, "assemble", prepare=prep, batch_size=2)
    assert len(asm.appends) == 2


# -- N1/N5/R9: each find result is re-checked; exactly one exact hit ----------------------------


@pytest.mark.parametrize(
    ("items", "code"),
    [
        ([], "source_not_found"),
        ([_mi(B, "u1"), _mi(B, "u2")], "source_ambiguous"),  # two hits, one bin
        ([_mi(B, "u1", ("Night 8", "CCN")), _mi(B, "u2", ("Night 9", "CCN"))], "source_ambiguous"),
        ([_mi(B, "u1", ()), _mi(B, "u2", ("aar", "B"))], "source_ambiguous"),
        ([_mi(B, "u1"), _mi(B, "u2"), _mi(B, "u3")], "source_ambiguous"),
        ([_mi("SOME OTHER CLIP", "u1")], "forged_phase_result"),
        ([_mi("b001 MULTICAM", "u1")], "forged_phase_result"),  # case differs
        ([_mi("B001 multicam ", "u1")], "forged_phase_result"),  # trailing space
        ([_mi("B001\u00a0multicam", "u1")], "forged_phase_result"),  # no-break space
        ([_mi(B, "u1"), _mi("B001 multicam (1)", "u2")], "forged_phase_result"),
        ([_mi(B, "u1", None)], "forged_phase_result"),  # no bin: scope unprovable
    ],
)
def test_pool_source_find_hits(items: list[dict], code: str) -> None:
    assert _prepare_code(_set_find(prepare_job(P, TD), "find:source:b", items)) == code


@pytest.mark.parametrize("bin_path", [(), ("Night 8",), ("aar", "B", "old")])
def test_pool_source_one_exact_hit_anywhere_in_the_pool_is_taken(bin_path: tuple) -> None:
    job = _set_find(prepare_job(P, TD), "find:source:b", [_mi(B, "the-one", bin_path)])
    prep = _prepare(job)
    pool_b = next(s for s in prep.sources if s.source_id == "b")
    assert pool_b.media == MediaRef(unique_id="the-one", bin_path=bin_path)


def test_a_truncated_find_is_ambiguous() -> None:
    job = _set_find(prepare_job(P, TD), "find:source:b", [_mi(B, "u1")], truncated=True)
    assert _prepare_code(job) == "source_ambiguous"


@pytest.mark.parametrize(
    "scope", [{"bin_path": ("aar", "B")}, {"recursive": False}, {"bin_path": ("Night 8",)}]
)
def test_a_find_result_of_another_scope_is_refused(scope: dict) -> None:
    job = _set_find(prepare_job(P, TD), "find:source:b", [_mi(B, "u1", ("Night 8", "x"))], **scope)
    assert _prepare_code(job) == "forged_phase_result"


@pytest.mark.parametrize(
    ("items", "code"),
    [
        ([], "title_template_missing"),
        ([_mi("LT", "t1"), _mi("LT", "t2", ("Titles",))], "title_template_ambiguous"),
        ([_mi("Not The Lower Third", "t1")], "forged_phase_result"),
        ([_mi("lt", "t1")], "forged_phase_result"),
    ],
)
def test_title_find_hits(items: list[dict], code: str) -> None:
    assert _prepare_code(_set_find(prepare_job(P, TD), "find:title:LT", items)) == code


def test_ccn_title_find_hits_are_checked_per_template() -> None:
    job = _set_find(prepare_job(LF, LF_TD), "find:title:CCN Sting", [_mi("CCN Lower Third", "x")])
    assert _prepare_code(job, LF, LF_TD) == "forged_phase_result"


# -- N4 through the builder ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("step", "media_id"),
    [
        ("find:source:b", B),  # its own pool name passed through as an id (W01)
        ("find:source:b", "b"),  # its source id
        ("find:source:b", "uid:1"),
        ("find:source:b", "E:/X/a.mov"),
        ("find:title:LT", "LT"),
        ("find:title:LT", "t\\1"),
    ],
)
def test_builder_refuses_a_forged_pool_id(step: str, media_id: str) -> None:
    job = prepare_job(P, TD)
    hit = job_step(job, step)["value"]["media"][0]
    hit["media_id"] = media_id
    assert _prepare_code(job) == "forged_phase_result"


def test_builder_refuses_an_import_id_that_is_a_path() -> None:
    job = prepare_job(P, TD)
    imp = next(s for s in job["steps"] if s["op"] == "media.import")
    imp["value"]["media"][0]["media"]["unique_id"] = "E:/X/a.mov"
    assert _prepare_code(job) == "forged_phase_result"


# -- imports: the result must be of the step's own request --------------------------------------


def _import_step(job: dict) -> dict:
    return next(s for s in job["steps"] if s["op"] == "media.import")


@pytest.mark.parametrize("edit", ["bin", "path", "extra", "none", "readback"])
def test_import_result_must_match_its_request(edit: str) -> None:
    job = prepare_job(P, TD)
    value = _import_step(job)["value"]
    entry = value["media"][0]
    if edit == "bin":
        value["bin_path"] = ["aar", "Z"]
    elif edit == "path":
        entry["path"] = entry["file_path"] = "E:/X/other.mov"
    elif edit == "extra":
        value["media"].append({**copy.deepcopy(entry), "path": "E:/X/z.mov"})
    elif edit == "none":
        value["media"] = []
    else:  # the requested path, but Resolve read back another file
        entry["file_path"] = "E:/Elsewhere/Z.mov"
    assert _prepare_code(job) == "forged_phase_result"


def test_import_paths_compare_by_path_key() -> None:
    job = prepare_job(P, TD)
    entry = _import_step(job)["value"]["media"][0]
    entry["path"] = entry["file_path"] = "e:\\x\\A.MOV"
    assert _ids(_prepare(job))["a"] == "pool-a"


# -- N7: two sources sharing a file share one import --------------------------------------------


def test_two_sources_sharing_a_file_map_to_one_pool_id() -> None:
    p = plan(
        sources=[
            source("a", media="E:/X/shared.mov", camera="A"),
            source("b", media="E:/X/SHARED.mov", camera="B"),
        ],
        segments=[seg(src="a", a=0, b=24), seg(src="b", a=24, b=48)],
    )
    td = template_describe(5, name=p.template)
    job = prepare_job(p, td)
    imports = [s for s in job["steps"] if s["op"] == "media.import"]
    assert [(s["params"]["bin_path"], s["params"]["paths"]) for s in imports] == [
        (["aar", "A"], ["E:/X/shared.mov"])
    ]
    assert _ids(_prepare(job, p, td)) == {"a": "pool-a", "b": "pool-a"}


# -- N3: the branch must be of the plan's template ----------------------------------------------


def test_a_branch_of_another_template_is_refused() -> None:
    job = prepare_job(P, TD)
    job_step(job, "branch")["value"]["source"]["unique_id"] = "some-other-template"
    assert _prepare_code(job) == "template_mismatch"


def test_a_branch_that_is_the_template_itself_is_refused() -> None:
    job = prepare_job(P, TD)
    job_step(job, "branch")["value"]["timeline"]["unique_id"] = TD.timeline.unique_id
    assert _prepare_code(job) == "forged_phase_result"


# -- N3: completeness -----------------------------------------------------------------------------


@pytest.mark.parametrize("drop", ["find:title:LT", "find:source:b", "branch"])
def test_a_missing_step_is_incomplete(drop: str) -> None:
    job = prepare_job(P, TD)
    job["steps"].remove(job_step(job, drop))
    assert _prepare_code(job) == "incomplete_job_result"


@pytest.mark.parametrize("name", ["branch", "import:aar/A:0", "find:source:c", "find:title:LT"])
@pytest.mark.parametrize("how", ["null", "absent"])
def test_a_step_without_a_result_is_incomplete(name: str, how: str) -> None:
    job = prepare_job(P, TD)
    step = job_step(job, name)
    if how == "null":
        step["value"] = None
    else:
        del step["value"]
    assert _prepare_code(job) == "incomplete_job_result"


def test_an_assemble_step_without_a_result_is_incomplete() -> None:
    prep = _prepare(prepare_job(P, TD))
    job = assemble_job(P, prep)
    job_step(job, "tracks")["value"] = None
    assert _assemble_code(job, prep) == "incomplete_job_result"


def test_assemble_drops_a_trailing_append_as_incomplete() -> None:
    prep = _prepare(prepare_job(P, TD))
    job = assemble_job(P, prep)
    job["steps"].pop()
    assert _assemble_code(job, prep) == "incomplete_job_result"


# -- N3: typed refusals for malformed input, never a bare exception -----------------------------


def _valid() -> dict:
    return prepare_job(P, TD)


@pytest.mark.parametrize(
    "bad",
    [
        [],
        None,
        "job",
        42,
        {"steps": _valid()["steps"]},  # no job_id
        {"job_id": "", "steps": _valid()["steps"]},
        {"job_id": "j", "steps": {"a": 1}},
        {"job_id": "j", "steps": []},
        {"job_id": "j", "steps": [{"name": "branch", "params": {}, "value": {}}]},  # no op
        {"job_id": "j", "steps": [{"op": "timeline.branch", "params": {}, "value": {}}]},
        {"job_id": "j", "steps": [{"op": "timeline.branch", "name": "branch", "value": {}}]},
        {"job_id": "j", "steps": [{"op": "o", "name": "n", "params": [], "value": {}}]},
        {"job_id": "j", "steps": [{"op": "o", "name": "n", "params": {}, "value": [1]}]},
        {"job_id": "j", "steps": [{"op": "o", "name": "n", "params": {}, "extra": 1}]},
        {**_valid(), "state": "done"},  # unknown top-level key
    ],
)
def test_malformed_job_shapes_are_typed(bad: Any) -> None:
    assert _prepare_code(bad) == "malformed_job_result"


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("branch", {}),
        ("branch", {"source": "x"}),
        ("find:source:b", {"media": "x"}),
        ("find:title:LT", {"bin_path": [], "recursive": True}),
        ("import:aar/A:0", {"bin_path": ["aar", "A"], "media": [], "notes": [], "x": 1}),
    ],
)
def test_a_result_that_is_not_its_ops_result_model_is_malformed(name: str, value: dict) -> None:
    job = prepare_job(P, TD)
    job_step(job, name)["value"] = value
    assert _prepare_code(job) == "malformed_job_result"


_JUNK = [None, [], {}, 1, "x", -1, True]


def _mutations(job: dict) -> list[dict]:
    """Every one-point corruption of a job: each key of each step, and each key of each step's
    result (one level down too), replaced by each junk value or deleted."""
    out: list[dict] = []
    for i, step in enumerate(job["steps"]):
        for key in list(step):
            for junk in [*_JUNK, KeyError]:
                bad = copy.deepcopy(job)
                if junk is KeyError:
                    del bad["steps"][i][key]
                else:
                    bad["steps"][i][key] = junk
                out.append(bad)
        value = step.get("value") or {}
        for key in list(value):
            for junk in [*_JUNK, KeyError]:
                bad = copy.deepcopy(job)
                if junk is KeyError:
                    del bad["steps"][i]["value"][key]
                else:
                    bad["steps"][i]["value"][key] = junk
                out.append(bad)
            inner = value[key]
            if isinstance(inner, list) and inner and isinstance(inner[0], dict):
                for sub in list(inner[0]):
                    bad = copy.deepcopy(job)
                    bad["steps"][i]["value"][key][0][sub] = None
                    out.append(bad)
    return out


def test_no_bare_exception_escapes_a_corrupted_prepare_job() -> None:
    for bad in _mutations(prepare_job(P, TD)):
        with contextlib.suppress(CompileError):
            phase_result_from_job(bad, P, "prepare", template=TD)


def test_no_bare_exception_escapes_a_corrupted_assemble_job() -> None:
    prep = _prepare(prepare_job(P, TD))
    for bad in _mutations(assemble_job(P, prep)):
        with contextlib.suppress(CompileError):
            phase_result_from_job(bad, P, "assemble", prepare=prep)


# -- N2/N3: appends must cover exactly their own ranges -----------------------------------------


def _assemble_fixture() -> tuple[PrepareResult, dict]:
    prep = _prepare(prepare_job(LF, LF_TD), LF, LF_TD)
    return prep, assemble_job(LF, prep)


def _append_value(job: dict, name: str = "append:v1:0") -> dict:
    return job_step(job, name)["value"]


@pytest.mark.parametrize("edit", ["not_done", "next_cursor", "short"])
def test_an_append_that_did_not_finish_is_incomplete(edit: str) -> None:
    prep, job = _assemble_fixture()
    value = _append_value(job)
    if edit == "not_done":
        value["done"] = False
    elif edit == "next_cursor":
        value["next_cursor"] = 1
    else:
        value["ranges"].pop()
    assert _assemble_code(job, prep, LF) == "incomplete_job_result"


@pytest.mark.parametrize(
    "edit", ["record_in", "video_track", "index", "audio_track", "audio_type", "no_video"]
)
def test_an_append_of_other_ranges_is_refused(edit: str) -> None:
    prep, job = _assemble_fixture()
    rng = _append_value(job)["ranges"][1]
    if edit == "record_in":
        rng["planned_record_in"] += 1
    elif edit == "video_track":
        rng["video"]["track"] += 1
    elif edit == "index":
        rng["index"] = 0
    elif edit == "audio_track":
        rng["audio"][0]["track"] += 1
    elif edit == "audio_type":
        rng["audio"][0]["track_type"] = "video"
    else:
        rng["video"] = None
    assert _assemble_code(job, prep, LF) == "forged_phase_result"


def test_an_audio_only_append_must_carry_audio_and_no_video() -> None:
    prep, job = _assemble_fixture()
    audio_step = next(s for s in job["steps"] if s["name"].startswith("append:outro-a"))
    video_step = job_step(job, "append:outro-v1")
    audio_step["value"]["ranges"][0]["video"] = copy.deepcopy(
        video_step["value"]["ranges"][0]["video"]
    )
    assert _assemble_code(job, prep, LF) == "forged_phase_result"
    prep, job = _assemble_fixture()
    audio_step = next(s for s in job["steps"] if s["name"].startswith("append:outro-a"))
    audio_step["value"]["ranges"][0]["audio"] = []
    assert _assemble_code(job, prep, LF) == "forged_phase_result"


def test_every_append_on_another_timeline_is_a_mismatch() -> None:
    # The branch is the prepare envelope's (N2), never taken from the appends themselves.
    prep, job = _assemble_fixture()
    for step in job["steps"]:
        if step["op"] == "timeline.append_ranges":
            step["value"]["timeline"]["unique_id"] = "tl-other"
    assert _assemble_code(job, prep, LF) == "phase_result_mismatch"


def test_tracks_ensured_on_another_timeline_are_a_mismatch() -> None:
    prep, job = _assemble_fixture()
    job_step(job, "tracks")["value"]["timeline"]["unique_id"] = "tl-other"
    assert _assemble_code(job, prep, LF) == "phase_result_mismatch"


def test_an_append_from_another_timeline_is_a_mismatch() -> None:
    prep, job = _assemble_fixture()
    _append_value(job)["timeline"]["unique_id"] = "tl-other"
    assert _assemble_code(job, prep, LF) == "phase_result_mismatch"


# -- inputs ---------------------------------------------------------------------------------------


def test_assemble_needs_the_prepare_envelope() -> None:
    _, job = _assemble_fixture()
    with pytest.raises(CompileError) as ei:
        phase_result_from_job(job, LF, "assemble")
    assert ei.value.code == "missing_prepare_result"


def test_assemble_refuses_a_stale_prepare_envelope() -> None:
    prep, job = _assemble_fixture()
    stale = prep.model_copy(update={"plan_sha256": "0" * 64})
    assert _assemble_code(job, stale, LF) == "stale_phase_result"


def test_prepare_needs_the_template_describe() -> None:
    with pytest.raises(CompileError) as ei:
        phase_result_from_job(prepare_job(P, TD), P, "prepare")
    assert ei.value.code == "missing_template"


def test_finish_is_not_built_from_a_job() -> None:
    with pytest.raises(ValueError, match="prepare/assemble"):
        phase_result_from_job(prepare_job(P, TD), P, "finish", template=TD)


# -- O1: the branch timeline id gets the forged-id rule ------------------------------------------

_BAD_BRANCH_IDS = [
    "E:/x",
    "a/b",
    "a\\b",
    "t:1",
    LF.template,
    LF.project,
    LF.timeline,
    LF.sources[0].id,
    LF.sources[0].camera,
    LF.sources[0].media,
    "CCN Lower Third",
    "",
    " ",
    "\t\n",
]


@pytest.mark.parametrize("uid", _BAD_BRANCH_IDS)
def test_o1_job_builder_refuses_a_forged_branch_id(uid: str) -> None:
    job = prepare_job(LF, LF_TD, uid=uid)
    assert _prepare_code(job, LF, LF_TD) == "forged_phase_result"


@pytest.mark.parametrize("uid", _BAD_BRANCH_IDS)
def test_o1_hand_written_prepare_file_refuses_a_forged_branch_id(uid: str) -> None:
    bad = synth_prepare(LF, unique_id=uid)
    for fn in (compile_assemble, compile_finish):
        with pytest.raises(CompileError) as ei:
            fn(LF, prepare=bad)
        assert ei.value.code == "forged_phase_result"


@pytest.mark.parametrize("uid", _BAD_BRANCH_IDS)
def test_o1_hand_written_assemble_file_refuses_a_forged_branch_id(uid: str) -> None:
    good = synth_prepare(LF)
    asm = synth_assemble(LF, good)
    bad = asm.model_copy(update={"branch": asm.branch.model_copy(update={"unique_id": uid})})
    with pytest.raises(CompileError) as ei:
        compile_finish(LF, prepare=good, assemble=bad)
    assert ei.value.code == "forged_phase_result"


@pytest.mark.parametrize("uid", _BAD_BRANCH_IDS)
def test_o1_assemble_builder_refuses_a_prepare_envelope_with_a_forged_branch(uid: str) -> None:
    good = _prepare(prepare_job(LF, LF_TD), LF, LF_TD)
    job = assemble_job(LF, good)
    bad = good.model_copy(update={"branch": good.branch.model_copy(update={"unique_id": uid})})
    assert _assemble_code(job, bad, LF) == "forged_phase_result"


def _all_branch_paths(prep: PrepareResult, good: PrepareResult, a: AssembleResult) -> None:
    calls = (
        lambda: compile_assemble(LF, prepare=prep),
        lambda: compile_finish(LF, prepare=prep),
        lambda: compile_finish(LF, prepare=good, assemble=a),
        lambda: phase_result_from_job(assemble_job(LF, good), LF, "assemble", prepare=prep),
    )
    for call in calls:
        with contextlib.suppress(CompileError):
            call()


def test_o1_no_untyped_error_escapes_for_any_branch_id() -> None:
    good = synth_prepare(LF)
    asm = synth_assemble(LF, good)
    for uid in [*_BAD_BRANCH_IDS, "\x00", "\u200b", "x" * 5000, "tl-ok"]:
        with contextlib.suppress(CompileError):
            phase_result_from_job(prepare_job(LF, LF_TD, uid=uid), LF, "prepare", template=LF_TD)
        prep = synth_prepare(LF, unique_id=uid)
        a = asm.model_copy(update={"branch": asm.branch.model_copy(update={"unique_id": uid})})
        _all_branch_paths(prep, good, a)


def test_o1_an_ordinary_branch_id_is_accepted() -> None:
    ok = _prepare(prepare_job(LF, LF_TD, uid="5f1c2e3a-0b9d-4c55-9a51-0d6c3a1f2b7e"), LF, LF_TD)
    assert compile_assemble(LF, prepare=ok).phase == "assemble"


# -- O2: an av append must come back with its linked audio ----------------------------------------


def test_o2_av_append_with_no_linked_audio_is_incomplete_and_names_the_segment() -> None:
    prep, job = _assemble_fixture()
    value = _append_value(job)
    index = next(
        i
        for i, r in enumerate(job_step(job, "append:v1:0")["params"]["ranges"])
        if r["media_type"] == "av"
    )
    value["ranges"][index]["audio"] = []
    with pytest.raises(CompileError) as ei:
        phase_result_from_job(job, LF, "assemble", prepare=prep)
    assert ei.value.code == "incomplete_job_result"
    assert "segment" in ei.value.message
