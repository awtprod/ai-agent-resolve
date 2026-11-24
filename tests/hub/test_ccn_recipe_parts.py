"""``ccn-longform`` with a reference camera of several files (S24a): the params, the per-part
transcript (its merge, reuse per part, resume per part), the sync step, the plan check and the
guide's layout. The full run is in ``test_ccn_recipe_parts_e2e.py``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ccn_recipe_support import CcnEnv, fake_transcriber, latest, make_run, outputs, start, work
from write_support import AGENT

from aar.analyze.errors import AnalyzeError
from aar.plan.inputs import Transcript, Word
from aar.plan.model import Source
from aar.recipes import registry
from aar.recipes.ccn import CcnLongformParams, set_transcriber
from aar.recipes.ccn.guide import part_layout
from aar.recipes.ccn.params import MAX_SOURCES
from aar.recipes.ccn.transcript import PartText, merge_parts
from aar.recipes.run import recipe_retry
from ccn_support import WORDS, make_parts, needs_ffmpeg


def _problems(env: CcnEnv, params: dict[str, Any]) -> list[str]:
    from write_support import brief_data

    from aar.hub.writes import Brief

    brief = Brief.model_validate(brief_data(env.hub.project_id, kind="longform"))
    return [
        f"{loc}: {msg}"
        for loc, msg in registry.registry_params_validator("ccn-longform", params, brief)
    ]


def _files(env: CcnEnv, count: int) -> list[Path]:
    paths = [env.media / f"cam-a_C{n + 1:03d}.mov" for n in range(count)]
    for path in paths:
        path.write_bytes(b"x")
    return paths


def _labelled(env: CcnEnv, paths: list[Path], cameras: list[tuple[str, int | None]], **kw: Any):  # type: ignore[no-untyped-def]
    params = env.params(*paths, **kw)
    for source, (camera, part) in zip(params["sources"], cameras, strict=True):
        source["camera"] = camera
        if part is not None:
            source["part"] = part
    return params


# -- params ------------------------------------------------------------------------------------


def test_params_take_a_reference_camera_of_parts_and_count_cameras(ccn_env: CcnEnv) -> None:
    env = ccn_env
    a1, a2, a3, b = _files(env, 4)
    parts = [("A", 1), ("A", 2), ("A", 3)]
    ok = _labelled(env, [a1, a2, a3], parts)
    assert _problems(env, ok) == []
    params = CcnLongformParams.model_validate(ok)
    assert params.cameras() == 1 and params.reference_parts() == 3
    assert params.pause_trim_on() is False  # one camera's room audio, however many files
    two = CcnLongformParams.model_validate(
        _labelled(env, [a2, b, a1], [("A", 2), ("B", None), ("A", 1)])
    )
    assert two.cameras() == 2 and two.pause_trim_on() is True
    placed = _labelled(env, [a1, a2], parts[:2], part_placement={"A": "contiguous"})
    assert _problems(env, placed) == []

    cases: list[tuple[dict[str, Any], str]] = [
        (_labelled(env, [a1, a2], [("A", None), ("A", None)]), "is given 2 times"),
        (_labelled(env, [a1, a2], [("A", 1), ("A", 1)]), "must be parts 1 to 2"),
        (_labelled(env, [a1, a2], [("A", 1), ("A", 3)]), "must be parts 1 to 2"),
        (_labelled(env, [a1, a2], [("A", 1), ("A", None)]), "is given 2 times"),
        (_labelled(env, [b, a1, a2], [("B", None), ("A", 1), ("A", 2)]), "S24b"),
        (_labelled(env, [a1, a2], parts[:2], part_placement={"B": "timecode"}), "not several"),
        (_labelled(env, [a1, a2], parts[:2], part_placement={"A": "sideways"}), "part_placement"),
        (_labelled(env, [a1, a2], parts[:2], laughs=True), "laughs is not supported"),
        (_labelled(env, [a1], [("A", 0)]), "part"),
    ]
    no_label = env.params(a1, a2)
    for n, source in enumerate(no_label["sources"]):
        source.pop("camera")
        source["part"] = n + 1
    cases.append((no_label, "no camera label"))
    for case, expected in cases:
        problems = _problems(env, case)
        assert problems and any(expected in p for p in problems), (expected, problems)


def test_the_source_cap_fits_three_cameras_of_fifteen_files(ccn_env: CcnEnv) -> None:
    env = ccn_env
    assert MAX_SOURCES >= 45
    paths = _files(env, 45)
    params = _labelled(env, paths, [("A", n + 1) for n in range(45)])
    assert _problems(env, params) == []
    over = _files(env, MAX_SOURCES + 1)
    too_many = _labelled(env, over, [("A", n + 1) for n in range(len(over))])
    assert any("sources" in p for p in _problems(env, too_many))


# -- the transcript merge (pure) ---------------------------------------------------------------


def _part(n: int, start_s: float, end_s: float, words: list[tuple[str, float, float]]) -> PartText:
    source = Source(id=f"cam-a-p{n}", media=f"/m/{n}.mov", camera="A", part=n)
    transcript = Transcript(words=[Word(text=t, start_s=a, end_s=b) for t, a, b in words])
    report = {"wordless": {"total_s": 1.0, "file_duration_s": end_s - start_s, "stretches": [
        {"start_s": start_s, "end_s": start_s + 1.0, "duration_s": 1.0, "level_dbfs": -60}
    ]}}  # fmt: skip
    return PartText(source, start_s, end_s, transcript, report)


def test_merge_keeps_words_in_their_part_and_marks_where_files_meet() -> None:
    parts = [
        _part(1, 0.0, 10.0, [("hello", 1.0, 1.5), ("over", 9.8, 10.3)]),
        _part(2, 10.0, 20.0, [("ghost", 9.0, 9.9), ("again", 11.0, 11.4)]),
        _part(3, 20.8, 30.0, [("last", 22.0, 22.5)]),
    ]
    merged, text, report = merge_parts(parts, "show-1", wordless_min_s=60.0)
    assert merged.recording == "show-1"
    # A word past its file's end is cut back to it; one wholly outside its file is dropped.
    assert [(w.text, w.start_s, w.end_s) for w in merged.words] == [
        ("hello", 1.0, 1.5),
        ("over", 9.8, 10.0),
        ("again", 11.0, 11.4),
        ("last", 22.0, 22.5),
    ]
    assert text.splitlines() == [
        "[0001.0-0001.5] hello",
        "[0009.8-0010.0] over",
        "[0010.0-0010.0] -- part 2 begins (no gap) --",
        "[0011.0-0011.4] again",
        "[0020.0-0020.8] -- gap 0.8 s (no file) between part 2 and part 3 --",
        "[0022.0-0022.5] last",
    ]
    assert [p["source"] for p in report["parts"]] == ["cam-a-p1", "cam-a-p2", "cam-a-p3"]
    wordless = report["wordless"]
    assert wordless["total_s"] == 3.0 and len(wordless["stretches"]) == 3
    assert [s["start_s"] for s in wordless["stretches"]] == [0.0, 10.0, 20.8]


def test_merge_keeps_the_earlier_files_words_where_two_files_overlap() -> None:
    # part 2 starts 2 frames (0.067 s) before part 1 ends; both transcribed the word there
    parts = [
        _part(1, 0.0, 10.0, [("so", 9.5, 9.98)]),
        _part(2, 9.933, 20.0, [("so", 9.94, 9.98), ("then", 10.5, 11.0)]),
    ]
    merged, _, _ = merge_parts(parts, "show-1", wordless_min_s=60.0)
    assert [(w.text, w.start_s, w.end_s) for w in merged.words] == [
        ("so", 9.5, 9.98),
        ("then", 10.5, 11.0),
    ]


# -- the steps on real files -------------------------------------------------------------------


def _parts_run(env: CcnEnv, files: list[tuple[Path, int]], key: str, **kw: Any) -> str:
    paths = [p for p, _ in files]
    params = _labelled(env, paths, [("A", n + 1) for n in range(len(paths))], **kw)
    run = make_run(env, params, key=key)
    start(env, run, key)
    work(env)
    return run


@needs_ffmpeg
def test_each_part_is_reused_by_its_own_key_and_resumes_after_a_failure(ccn_env: CcnEnv) -> None:
    env = ccn_env
    files = make_parts(env.media, [0, 24], seconds=4.0)
    third = str(files[2][0])
    real = fake_transcriber(env)
    failed: list[str] = []

    def fail_third_once(media: Any, out: Any, **kw: Any) -> dict[str, Any]:
        if str(media) == third and not failed:
            failed.append(str(media))
            raise AnalyzeError("decode-failed", "the decoder stopped", hint="retry")
        return real(media, out, **kw)

    set_transcriber(fail_third_once)
    run1 = _parts_run(env, files, "1")
    op = latest(env, run1, "transcript")
    assert op.status == "failed" and op.result_error["code"] == "transcript_decode-failed"
    assert [c["media"] for c in env.calls] == [str(files[0][0]), str(files[1][0])]
    # The retry resumes with the two finished parts: only the third is transcribed.
    recipe_retry(env.hub.log, run1, "transcript", AGENT, idempotency_key="retry-tx")
    work(env)
    tx = outputs(env, run1, "transcript")
    assert [c["media"] for c in env.calls] == [str(files[0][0]), str(files[1][0]), third]
    assert [p["cached"] for p in tx["parts"]] == [False, False, False]
    # the fake writes every word of WORDS; only the 7 inside each 4 s file are the file's
    assert tx["cached"] is False and tx["words"] == 3 * sum(1 for w in WORDS if w[1] < 4.0)
    assert not list((env.work / "aar-transcripts").glob("*/*"))  # every part's folder removed
    offsets = [p["offset_s"] for p in tx["parts"]]
    assert offsets[0] == 0.0 and offsets[1] < offsets[2]

    # A second run on the same files reuses every part by its own key: nothing transcribed.
    run2 = _parts_run(env, files, "2", timeline="Second Cut")
    again = outputs(env, run2, "transcript")
    assert len(env.calls) == 3
    assert [p["cached"] for p in again["parts"]] == [True, True, True]
    assert again["cached"] is True and again["transcript"] == tx["transcript"]
    assert [p["cache_key"] for p in again["parts"]] == [p["cache_key"] for p in tx["parts"]]
    # The first file on its own (the same key: offset 0) is reused from the parts run too.
    solo = make_run(env, env.params(files[0][0], timeline="Solo"), key="solo")
    start(env, solo, "solo")
    work(env)
    one = outputs(env, solo, "transcript")
    assert one["cached"] is True and one["cache_key"] == tx["parts"][0]["cache_key"]
    assert one["transcript"] == tx["parts"][0]["transcript"]
    assert len(env.calls) == 3


@needs_ffmpeg
def test_a_later_part_with_other_bytes_is_its_own_key(ccn_env: CcnEnv) -> None:
    env = ccn_env
    files = make_parts(env.media, [0], seconds=4.0)
    run1 = _parts_run(env, files, "1")
    first = outputs(env, run1, "transcript")
    # Part 2 replaced by another file (other words): only part 2 is transcribed again.
    other = make_parts(env.media / "other", [0], seconds=4.0, gain=0.2)
    files[1][0].write_bytes(other[1][0].read_bytes())
    run2 = _parts_run(env, files, "2", timeline="Again")
    second = outputs(env, run2, "transcript")
    assert [p["cached"] for p in second["parts"]] == [True, False]
    assert second["parts"][0]["transcript"] == first["parts"][0]["transcript"]
    assert len(env.calls) == 3


@needs_ffmpeg
def test_sync_leaves_the_parts_alone_and_the_guide_lays_them_out(ccn_env: CcnEnv) -> None:
    env = ccn_env
    files = make_parts(env.media, [12], seconds=4.0)
    run = _parts_run(env, files, "1")
    assert outputs(env, run, "sync")["reason"].startswith("one camera")
    stub = outputs(env, run, "stub")
    assert [s["part"] for s in stub["sources"]] == [1, 2]
    assert {"size", "mtime_ns"} <= set(stub["sources"][1])
    guide = env.hub.store.read_bytes(outputs(env, run, "guide")["guide"]).decode()
    assert "Camera A is 2 files" in guide and "range-crosses-gap" in guide


def test_the_guide_lays_out_the_parts_and_gaps_in_recording_seconds() -> None:
    stub = {
        "camera_labels": ["A"],
        "part_placement": "timecode",
        "parts": [
            {"id": "cam-a-p1", "part": 1, "start_s": 0.0, "end_s": 600.0, "gap_s": 0.0},
            {"id": "cam-a-p2", "part": 2, "start_s": 600.0, "end_s": 1200.0, "gap_s": 0.0},
            {"id": "cam-a-p3", "part": 3, "start_s": 1200.8, "end_s": 1800.0, "gap_s": 0.8},
        ],
    }
    lines = part_layout(stub)
    assert lines[0].startswith("Camera A is 3 files") and "placed by timecode" in lines[0]
    body = "\n".join(lines)
    assert "- part 1 `cam-a-p1`: 0.000-600.000 s\n- (no gap" in body
    assert "- gap 0.800 s: no file, no picture or sound\n- part 3 `cam-a-p3`: 1200.800-" in body
    assert "range-crosses-gap" in body
    assert part_layout({"parts": []}) == []


def test_the_guide_counts_cameras_not_files() -> None:
    from test_lessons_guide import _guide_args

    from aar.recipes.ccn.guide import build_guide

    args = _guide_args()
    args["stub"].update(
        cameras=2,
        camera_labels=["A", "B"],
        sources=[
            {"camera": "A", "id": "cam-a-p1", "part": 1},
            {"camera": "A", "id": "cam-a-p2", "part": 2},
            {"camera": "B", "id": "cam-b"},
        ],
        parts=[
            {"id": "cam-a-p1", "part": 1, "start_s": 0.0, "end_s": 6.0, "gap_s": 0.0},
            {"id": "cam-a-p2", "part": 2, "start_s": 6.5, "end_s": 12.0, "gap_s": 0.5},
        ],
        part_placement="timecode",
    )
    guide = build_guide(**args)
    assert "cameras: A = `cam-a-p1`, `cam-a-p2`, B = `cam-b`" in guide
    assert '"camera": "A"' in guide and '"camera": "B"' in guide  # the schedule example
    assert "--schedule" in guide
    one = _guide_args()
    one["stub"].update(cameras=1, camera_labels=["A"], sources=args["stub"]["sources"][:2])
    assert "--schedule" not in build_guide(**one)


@needs_ffmpeg
def test_the_plan_check_pins_each_parts_place_on_the_clock(ccn_env: CcnEnv) -> None:
    from test_ccn_recipe_r2 import _submit_edited_stub

    env = ccn_env
    files = make_parts(env.media, [12])
    run = _parts_run(env, files, "1")
    _submit_edited_stub(env, run, lambda d: d["sources"][1].update(recording_origin=999), "p1")
    work(env)
    op = latest(env, run, "render")
    assert op.result_error["code"] == "plan_not_from_stub"
    assert "source 'cam-a-p2' recording_origin is 999, the stub's 372" in op.result_error["message"]
