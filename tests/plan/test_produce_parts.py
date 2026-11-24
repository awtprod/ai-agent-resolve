"""``aar plan produce`` with a camera of several files (S24a): a camera label resolves to its
parts, a range is split where one file ends and the next begins, a range across a gap is cut
there with the frames with no file left out (and a warning), and two frame-contiguous files
play as one shot. Real ffmpeg media from ``ccn_support``; the transcript is the files' own
words in recording time."""

from __future__ import annotations

from fractions import Fraction
from typing import Any

import pytest

from aar.plan.inputs import Transcript, Word
from aar.plan.lint import lint
from aar.plan.model import EditPlan
from aar.plan.produce.build import ProduceResult, apply_sync, produce
from aar.plan.produce.errors import ProduceInputError
from aar.plan.produce.inputs import CameraSchedule, KeepList, SyncReport
from aar.plan.produce.timeline import Coverage, less_holes, on_camera, resolve_camera
from aar.plan.profile import load_profile
from aar.plan.stub import StubSource, build_stub
from ccn_support import WORDS, make_parts, make_show, needs_ffmpeg

#: Part 1 is recording frames 0-360, part 2 360-720 (no gap), then 90 frames (3.003 s) with no
#: file, part 3 810-1170.
GAPS = [0, 90]
ORIGINS = (0, 360, 810)


def _s(frames: int) -> float:
    return float(Fraction(frames * 1001, 30000))


@pytest.fixture(scope="module")
def show(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    folder = tmp_path_factory.mktemp("parts")
    files = make_parts(folder, GAPS)
    other = make_show(folder / "cam-b.mov", timecode="01:00:00:00")
    sources = [StubSource(path, camera="A", part=n + 1) for n, (path, _) in enumerate(files)]
    stub = build_stub(
        [*sources, StubSource(other, camera="B")],
        recording_id="show-1",
        project="Synthetic Show",
        timeline="Synthetic v1",
        kind="longform",
        profile="ccn",
    ).plan
    assert [s.recording_origin for s in stub.sources] == [*ORIGINS, 0]
    words = [
        Word(text=t, start_s=a + _s(origin), end_s=b + _s(origin))
        for origin in ORIGINS
        for t, a, b in WORDS
    ]
    return {"stub": stub, "transcript": Transcript(recording="show-1", words=words)}


def _produce(show: dict[str, Any], ranges: list[tuple[float, float]], **kw: Any) -> ProduceResult:
    keep = KeepList.model_validate(
        {"recording": "show-1", "ranges": [{"start_s": a, "end_s": b} for a, b in ranges]}
    )
    kw.setdefault("pause_trim", False)
    return produce(
        show["stub"], transcript=show["transcript"], keep=keep, profile=load_profile("ccn"), **kw
    )


def _segs(result: ProduceResult) -> list[tuple[str, int, int, float]]:
    return [
        (s.source, s.src_in, s.src_out, s.transform.zoom if s.transform else 1.0)
        for s in result.plan.segments
    ]


@needs_ffmpeg
def test_a_range_across_two_contiguous_files_plays_as_one_shot(show: dict[str, Any]) -> None:
    # 8.9-12.4 s: part 1's last sentence, the seam at 12.012 s, 0.39 s of part 2 (a piece under
    # min_shot_s on its own, but one shot with the piece before it: not a sliver).
    result = _produce(show, [(8.9, 12.4)])
    segs = _segs(result)
    assert [s[0] for s in segs] == ["cam-a-p1", "cam-a-p2"]
    assert segs[0][2] == 360 and segs[1][1] == 0
    assert segs[0][3] == segs[1][3] == 1.0  # no punch-in reset at the seam
    assert result.plan.transitions == [] and result.cuts == 0
    assert not result.folds
    assert not [w for w in result.warnings if w.code == "range-crosses-gap"]
    # A human cut later on the same camera still punches in, and the seam keeps that level.
    result = _produce(show, [(4.9, 8.1), (8.9, 12.4)])
    assert [s[3] for s in _segs(result)] == [1.0, 1.25, 1.25]
    assert len(result.plan.transitions) == 1
    assert lint(result.plan, load_profile("ccn"), transcript=show["transcript"]).ok


@needs_ffmpeg
@pytest.mark.parametrize("pause_trim", [False, True])
def test_a_range_across_a_gap_is_cut_there_and_warns(
    show: dict[str, Any], pause_trim: bool
) -> None:
    result = _produce(show, [(20.9, 31.5)], pause_trim=pause_trim)
    segs = _segs(result)
    assert [s[0] for s in segs] == ["cam-a-p2", "cam-a-p3"]
    # The hole: the frames with no file, and the crossfade's 1-frame handle at each file edge.
    assert segs[0][2] == 360 - 1 and segs[1][1] == 1
    assert segs[0][3] != segs[1][3]  # a jump cut on one camera: the framing changes
    assert [t.after for t in result.plan.transitions] == [0]
    (warning,) = [w for w in result.warnings if w.code == "range-crosses-gap"]
    assert warning.at_s == round(_s(720), 3) and warning.source == "cam-a-p3"
    assert "3.003 s gap" in warning.message and "'cam-a-p2' ends at 24.024 s" in warning.message
    report = lint(result.plan, load_profile("ccn"), transcript=show["transcript"])
    assert report.ok, [f.message for f in report.findings if f.severity == "error"]


@needs_ffmpeg
def test_a_short_piece_before_a_gap_is_left_out_with_it(show: dict[str, Any]) -> None:
    # 23.5 s is 0.52 s before part 2 ends: too short to keep on its own.
    result = _produce(show, [(23.5, 31.5)])
    segs = _segs(result)
    assert [s[0] for s in segs] == ["cam-a-p3"] and segs[0][1] == 1
    (warning,) = [w for w in result.warnings if w.code == "range-crosses-gap"]
    assert "too short to keep, 23.523-23.991 s went too" in warning.message


@needs_ffmpeg
def test_a_range_in_a_gap_or_past_the_last_file_is_refused(show: dict[str, Any]) -> None:
    with pytest.raises(ProduceInputError) as caught:
        _produce(show, [(24.5, 26.0)])
    error = caught.value
    assert error.code == "range-in-gap"
    assert "24.524-25.993 s on camera 'A' lies wholly in a gap" in error.message
    assert "cam-a-p2 12.012-24.024 s" in error.message
    assert "cam-a-p3 27.027-39.039 s" in error.message
    with pytest.raises(ProduceInputError) as caught:
        _produce(show, [(36.0, 40.5)])
    assert caught.value.code == "camera-does-not-cover"
    assert "its files cover" in caught.value.message


@needs_ffmpeg
def test_a_schedule_names_the_camera_and_its_parts_follow(show: dict[str, Any]) -> None:
    schedule = CameraSchedule.model_validate(
        {
            "spans": [
                {"start_s": 8.0, "end_s": 11.05, "camera": "B"},
                {"start_s": 11.05, "end_s": 13.0, "camera": "A"},
            ]
        }
    )
    result = _produce(show, [(8.9, 12.4)], schedule=schedule)
    assert [s[0] for s in _segs(result)] == ["cam-b", "cam-a-p1", "cam-a-p2"]


@needs_ffmpeg
def test_a_sync_report_never_moves_a_placed_part(show: dict[str, Any]) -> None:
    stub: EditPlan = show["stub"]
    a1, a2, _, b = stub.sources
    report = SyncReport.model_validate(
        {
            "fps": "30000/1001",
            "reference": {"file": a1.media, "recording_origin": 0},
            "pairs": [{"file": b.media, "status": "ok", "suggested_recording_origin": 15}],
        }
    )
    synced = apply_sync(stub, report, None)
    assert [s.recording_origin for s in synced.sources] == [*ORIGINS, 15]
    bad = report.model_copy(
        update={
            "pairs": [
                *report.pairs,
                report.pairs[0].model_copy(update={"file": a2.media}),
            ]
        }
    )
    with pytest.raises(ProduceInputError) as caught:
        apply_sync(stub, bad, None)
    assert caught.value.code == "sync-part-placed" and "part 2" in caught.value.message


# -- pure ----------------------------------------------------------------------------------------


def _plan(sources: list[dict[str, Any]]) -> EditPlan:
    return EditPlan.model_validate(
        {
            "project": "P",
            "timeline": "T",
            "kind": "longform",
            "profile": "ccn",
            "fps": {"num": 30, "den": 1},
            "resolution": {"width": 320, "height": 180},
            "sources": sources,
            "segments": [],
        }
    )


def test_a_label_on_overlapping_sources_is_still_ambiguous() -> None:
    parts = _plan(
        [
            {
                "id": "a2",
                "media": "a2.mov",
                "camera": "A",
                "recording_id": "r",
                "recording_origin": 100,
                "duration": 50,
            },
            {"id": "a1", "media": "a1.mov", "camera": "A", "recording_id": "r", "duration": 100},
        ]
    )
    assert resolve_camera(parts, "A") == ("a1", "a2")
    assert resolve_camera(parts, "a2") == ("a2",)
    angles = _plan(
        [
            {"id": "a1", "media": "a1.mov", "camera": "A", "recording_id": "r", "duration": 100},
            {"id": "a2", "media": "a2.mov", "camera": "A", "recording_id": "r", "duration": 100},
        ]
    )
    with pytest.raises(ProduceInputError) as caught:
        resolve_camera(angles, "A")
    assert caught.value.code == "ambiguous-camera"


def test_on_camera_splits_at_file_edges_and_the_earlier_file_wins_an_overlap() -> None:
    from aar.plan.model import Fps

    fps = Fps(num=30, den=1)
    coverage = {
        "a1": Coverage(0, 100, "A"),
        "a2": Coverage(99, 100, "A"),  # one frame of overlap
        "a3": Coverage(229, 100, "A"),  # 30 frames of gap
    }
    pieces = on_camera(50, 300, 0, ("a1", "a2", "a3"), coverage, fps)
    assert [(p.source, p.rec_in, p.rec_out) for p in pieces] == [
        ("a1", 50, 100),
        ("a2", 100, 199),
        ("a3", 229, 300),
    ]
    assert less_holes([(0, 100), (120, 200)], [(90, 130), (150, 160)]) == [
        (0, 90),
        (130, 150),
        (160, 200),
    ]
