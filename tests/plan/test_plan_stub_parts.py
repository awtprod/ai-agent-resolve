"""A reference camera of several files in the plan stub (S24a): ``aar.plan.stub`` places each
part on the recording clock by its start timecode, or contiguously, and refuses parts that are
not one camera's consecutive files."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import pytest

from aar.cli import main
from aar.plan.produce.build import StartTimecode
from aar.plan.stub import StubError, StubSource, build_stub, place_parts
from ccn_support import frame_count, make_parts, make_show, needs_ffmpeg

HOUR = 108000  # 01:00:00:00 at 30000/1001, non-drop
NTSC = Fraction(30000, 1001)


def _stub(sources: list[StubSource], **kw: object):  # type: ignore[no-untyped-def]
    args: dict[str, object] = {
        "recording_id": "show-1",
        "project": "Synthetic Show",
        "timeline": "Synthetic v1",
        "kind": "longform",
        "profile": "ccn",
    }
    args.update(kw)
    return build_stub(sources, **args)  # type: ignore[arg-type]


def _code(sources: list[StubSource], **kw: object) -> StubError:
    with pytest.raises(StubError) as caught:
        _stub(sources, **kw)
    return caught.value


def _parts(files: list[tuple[Path, int]], camera: str = "A") -> list[StubSource]:
    return [StubSource(path, camera=camera, part=n + 1) for n, (path, _) in enumerate(files)]


# -- placement, pure ---------------------------------------------------------------------------


def _entries(durations: list[int]) -> list[dict[str, object]]:
    return [
        {"id": f"p{n + 1}", "part": n + 1, "duration": d, "recording_origin": 0}
        for n, d in enumerate(durations)
    ]


def test_timecode_placement_is_each_start_less_the_first() -> None:
    entries = _entries([100, 100, 100])
    tcs = {"p1": StartTimecode(HOUR, NTSC), "p2": StartTimecode(HOUR + 100, NTSC)}
    tcs["p3"] = StartTimecode(HOUR + 224, NTSC)
    placed = place_parts(entries, tcs, NTSC, "timecode")
    assert [e["recording_origin"] for e in entries] == [0, 100, 224]
    assert [p["gap_frames"] for p in placed] == [0, 0, 24]
    assert {p["placement"] for p in placed} == {"timecode"}
    assert [e["part_placement"] for e in entries] == ["timecode"] * 3


def test_contiguous_placement_ignores_the_timecodes() -> None:
    entries = _entries([100, 50, 70])
    tcs = {f"p{n}": StartTimecode(HOUR + 999 * n, NTSC) for n in (1, 2, 3)}
    placed = place_parts(entries, tcs, NTSC, "contiguous")
    assert [e["recording_origin"] for e in entries] == [0, 100, 150]
    assert [p["gap_frames"] for p in placed] == [0, 0, 0]


def test_a_part_past_midnight_is_the_next_day_at_drop_frame_too() -> None:
    day_df = 2589408  # the frames between two midnights at 29.97 drop-frame
    entries = _entries([300, 300])
    tcs = {
        "p1": StartTimecode(day_df - 100, NTSC, drop_frame=True),
        "p2": StartTimecode(210, NTSC, drop_frame=True),  # 10 frames after p1 ends
    }
    placed = place_parts(entries, tcs, NTSC, "timecode")
    assert [e["recording_origin"] for e in entries] == [0, 310]
    assert placed[1]["gap_frames"] == 10
    # Non-drop counts 30 labels a second.
    entries = _entries([300, 300])
    tcs = {"p1": StartTimecode(86400 * 30 - 100, NTSC), "p2": StartTimecode(200, NTSC)}
    place_parts(entries, tcs, NTSC, "timecode")
    assert entries[1]["recording_origin"] == 300


def test_overlapping_and_misordered_parts_are_refused() -> None:
    tcs = {"p1": StartTimecode(HOUR, NTSC), "p2": StartTimecode(HOUR + 90, NTSC)}
    with pytest.raises(StubError) as caught:
        place_parts(_entries([100, 100]), tcs, NTSC, "timecode")
    assert caught.value.code == "parts_overlap" and caught.value.details["overlap_frames"] == 10
    # Two frames of overlap are timecode rounding: allowed.
    tcs["p2"] = StartTimecode(HOUR + 98, NTSC)
    placed = place_parts(_entries([100, 100]), tcs, NTSC, "timecode")
    assert placed[1]["gap_frames"] == -2
    # A part whose timecode is before the previous one's, not across midnight.
    tcs["p2"] = StartTimecode(HOUR - 5000, NTSC)
    with pytest.raises(StubError) as caught:
        place_parts(_entries([100, 100]), tcs, NTSC, "timecode")
    assert caught.value.code == "parts_out_of_order"
    # A timecode at another rate than the plan's cannot place a part.
    tcs = {"p1": StartTimecode(90000, Fraction(25)), "p2": StartTimecode(90100, Fraction(25))}
    with pytest.raises(StubError) as caught:
        place_parts(_entries([100, 100]), tcs, NTSC, "timecode")
    assert caught.value.code == "parts_rate"


# -- build_stub on real files ------------------------------------------------------------------


@needs_ffmpeg
def test_three_parts_by_timecode_with_ids_origins_and_report(tmp_path: Path) -> None:
    files = make_parts(tmp_path, [0, 24], seconds=2.0)
    frames = frame_count(files[0][0])
    other = make_show(tmp_path / "cam-b.mov", seconds=2.0)
    # Given out of order and with another camera between: the stub lists the parts first.
    sources = _parts(files)
    result = _stub([sources[1], StubSource(other, camera="B"), sources[0], sources[2]])
    plan = result.plan
    assert [(s.id, s.camera, s.part, s.recording_origin) for s in plan.sources] == [
        ("cam-a-p1", "A", 1, 0),
        ("cam-a-p2", "A", 2, frames),
        ("cam-a-p3", "A", 3, 2 * frames + 24),
        ("cam-b", "B", None, 0),
    ]
    assert [s.part_placement for s in plan.sources] == ["timecode"] * 3 + [None]
    assert [s.start_timecode for s in plan.sources[:3]] == [tc for _, tc in files]
    report = result.report
    assert report["cameras"] == ["A", "B"] and report["part_placement"] == "timecode"
    assert [(p["id"], p["recording_origin"], p["gap_frames"]) for p in report["parts"]] == [
        ("cam-a-p1", 0, 0),
        ("cam-a-p2", frames, 0),
        ("cam-a-p3", 2 * frames + 24, 24),
    ]
    assert [p["duration"] for p in report["parts"]] == [frames] * 3


@needs_ffmpeg
def test_contiguous_placement_on_request_and_without_timecodes(tmp_path: Path) -> None:
    files = make_parts(tmp_path, [24], seconds=2.0)
    frames = frame_count(files[0][0])
    plan = _stub(_parts(files), placement={"A": "contiguous"}).plan
    assert [s.recording_origin for s in plan.sources] == [0, frames]
    assert {s.part_placement for s in plan.sources} == {"contiguous"}
    # No timecode at all: refused unless the caller opts out, then placed contiguously.
    bare = [make_show(tmp_path / f"bare{n}.mov", seconds=2.0, timecode=None) for n in (1, 2)]
    parts = [StubSource(p, camera="A", part=n + 1) for n, p in enumerate(bare)]
    assert _code(parts).code == "timecode_unknown"
    result = _stub(parts, allow_unknown_timecode=True)
    assert [s.recording_origin for s in result.plan.sources] == [0, frames]
    assert "parts-contiguous" in [w["code"] for w in result.report["warnings"]]
    error = _code(parts, allow_unknown_timecode=True, placement={"A": "timecode"})
    assert error.code == "timecode_unknown"


@needs_ffmpeg
def test_parts_across_midnight_and_refused_layouts(tmp_path: Path) -> None:
    near = 86400 * 30 - 30  # 23:59:59:00
    files = make_parts(tmp_path / "wrap", [6], seconds=2.0, start_frames=near)
    frames = frame_count(files[0][0])
    assert files[1][1] < near  # its label is past midnight: 00:00:01:06
    plan = _stub(_parts(files)).plan
    assert [s.recording_origin for s in plan.sources] == [0, frames + 6]

    files = make_parts(tmp_path / "overlap", [-30], seconds=2.0)
    assert _code(_parts(files)).code == "parts_overlap"
    files = make_parts(tmp_path / "order", [0], seconds=2.0)
    swapped = [
        StubSource(files[1][0], camera="A", part=1),
        StubSource(files[0][0], camera="A", part=2),
    ]
    assert _code(swapped).code == "parts_out_of_order"


@needs_ffmpeg
def test_part_numbers_labels_and_other_cameras_are_checked(tmp_path: Path) -> None:
    files = make_parts(tmp_path, [0, 0], seconds=2.0)
    a, b, c = (path for path, _ in files)
    assert _code([StubSource(a, camera="A", part=1), StubSource(b, camera="A", part=3)]).code == (
        "bad_parts"
    )
    assert _code([StubSource(a, camera="A", part=1), StubSource(b, camera="A", part=1)]).code == (
        "bad_parts"
    )
    assert _code([StubSource(a, camera="A"), StubSource(b, camera="A")]).code == "bad_parts"
    assert _code([StubSource(a, part=1), StubSource(b, part=2)]).code == "bad_parts"
    error = _code(
        [
            StubSource(a, camera="A"),
            StubSource(b, camera="B", part=1),
            StubSource(c, camera="B", part=2),
        ]
    )
    assert error.code == "parts_unsupported" and "S24b" in error.hint
    error = _code([StubSource(a, camera="A")], placement={"A": "contiguous"})
    assert error.code == "bad_parts"
    # One file given as part 1 is just a camera of one file.
    plan = _stub([StubSource(a, camera="A", part=1)]).plan
    assert [(s.id, s.part, s.part_placement) for s in plan.sources] == [("cam-a", None, None)]


@needs_ffmpeg
def test_cli_takes_parts_and_a_placement(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    files = make_parts(tmp_path, [24], seconds=2.0)
    frames = frame_count(files[0][0])
    out = tmp_path / "stub.json"
    argv = ["plan", "stub", *(str(p) for p, _ in files), "--out", str(out), "--recording", "r"]
    argv += ["--project", "P", "--timeline", "T", "--camera", "A", "--camera", "A"]
    argv += ["--part", "1", "--part", "2", "--json"]
    assert main([*argv, "--placement", "A=contiguous"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert [p["recording_origin"] for p in report["parts"]] == [0, frames]
    assert main([*argv, "--force"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert [p["recording_origin"] for p in report["parts"]] == [0, frames + 24]
    stub = json.loads(out.read_text())
    assert [s["part"] for s in stub["sources"]] == [1, 2]
    assert main([*argv, "--force", "--placement", "A=sideways"]) == 2


@needs_ffmpeg
def test_parts_and_extra_sources_compose(tmp_path: Path) -> None:
    """S24 graphics overlays (extra sources) beside a reference camera of several files: the
    parts come first and are placed; the overlay is never a part, has no recording and is not a
    camera."""
    files = make_parts(tmp_path, [24], seconds=2.0)
    frames = frame_count(files[0][0])
    gfx = make_show(tmp_path / "gfx.mov", seconds=2.0, audio=False, timecode="00:00:00:00")
    sources = _parts(files)
    result = _stub([sources[1], sources[0]], extra_sources=[StubSource(gfx, id="gfx-1")])
    plan = result.plan
    assert [(s.id, s.part, s.recording_id, s.recording_origin) for s in plan.sources] == [
        ("cam-a-p1", 1, "show-1", 0),
        ("cam-a-p2", 2, "show-1", frames + 24),
        ("gfx-1", None, None, 0),
    ]
    assert result.report["cameras"] == ["A"] and len(result.report["parts"]) == 2
    error = _code(sources, extra_sources=[StubSource(gfx, id="gfx-1", part=3)])
    assert error.code == "bad_parts"
