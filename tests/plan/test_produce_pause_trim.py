"""S19d: pause trim is a real switch (``[produce] pause_trim``, ``--no-pause-trim``).

The S19 pilot's shape: one camera, two keep ranges. With pause trim on, the synthetic
long-form's quiet runs become jump cuts; off, the keep list gives exactly 2 segments and the
one human cut, while every other step (keep edges, the join nudge into quiet) still runs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from produce_support import needs_ffmpeg, stage

from aar.cli import main
from aar.plan.produce import load_keep, load_stub, produce, read_transcript
from aar.plan.profile import load_profile

pytestmark = needs_ffmpeg

PARTS = ("stub", "transcript", "keep")


@pytest.fixture(scope="module")
def show(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    folder = tmp_path_factory.mktemp("one-camera")
    return {"dir": folder, **stage("longform", folder, PARTS)}


def one_camera(show: dict[str, Path], **kw: Any) -> Any:
    """No schedule: the profile's default camera A plays both keep ranges."""
    return produce(
        load_stub(show["stub"]),
        transcript=read_transcript(show["transcript"]),
        keep=load_keep(show["keep"]),
        profile=kw.pop("profile", load_profile("ccn")),
        media_root=show["dir"],
        **kw,
    )


def frames(result: Any) -> int:
    return sum(s.src_out - s.src_in for s in result.plan.segments)


def no_trim_profile(tmp_path: Path) -> Path:
    path = tmp_path / "notrim.toml"
    text = (Path(__file__).parents[2] / "src/aar/plan/profiles/ccn.toml").read_text("utf-8")
    assert "pause_trim = true" in text
    path.write_text(text.replace("pause_trim = true", "pause_trim = false"), encoding="utf-8")
    return path


def test_the_default_still_trims_pauses(show: dict[str, Path]) -> None:
    result = one_camera(show)
    log = result.log()
    assert log["pause_trim"] == {"ran": True, "source": "profile"}
    assert len(log["trims"]) == 3
    assert len(result.plan.segments) == 5 and result.cuts == 4


def test_pause_trim_off_gives_two_segments_and_one_cut(show: dict[str, Path]) -> None:
    result = one_camera(show, pause_trim=False)
    report = result.report()
    assert report.segments == 2 and report.cuts == 1
    assert report.log["pause_trim"] == {"ran": False, "source": "argument"}
    assert report.log["trims"] == []
    # Every other step still runs: the second range's in point still moves into quiet.
    assert [
        (n["join"], n["side"], n["from_frame"], n["to_frame"]) for n in report.log["nudges"]
    ] == [(1, "in", 582, 584)]
    # The keep ranges play whole: exactly what the trimming run plays plus what its trims
    # removed (both runs move the same keep edges inward and nudge the same in point).
    trimmed = one_camera(show)
    removed = sum(t.removed_frames for t in trimmed.trims)
    assert removed > 0
    assert frames(result) == frames(trimmed) + removed
    assert report.lint.ok


def test_the_profile_switch_and_the_argument_override_it(
    show: dict[str, Path], tmp_path: Path
) -> None:
    profile = load_profile(str(no_trim_profile(tmp_path)))
    assert profile.produce.pause_trim is False
    off = one_camera(show, profile=profile)
    assert off.log()["pause_trim"] == {"ran": False, "source": "profile"}
    assert len(off.plan.segments) == 2
    on = one_camera(show, profile=profile, pause_trim=True)
    assert on.log()["pause_trim"] == {"ran": True, "source": "argument"}
    assert len(on.plan.segments) == 5


def test_pause_trim_off_never_scans_for_pauses(
    show: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    import aar.plan.produce.build as build

    def no_scan(*_args: Any, **_kwargs: Any) -> list[tuple[float, float]]:
        raise AssertionError("the pause scan ran with pause trim off")

    monkeypatch.setattr(build, "pause_runs", no_scan)
    result = one_camera(show, pause_trim=False)
    assert len(result.plan.segments) == 2
    with pytest.raises(AssertionError, match="pause scan ran"):
        one_camera(show)


def test_off_is_not_a_large_min_pause(show: dict[str, Path], tmp_path: Path) -> None:
    profile = load_profile("ccn")
    assert profile.produce.min_pause_s == 0.9  # the switch leaves the pause settings alone
    result = one_camera(show, profile=profile, pause_trim=False)
    assert result.log()["trims"] == [] and profile.produce.min_pause_s == 0.9


def run(capsys: pytest.CaptureFixture[str], *argv: object) -> tuple[int, str]:
    code = main(["plan", "produce", *(str(a) for a in argv)])
    return code, capsys.readouterr().out


def test_cli_flags(
    show: dict[str, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    common = [show["stub"], "--transcript", show["transcript"], "--keep", show["keep"]]
    code, text = run(capsys, *common, "--out", tmp_path / "off.json", "--no-pause-trim")
    assert code == 0
    assert "2 segment(s), 1 cut(s)" in text and "pause trim off" in text
    code, text = run(capsys, *common, "--out", tmp_path / "on.json", "--json")
    assert code == 0 and json.loads(text)["log"]["pause_trim"] == {"ran": True, "source": "profile"}
    assert json.loads(text)["segments"] == 5
    notrim = no_trim_profile(tmp_path)
    code, text = run(capsys, *common, "--out", tmp_path / "p.json", "--profile", notrim, "--json")
    assert code == 0 and json.loads(text)["log"]["pause_trim"] == {
        "ran": False,
        "source": "profile",
    }
    code, text = run(
        capsys, *common, "--out", tmp_path / "q.json", "--profile", notrim, "--pause-trim", "--json"
    )
    body = json.loads(text)
    assert code == 0 and body["log"]["pause_trim"] == {"ran": True, "source": "argument"}
    assert body["segments"] == 5


def test_cli_flags_are_exclusive(
    show: dict[str, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["plan", "produce", str(show["stub"]), "--transcript", str(show["transcript"]),
                 "--keep", str(show["keep"]), "--out", str(tmp_path / "x.json"),
                 "--pause-trim", "--no-pause-trim"])  # fmt: skip
    assert code == 2 and "not allowed with" in capsys.readouterr().err
    assert not (tmp_path / "x.json").exists()
