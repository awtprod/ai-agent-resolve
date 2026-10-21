"""check() end to end on generated media: a pass and a fail fixture per check, boundaries,
measured values with stated tolerances, and hostile inputs."""

from __future__ import annotations

import importlib
import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from media_support import (
    FFMPEG,
    aeval,
    fake_tool,
    gapped,
    needs_ffmpeg,
    plan_dict,
    posix_only,
    render,
    sine,
    square,
    truncate,
)

from aar.media import MediaError, MediaTimeout, MediaToolMissing, OutputTooLarge
from aar.plan.inputs import Transcript, Word
from aar.plan.model import EditPlan
from aar.plan.profile import Profile, load_profile
from aar.qa import CHECK_IDS, QAReport, check, kept_words

pytestmark = needs_ffmpeg

CCN = load_profile("ccn")


def profile_with(**qa: Any) -> Profile:
    return CCN.model_copy(update={"qa": CCN.qa.model_copy(update=qa)})


def run(path: Path | str, kind: str = "short", **kw: Any) -> QAReport:
    kw.setdefault("profile", CCN)
    return check(path, kind=kind, **kw)  # type: ignore[arg-type]


def plan_of(data: dict[str, Any]) -> EditPlan:
    return EditPlan.model_validate(data)


# ---------------------------------------------------------------- report shape


def test_report_has_every_check_once_in_order(tmp_path):
    report = run(sine(tmp_path / "s.wav", -14.5))
    assert [c.id for c in report.checks] == list(CHECK_IDS)
    assert report.status == "pass" and report.ok
    assert report.counts == {"skipped": 5, "pass": 4, "warn": 0, "fail": 0}
    skipped = {c.id: c.skip_reason for c in report.checks if c.status == "skipped"}
    assert skipped == {
        "size-cap": "not-applicable",
        "duration": "no-plan",
        "join-quiet": "no-plan",
        "join-retranscribe": "no-plan",
        "caption-coverage": "no-plan",
    }
    for entry in report.checks:
        assert entry.message and "\n" not in entry.message
        if entry.status == "skipped":
            assert entry.message.startswith("Skipped: ")
    assert report.media is not None and report.media["streams"][0]["kind"] == "audio"
    assert QAReport.model_validate_json(report.model_dump_json()) == report


# ---------------------------------------------------------------- loudness


@pytest.mark.parametrize(
    ("peak", "status"),
    [(-14.5, "pass"), (-14.0, "pass"), (-15.0, "pass"), (-13.9, "fail"), (-15.1, "fail")],
)
def test_short_loudness_at_and_past_the_tolerance(tmp_path, peak, status):
    entry = run(sine(tmp_path / "s.wav", peak)).check("loudness")
    assert entry.status == status
    assert entry.measured["integrated_lufs"] == pytest.approx(peak, abs=0.05)
    assert entry.measured["true_peak_dbtp"] == pytest.approx(peak, abs=0.05)
    assert entry.target == {
        "integrated_lufs": -14.5,
        "tolerance_lu": 0.5,
        "true_peak_max_dbtp": -1.0,
    }


@pytest.mark.parametrize(
    ("peak", "kind", "status"),
    [
        (-21.5, "longform", "pass"),
        (-22.5, "longform", "pass"),
        (-21.4, "longform", "fail"),
        (-22.0, "review", "pass"),
        (-14.5, "review", "fail"),
    ],
)
def test_longform_and_review_use_the_longform_target(tmp_path, peak, kind, status):
    entry = run(sine(tmp_path / "s.wav", peak), kind=kind).check("loudness")
    assert entry.status == status
    assert entry.target["integrated_lufs"] == -22.0


def test_quiet_short_fails_with_the_gain_to_add(tmp_path):
    entry = run(sine(tmp_path / "s.wav", -20.0)).check("loudness")
    assert entry.status == "fail"
    assert "raise the gain by 5.5 dB" in entry.message


@pytest.mark.parametrize(("peak", "status"), [(-1.0, "pass"), (-0.9, "fail")])
def test_true_peak_exactly_minus_one_passes(tmp_path, peak, status):
    # A target of -1 LUFS makes the integrated level pass, so only the true peak decides.
    profile = profile_with(short_lufs=-1.0)
    entry = run(sine(tmp_path / "s.wav", peak), profile=profile).check("loudness")
    assert entry.measured["true_peak_dbtp"] == pytest.approx(peak, abs=0.05)
    assert entry.status == status
    if status == "fail":
        assert entry.message.startswith("True peak -0.9 dBTP is above -1")
        assert "integrated" not in entry.message.lower()


def test_digital_silence_has_no_loudness(tmp_path):
    entry = run(aeval(tmp_path / "z.wav", "0", seconds=1.0)).check("loudness")
    assert entry.status == "fail"
    assert entry.measured["integrated_lufs"] == -70.0
    assert entry.measured["true_peak_dbtp"] is None  # -inf


# ---------------------------------------------------------------- clipping


def test_minus_six_dbfs_sine_has_no_clipped_samples(tmp_path):
    entry = run(sine(tmp_path / "s.wav", -6.0)).check("clipping")
    assert entry.status == "pass"
    assert entry.measured["clipped_samples"] == 0
    assert entry.measured["sample_peak_dbfs"] == pytest.approx(-6.0, abs=0.01)
    assert entry.measured["method"] == "sample peak below the clip level"


def test_clipped_square_fails_with_every_sample_counted(tmp_path):
    entry = run(square(tmp_path / "sq.wav", seconds=1.0, channels=1)).check("clipping")
    assert entry.status == "fail"
    assert entry.measured["clipped_samples"] == 48000
    assert entry.measured["first_clip_s"] == 0.0
    assert "48000 clipped sample(s) (first at 0:00.00)" in entry.message


@pytest.mark.parametrize(("peak", "status"), [(-0.02, "pass"), (0.0, "fail")])
def test_clip_level_boundary(tmp_path, peak, status):
    entry = run(sine(tmp_path / "s.wav", peak, seconds=0.5)).check("clipping")
    assert entry.status == status
    if status == "fail":
        assert entry.measured["clipped_samples"] > 0
        assert entry.measured["method"] == "decoded float samples counted"


def test_nan_samples_fail_clipping(tmp_path):
    path = aeval(tmp_path / "nan.wav", "if(lt(t\\,0.25)\\,0/0\\,0.1)", seconds=1.0)
    entry = run(path).check("clipping")
    assert entry.status == "fail"
    assert entry.measured["non_finite_per_channel"] == 12000


# ---------------------------------------------------------------- size cap


def test_size_cap_applies_to_review_renders_only(tmp_path):
    path = render(tmp_path / "r.mp4", frames=30)
    size = path.stat().st_size
    at_cap = run(path, kind="review", profile=profile_with(review_mp4_max_mb=size / 1e6))
    entry = at_cap.check("size-cap")
    assert entry.status == "pass"
    assert entry.measured["size_bytes"] == size
    over = run(path, kind="review", profile=profile_with(review_mp4_max_mb=(size - 1) / 1e6))
    assert over.check("size-cap").status == "fail"
    assert "over the" in over.check("size-cap").message
    short = run(path, kind="short").check("size-cap")
    assert short.status == "skipped" and short.skip_reason == "not-applicable"


# ---------------------------------------------------------------- full decode


def test_clean_render_decodes(tmp_path):
    entry = run(render(tmp_path / "r.mp4", frames=30)).check("decode")
    assert entry.status == "pass"
    assert entry.measured == {
        "readable": True,
        "exit_code": 0,
        "error_lines": 0,
        "errors": [],
        "log_truncated": False,
    }


def test_truncated_render_fails_decode(tmp_path):
    whole = render(tmp_path / "r.mp4", frames=60)
    report = run(truncate(whole, tmp_path / "cut.mp4"))
    entry = report.check("decode")
    assert entry.status == "fail"
    assert entry.measured["readable"] is True  # moov is at the front, the data is cut
    assert entry.measured["error_lines"] >= 1 and entry.measured["exit_code"] != 0
    assert not report.ok


def test_render_without_moov_is_unreadable(tmp_path):
    whole = render(tmp_path / "r.mp4", frames=60, faststart=False)
    report = run(truncate(whole, tmp_path / "cut.mp4"))
    decode = report.check("decode")
    assert decode.status == "fail" and decode.measured["readable"] is False
    assert "moov atom not found" in decode.message or "Invalid data" in decode.message
    for check_id in ("loudness", "clipping", "longest-silence", "size-cap", "duration"):
        assert report.check(check_id).skip_reason == "render-unreadable"
    assert report.status == "fail"


# ---------------------------------------------------------------- longest silence


def test_longest_silence_is_reported(tmp_path):
    path = gapped(tmp_path / "g.wav", before_s=0.5, gap_s=1.0, after_s=0.5, peak_dbfs=-14.5)
    entry = run(path).check("longest-silence")
    assert entry.status == "pass"
    assert entry.measured["longest_silence_s"] == pytest.approx(1.0, abs=0.005)
    assert entry.measured["start_s"] == pytest.approx(0.5, abs=0.005)
    assert entry.target == {"silence_db": -42.0, "min_s": 0.2}
    assert "1.00 s at 0:00.50" in entry.message


def test_longest_silence_warns_only_when_asked(tmp_path):
    path = gapped(tmp_path / "g.wav", before_s=0.5, gap_s=1.0, after_s=0.5, peak_dbfs=-14.5)
    entry = run(path, profile=profile_with(longest_silence_warn_s=0.5)).check("longest-silence")
    assert entry.status == "warn"
    assert entry.target["warn_s"] == 0.5
    assert "longer than 0.5 s" in entry.message
    longer = run(path, profile=profile_with(longest_silence_warn_s=1.5))
    assert longer.check("longest-silence").status == "pass"


def test_venue_silence_threshold(tmp_path):
    path = gapped(
        tmp_path / "v.wav", before_s=0.5, gap_s=1.0, after_s=0.5, peak_dbfs=-10, gap_dbfs=-39
    )
    house = run(path).check("longest-silence")
    assert house.measured["longest_silence_s"] == 0.0
    venue = run(path, silence_db=-36).check("longest-silence")
    assert venue.target["silence_db"] == -36
    assert venue.measured["longest_silence_s"] == pytest.approx(1.0, abs=0.005)


def test_trailing_silence_counts_to_the_end(tmp_path):
    path = gapped(tmp_path / "t.wav", before_s=0.5, gap_s=0.8, after_s=0.0, peak_dbfs=-14.5)
    entry = run(path).check("longest-silence")
    assert entry.measured["longest_silence_s"] == pytest.approx(0.8, abs=0.005)
    assert entry.measured["end_s"] == pytest.approx(1.3, abs=0.005)


# ---------------------------------------------------------------- duration


def _duration_plan(frames: int) -> EditPlan:
    return plan_of(
        plan_dict(
            [{"id": "a", "media": "unused.mov"}],
            [{"track": 1, "source": "a", "src_in": 0, "src_out": frames}],
            fps="30000/1001",
        )
    )


@pytest.mark.parametrize(
    ("frames", "diff", "status"),
    [(30, 0, "pass"), (31, 1, "pass"), (32, 2, "fail"), (28, -2, "fail")],
)
def test_duration_within_one_frame(tmp_path, frames, diff, status):
    path = render(tmp_path / "r.mp4", frames=frames)
    entry = run(path, plan=_duration_plan(30)).check("duration")
    assert entry.status == status
    assert entry.measured["diff_frames"] == pytest.approx(diff, abs=0.01)
    assert entry.measured["measured_from"] == "video stream"
    assert entry.target["plan_frames"] == 30
    if status == "fail":
        assert ("longer" if diff > 0 else "shorter") in entry.message


def test_audio_only_render_duration_comes_from_the_audio(tmp_path):
    plan = plan_of(
        plan_dict(
            [{"id": "a", "media": "unused.wav"}],
            [{"track": 1, "source": "a", "src_in": 0, "src_out": 100}],
        )
    )
    entry = run(sine(tmp_path / "s.wav", -14.5, seconds=2.0), plan=plan).check("duration")
    assert entry.status == "pass"
    assert entry.measured["measured_from"] == "audio stream"
    assert entry.measured["render_s"] == pytest.approx(2.0, abs=1e-6)


# ---------------------------------------------------------------- join source-audio quiet


def _join_plan(
    tmp_path: Path, incoming: Path, src_in: int, *, src_out: int | None = None, fps: str = "50"
) -> EditPlan:
    """Two sources of different recordings, so the join between them is a cut."""
    other = sine(tmp_path / "other.wav", -14.5, seconds=1.0)
    return plan_of(
        plan_dict(
            [{"id": "o", "media": str(other)}, {"id": "a", "media": incoming.name}],
            [
                {"track": 1, "source": "o", "src_in": 0, "src_out": 10},
                {
                    "track": 1,
                    "source": "a",
                    "src_in": src_in,
                    "src_out": src_out if src_out is not None else src_in + 20,
                },
            ],
            fps=fps,
        )
    )


def quiet_join(tmp_path: Path, plan: EditPlan, **kw: Any) -> dict[str, Any]:
    """Run QA with ``plan`` and return the join-quiet check plus its single join's detail."""
    entry = run(sine(tmp_path / "r.wav", -14.5), plan=plan, media_root=tmp_path, **kw).check(
        "join-quiet"
    )
    assert len(entry.details) == 1
    return {"check": entry, **entry.details[0]}


@pytest.mark.parametrize(("gap_s", "status"), [(0.060, "pass"), (0.059, "fail")])
def test_join_quiet_60_versus_59_ms(tmp_path, gap_s, status):
    source = gapped(tmp_path / "a.wav", before_s=0.5, gap_s=gap_s, after_s=0.5)
    join = quiet_join(tmp_path, _join_plan(tmp_path, source, src_in=26))  # 0.52 s, in the gap
    assert join["status"] == join["check"].status == status
    assert join["quiet_ms_at_cut"] == pytest.approx(gap_s * 1000, abs=0.25)
    assert join["run_at_cut"]["start_s"] == pytest.approx(0.5, abs=0.0005)
    assert join["window_s"] == [pytest.approx(0.42), pytest.approx(0.62)]
    assert join["quietest_dbfs"] is None  # digital silence: -inf
    assert join["record_s"] == 0.2 and join["src_in_s"] == 0.52 and join["src_in"] == 26
    assert join["check"].target == {
        "quiet_db": -40.0,
        "min_ms": 60.0,
        "slack_ms": 10.0,
        "search_s": 0.1,
        "window_ms": 10.0,
    }
    if status == "fail":
        assert join["nearest_pause"] is None
        assert "the pause at src_in is only 59 ms; no 60 ms pause" in join["hint"]


def test_a_pause_that_ends_before_src_in_fails_with_a_suggestion(tmp_path):
    # Pause 0.45-0.57 s, src_in 0.60 s (frame 30 at 50 fps): 70 ms of the pause lies in the
    # ±0.1 s window, so the old "a run anywhere in the window" rule passed this cut, yet the
    # incoming clip starts 30 ms after the pause, in the sound.
    source = gapped(tmp_path / "a.wav", before_s=0.45, gap_s=0.12, after_s=0.5)
    join = quiet_join(tmp_path, _join_plan(tmp_path, source, src_in=30))
    assert join["status"] == "fail"
    assert join["quiet_ms_at_cut"] == 0.0 and join["run_at_cut"] is None
    assert join["longest_quiet_ms"] == pytest.approx(70, abs=0.25)  # cut short by the window
    assert join["nearest_pause"]["end_s"] == pytest.approx(0.57, abs=0.0005)
    assert join["suggested_src_in"] == 27  # midpoint of the in-window part, 0.535 s
    assert join["suggested_src_in_s"] == 0.54
    assert "move src_in of join 1 to frame 27" in join["check"].message
    assert "1 of 1 cuts do not start in a pause of 60 ms below -40 dBFS" in join["check"].message


def test_a_pause_80_ms_before_src_in(tmp_path):
    # Pause 0.40-0.52 s, src_in 0.60 s: only 20 ms of it is within ±0.1 s, so the default
    # search finds nothing to suggest; a ±0.25 s search suggests its middle.
    source = gapped(tmp_path / "a.wav", before_s=0.40, gap_s=0.12, after_s=0.5)
    plan = _join_plan(tmp_path, source, src_in=30)
    default = quiet_join(tmp_path, plan)
    assert default["status"] == "fail" and default["nearest_pause"] is None
    assert "no 60 ms pause within ±0.1 s of src_in" in default["hint"]
    assert "suggested_src_in" not in default
    wide = quiet_join(tmp_path, plan, profile=profile_with(join_quiet_search_s=0.25))
    assert wide["status"] == "fail"
    assert wide["nearest_pause"]["ms"] == pytest.approx(120, abs=0.25)
    assert wide["suggested_src_in"] == 23  # 0.46 s
    assert "move src_in of join 1 to frame 23" in wide["hint"]


def test_a_pause_containing_src_in_passes(tmp_path):
    source = gapped(tmp_path / "a.wav", before_s=0.5, gap_s=0.1, after_s=0.5)
    join = quiet_join(tmp_path, _join_plan(tmp_path, source, src_in=27))  # 0.54 s
    assert join["status"] == "pass"
    assert join["quiet_ms_at_cut"] == pytest.approx(100, abs=0.25)
    assert join["nearest_pause"] is None and "hint" not in join
    assert "All 1 cuts start in a pause" in join["check"].message


@pytest.mark.parametrize(
    ("pause_start_s", "status"),
    [
        (0.491, "pass"),  # pause ends 9 ms before src_in 0.60 s
        (0.489, "fail"),  # ... 11 ms before
        (0.609, "pass"),  # pause starts 9 ms after src_in
        (0.611, "fail"),  # ... 11 ms after
    ],
)
def test_join_quiet_slack_either_side_of_the_run(tmp_path, pause_start_s, status):
    source = gapped(tmp_path / "a.wav", before_s=pause_start_s, gap_s=0.1, after_s=0.5)
    join = quiet_join(tmp_path, _join_plan(tmp_path, source, src_in=30))
    assert join["status"] == status
    if status == "pass":
        # The 100 ms pause is clipped by the window [0.5, 0.7] to 91 ms.
        assert join["run_at_cut"]["ms"] == pytest.approx(91, abs=0.25)
    else:
        assert join["run_at_cut"] is None
        mid = pause_start_s + 0.05
        assert join["suggested_src_in"] == round(mid * 50)
    slack0 = quiet_join(
        tmp_path,
        _join_plan(tmp_path, source, src_in=30),
        profile=profile_with(join_quiet_slack_ms=0),
    )
    assert slack0["status"] == "fail"


@pytest.mark.parametrize(
    ("fps", "src_in", "frame"),
    [("24", 28, 25), ("25", 30, 26), ("30000/1001", 35, 31)],
)
def test_suggested_src_in_is_the_frame_nearest_the_pause_middle(tmp_path, fps, src_in, frame):
    # Pause 1.00-1.10 s (middle 1.05 s); src_in sits 67-100 ms after it.
    source = gapped(tmp_path / "a.wav", before_s=1.0, gap_s=0.1, after_s=0.5)
    plan = _join_plan(tmp_path, source, src_in=src_in, fps=fps)
    join = quiet_join(tmp_path, plan, profile=profile_with(join_quiet_search_s=0.25))
    assert join["status"] == "fail"
    assert join["suggested_src_in"] == frame == plan.fps.frames(1.05)
    assert 1.0 < join["suggested_src_in_s"] < 1.1  # the suggestion lands inside the pause
    assert join["suggested_src_in_s"] == round(plan.fps.seconds(frame), 4)
    assert f"move src_in of join 1 to frame {frame}" in join["check"].message


def test_join_quiet_without_a_pause_says_so(tmp_path):
    source = sine(tmp_path / "a.wav", -6.0, seconds=1.0)
    join = quiet_join(tmp_path, _join_plan(tmp_path, source, src_in=25))
    assert join["status"] == "fail"
    assert join["longest_quiet_ms"] == 0.0
    assert join["quietest_dbfs"] == pytest.approx(-9.0, abs=0.5)
    assert "no 60 ms pause within ±0.1 s of src_in, so re-cut it by ear" in join["hint"]


@pytest.mark.parametrize(("lead_silence", "status"), [(True, "pass"), (False, "fail")])
def test_join_at_the_very_start_of_a_source(tmp_path, lead_silence, status):
    if lead_silence:
        source = gapped(tmp_path / "a.wav", before_s=0.0, gap_s=0.07, after_s=1.0)
    else:
        source = sine(tmp_path / "a.wav", -6.0, seconds=1.0)
    join = quiet_join(tmp_path, _join_plan(tmp_path, source, src_in=0, src_out=50))
    assert join["status"] == status
    assert join["window_s"] == [0.0, pytest.approx(0.1)]
    if lead_silence:
        assert join["quiet_ms_at_cut"] == pytest.approx(70, abs=0.25)
        assert join["run_at_cut"]["start_s"] == 0.0
    else:
        assert join["quiet_ms_at_cut"] < 1


def test_join_quiet_reads_room_tone_as_quiet(tmp_path):
    # A -46 dBFS tone (RMS -49) is below -40 dBFS; at -30 dBFS it is not.
    quiet = gapped(
        tmp_path / "a.wav", before_s=0.5, gap_s=0.1, after_s=0.5, peak_dbfs=-6, gap_dbfs=-46
    )
    join = quiet_join(tmp_path, _join_plan(tmp_path, quiet, src_in=27))
    assert join["status"] == "pass"
    assert join["run_at_cut"]["level_dbfs"] == pytest.approx(-49.0, abs=0.5)
    loud = gapped(
        tmp_path / "b.wav", before_s=0.5, gap_s=0.1, after_s=0.5, peak_dbfs=-6, gap_dbfs=-30
    )
    join = quiet_join(tmp_path, _join_plan(tmp_path, loud, src_in=27))
    assert join["status"] == "fail"
    assert join["quietest_dbfs"] == pytest.approx(-33.0, abs=0.5)


def test_join_quiet_continuous_joins_are_not_cuts(tmp_path):
    source = sine(tmp_path / "a.wav", -6.0, seconds=2.0)
    plan = plan_of(
        plan_dict(
            [{"id": "a", "media": str(source)}],
            [
                {"track": 1, "source": "a", "src_in": 0, "src_out": 40},
                {"track": 1, "source": "a", "src_in": 41, "src_out": 80},  # within 1 frame
            ],
        )
    )
    entry = run(sine(tmp_path / "r.wav", -14.5), plan=plan).check("join-quiet")
    assert entry.status == "pass" and entry.measured == {"cuts": 0}
    assert not entry.incomplete


def test_join_quiet_with_no_readable_source_warns_and_is_listed_unchecked(tmp_path):
    plan = plan_of(
        plan_dict(
            [
                {"id": "o", "media": "missing-o.wav"},
                {"id": "a", "media": "missing-a.wav"},
                {"id": "u", "media": "https://example.invalid/x.wav"},
            ],
            [
                {"track": 1, "source": "o", "src_in": 0, "src_out": 25},
                {"track": 1, "source": "a", "src_in": 10, "src_out": 35},
                {"track": 1, "source": "u", "src_in": 10, "src_out": 35},
            ],
        )
    )
    # The render matches the plan's 1.5 s, so join-quiet is the only check that is not a pass.
    report = run(sine(tmp_path / "r.wav", -14.5, seconds=1.5), plan=plan, media_root=tmp_path)
    entry = report.check("join-quiet")
    assert entry.status == "warn" and entry.skip_reason is None
    assert entry.incomplete
    assert entry.measured == {
        "cuts": 2,
        "failed": 0,
        "unchecked": 2,
        "unchecked_reason": "source-unreadable",
    }
    assert entry.message.startswith("None of the 2 cuts could be checked (")
    assert "make the plan's source media readable" in entry.message
    reasons = [d["reason"] for d in entry.details]
    assert "file not found" in reasons[0]
    assert "not a local path" in reasons[1]
    assert [d["status"] for d in entry.details] == ["unchecked", "unchecked"]
    assert report.status == "warn" and report.ok
    assert report.check("duration").status == "pass"
    assert report.unchecked == [
        "size-cap",
        "join-quiet",
        "join-retranscribe",
        "caption-coverage",
    ]


def test_join_quiet_some_sources_unreadable_warns(tmp_path):
    good = gapped(tmp_path / "a.wav", before_s=0.5, gap_s=0.1, after_s=0.5)
    video_only = render(tmp_path / "v.mp4", frames=60, audio="none")
    plan = plan_of(
        plan_dict(
            [
                {"id": "o", "media": "missing.wav"},
                {"id": "a", "media": good.name},
                {"id": "v", "media": video_only.name},
            ],
            [
                {"track": 1, "source": "o", "src_in": 0, "src_out": 25},
                {"track": 1, "source": "a", "src_in": 27, "src_out": 50},
                {"track": 1, "source": "v", "src_in": 0, "src_out": 25},
            ],
        )
    )
    report = run(sine(tmp_path / "r.wav", -14.5), plan=plan, media_root=tmp_path)
    entry = report.check("join-quiet")
    assert entry.status == "warn" and entry.incomplete
    assert [d["status"] for d in entry.details] == ["pass", "unchecked"]
    assert "no audio stream" in entry.details[1]["reason"]
    assert entry.measured == {
        "cuts": 2,
        "failed": 0,
        "unchecked": 1,
        "unchecked_reason": "source-unreadable",
    }
    assert "join-quiet" in report.unchecked


def test_join_quiet_failures_outrank_unreadable_sources(tmp_path):
    loud = sine(tmp_path / "a.wav", -6.0, seconds=1.0)
    plan = plan_of(
        plan_dict(
            [{"id": "o", "media": "missing.wav"}, {"id": "a", "media": loud.name}],
            [
                {"track": 1, "source": "o", "src_in": 0, "src_out": 25},
                {"track": 1, "source": "a", "src_in": 25, "src_out": 40},
                {"track": 1, "source": "o", "src_in": 30, "src_out": 40},
            ],
        )
    )
    entry = run(sine(tmp_path / "r.wav", -14.5), plan=plan, media_root=tmp_path).check("join-quiet")
    assert entry.status == "fail" and entry.incomplete
    assert "1 more could not be checked" in entry.message


@posix_only
def test_join_quiet_source_read_timeout_is_unchecked_not_failed(tmp_path, monkeypatch):
    # The source pauses at the cut, so a completed read would pass. Only the windowed source
    # read (the one ffmpeg call with -ss) stalls; every other call runs the real ffmpeg.
    source = gapped(tmp_path / "a.wav", before_s=0.5, gap_s=0.1, after_s=0.5)
    plan = _join_plan(tmp_path, source, src_in=27)
    render_file = sine(tmp_path / "r.wav", -14.5, seconds=0.6)
    stall = 'case " $* " in *" -ss "*) sleep 30;; esac\nexec "' + str(FFMPEG) + '" "$@"'
    monkeypatch.setenv("AAR_FFMPEG", str(fake_tool(tmp_path, "ffmpeg", stall)))
    started = time.monotonic()
    report = run(render_file, plan=plan, media_root=tmp_path, timeout_s=3.0)
    assert time.monotonic() - started < 20
    entry = report.check("join-quiet")
    [join] = entry.details
    assert join["status"] == "unchecked"
    assert "timed out after 3 s" in join["reason"]
    assert "hint" not in join
    assert entry.status == "warn" and entry.incomplete
    assert entry.measured == {
        "cuts": 1,
        "failed": 0,
        "unchecked": 1,
        "unchecked_reason": "source-unreadable",
    }
    assert "join-quiet" in report.unchecked
    assert report.status != "pass"
    assert report.check("decode").status == "pass"  # the render's own calls were not stalled


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (MediaTimeout("ffmpeg timed out after 7 s", timeout_s=7.0, tool="ffmpeg"), "unchecked"),
        (MediaError("ffmpeg exited with status 1", tool="ffmpeg", exit_code=1), "fail"),
        (OutputTooLarge("ffmpeg wrote more than 10 bytes", tool="ffmpeg"), "fail"),
    ],
)
def test_join_quiet_only_a_read_timeout_is_unchecked(tmp_path, monkeypatch, error, status):
    def broken(*args: Any, **kwargs: Any) -> None:
        raise error

    source = gapped(tmp_path / "a.wav", before_s=0.5, gap_s=0.1, after_s=0.5)
    # aar.qa re-exports check(), which shadows the module name, so patch the module itself.
    monkeypatch.setattr(importlib.import_module("aar.qa.check"), "read_pcm", broken)
    report = run(
        sine(tmp_path / "r.wav", -14.5),
        plan=_join_plan(tmp_path, source, src_in=27),
        media_root=tmp_path,
    )
    entry = report.check("join-quiet")
    [join] = entry.details
    assert join["status"] == status
    assert report.status != "pass" and entry.status != "pass"
    if status == "unchecked":
        assert join["reason"].startswith("reading the source audio timed out after 7 s")
        assert entry.status == "warn" and entry.incomplete
        assert "join-quiet" in report.unchecked
    else:
        assert join["reason"].startswith("could not read the source audio: ")
        assert entry.status == "fail" and not entry.incomplete
        assert join["hint"] == f"join 1: {join['reason']}"


# ---------------------------------------------------------------- join re-transcription


class FakeTranscriber:
    """Returns the true render words inside the window, with optional damage per call."""

    def __init__(self, truth: list[Word], damage: dict[int, Any] | None = None) -> None:
        self.truth = truth
        self.damage = damage or {}
        self.calls: list[tuple[Path, float, float]] = []

    def transcribe(self, path: Path, start_s: float, end_s: float) -> Sequence[Word]:
        index = len(self.calls)
        self.calls.append((path, start_s, end_s))
        words = [w for w in self.truth if start_s <= w.start_s and w.end_s <= end_s]
        hurt = self.damage.get(index)
        if callable(hurt):
            return hurt(words)
        return words


def _speech(tmp_path: Path, *, cut_through_word: bool = False):
    """A 6 s talk transcript; the plan keeps 0-2 s and 4-6 s (fps 50), one cut at 2.0 s."""
    names = ["zero", "one", "two", "three", "four", "five"]
    names += ["six", "seven", "eight", "nine", "ten", "eleven"]
    words = [
        Word(text=f"{names[i % 12]}{i // 12}", start_s=i * 0.25, end_s=i * 0.25 + 0.2)
        for i in range(24)
    ]
    if cut_through_word:
        words[7] = Word(text="sliced", start_s=1.9, end_s=2.15)
    transcript = Transcript(recording="rec", words=words)
    source = sine(tmp_path / "talk.wav", -14.5, seconds=6.0)
    plan = plan_of(
        plan_dict(
            [{"id": "a", "media": str(source), "recording_id": "rec"}],
            [
                {"track": 1, "source": "a", "src_in": 0, "src_out": 100},
                {"track": 1, "source": "a", "src_in": 200, "src_out": 300},
            ],
        )
    )
    truth = [
        Word(text=k.text, start_s=k.record_start_s, end_s=k.record_end_s)
        for k in kept_words(plan, transcript)
    ]
    rendered = sine(tmp_path / "render.wav", -14.5, seconds=4.0)
    return plan, transcript, truth, rendered


def test_retranscribe_agreeing_windows_pass(tmp_path):
    plan, transcript, truth, rendered = _speech(tmp_path)
    fake = FakeTranscriber(truth)
    entry = run(rendered, plan=plan, transcript=transcript, transcriber=fake).check(
        "join-retranscribe"
    )
    assert entry.status == "pass"
    (join,) = entry.details
    # Windows [J-2.5, J+1.5], [J-2, J+2], [J-1.5, J+2.5] clipped to the 4 s render; the
    # compared region is their overlap [J-1.5, J+1.5] less 0.25 s at each end.
    assert [(round(s, 3), round(e, 3)) for _, s, e in fake.calls] == [
        (0.0, 3.5),
        (0.0, 4.0),
        (0.5, 4.0),
    ]
    assert join["region_s"] == [0.75, 3.25]
    assert join["min_ratio"] == 1.0
    assert join["pair_ratios"] == [1.0, 1.0, 1.0]
    expected = join["expected"].split()
    assert len(expected) == 10
    assert expected[:2] == ["three0", "four0"] and expected[-1] == "eight1"
    assert all(call[0] == rendered for call in fake.calls)


def test_retranscribe_one_dropped_word_against_the_cutoff(tmp_path):
    plan, transcript, truth, rendered = _speech(tmp_path)

    def drop_one(words):
        return [w for w in words if w.text != "six0"]

    fake = FakeTranscriber(truth, damage={1: drop_one})
    loose = run(rendered, plan=plan, transcript=transcript, transcriber=fake)
    join = loose.check("join-retranscribe").details[0]
    assert loose.check("join-retranscribe").status == "pass"
    # 10 expected tokens versus 9: ratio 2*9/19.
    assert join["min_ratio"] == pytest.approx(18 / 19, abs=0.001)
    strict = run(
        rendered,
        plan=plan,
        transcript=transcript,
        transcriber=FakeTranscriber(truth, damage={1: drop_one}),
        profile=profile_with(join_agreement_min=0.97),
    ).check("join-retranscribe")
    assert strict.status == "fail"
    assert "agreement 0.95 is below 0.97" in strict.details[0]["reason"]


def test_retranscribe_garbled_window_fails(tmp_path):
    plan, transcript, truth, rendered = _speech(tmp_path)

    def garble(words):
        return [Word(text="mumble", start_s=w.start_s, end_s=w.end_s) for w in words]

    fake = FakeTranscriber(truth, damage={2: garble})
    entry = run(rendered, plan=plan, transcript=transcript, transcriber=fake).check(
        "join-retranscribe"
    )
    assert entry.status == "fail"
    join = entry.details[0]
    assert join["windows"][2]["ratio_vs_plan"] == 0.0
    assert join["pair_ratios"][0] == 1.0  # windows 0 and 1 still agree
    assert "do not re-transcribe consistently" in entry.message


def test_retranscribe_transcriber_crash_fails_the_join(tmp_path):
    plan, transcript, truth, rendered = _speech(tmp_path)

    def boom(_words):
        raise RuntimeError("model went away")

    fake = FakeTranscriber(truth, damage={0: boom})
    entry = run(rendered, plan=plan, transcript=transcript, transcriber=fake).check(
        "join-retranscribe"
    )
    assert entry.status == "fail"
    assert "the transcriber failed (RuntimeError: model went away)" in entry.details[0]["reason"]


def test_retranscribe_flags_a_word_the_cut_slices(tmp_path):
    plan, transcript, truth, rendered = _speech(tmp_path, cut_through_word=True)
    entry = run(
        rendered, plan=plan, transcript=transcript, transcriber=FakeTranscriber(truth)
    ).check("join-retranscribe")
    assert entry.status == "fail"
    assert entry.details[0]["straddled"] == [
        {"side": "out", "word": "sliced", "start_s": 1.9, "end_s": 2.15}
    ]


def test_retranscribe_two_windows(tmp_path):
    plan, transcript, truth, rendered = _speech(tmp_path)
    fake = FakeTranscriber(truth)
    profile = profile_with(join_windows=2)
    entry = run(
        rendered, plan=plan, transcript=transcript, transcriber=fake, profile=profile
    ).check("join-retranscribe")
    assert entry.status == "pass"
    assert len(fake.calls) == 2
    assert entry.details[0]["region_s"] == [0.5, 3.5]


def test_retranscribe_skips_say_what_is_missing(tmp_path):
    plan, transcript, truth, rendered = _speech(tmp_path)
    fake = FakeTranscriber(truth)
    assert run(rendered, transcriber=fake).check("join-retranscribe").skip_reason == "no-plan"
    no_words = run(rendered, plan=plan, transcriber=fake).check("join-retranscribe")
    assert no_words.skip_reason == "no-transcript"
    no_model = run(rendered, plan=plan, transcript=transcript).check("join-retranscribe")
    assert no_model.skip_reason == "no-transcriber"
    assert no_model.message == "Skipped: no transcriber was given."
    assert fake.calls == []


# ---------------------------------------------------------------- caption coverage


def _captioned(tmp_path: Path, captions: list[dict[str, Any]], kind: str = "short"):
    plan, transcript, _, rendered = _speech(tmp_path)
    data = plan.model_dump(mode="json")
    data["captions"] = captions
    data["kind"] = kind
    return plan_of(data), transcript, rendered


def test_every_kept_word_captioned_passes(tmp_path):
    plan, transcript, rendered = _captioned(
        tmp_path, [{"lines": ["ALL"], "record_in": 0, "record_out": 200}]
    )
    entry = run(rendered, plan=plan, transcript=transcript).check("caption-coverage")
    assert entry.status == "pass"
    assert entry.measured == {
        "kept_words": 16,
        "captions": 1,
        "uncovered": 0,
        "segments": 2,
        "segments_unchecked": 0,
    }
    assert not entry.incomplete


def test_uncovered_words_are_listed(tmp_path):
    # Captions stop at record 3.0 s: the kept words whose midpoint is later are uncovered.
    plan, transcript, rendered = _captioned(
        tmp_path,
        [
            {"lines": ["A"], "record_in": 0, "record_out": 100},
            {"lines": ["B"], "record_in": 100, "record_out": 150},
        ],
    )
    entry = run(rendered, plan=plan, transcript=transcript).check("caption-coverage")
    assert entry.status == "fail"
    assert entry.measured == {
        "kept_words": 16,
        "captions": 2,
        "uncovered": 4,
        "segments": 2,
        "segments_unchecked": 0,
    }
    assert [d["word"] for d in entry.details] == ["eight1", "nine1", "ten1", "eleven1"]
    assert entry.details[0]["record_start_s"] == pytest.approx(3.0)
    assert "4 of 16 kept words have no caption" in entry.message


def test_caption_gaps_between_captions_count(tmp_path):
    plan, transcript, rendered = _captioned(
        tmp_path,
        [
            {"lines": ["A"], "record_in": 0, "record_out": 60},
            {"lines": ["B"], "record_in": 70, "record_out": 200},
        ],
    )
    entry = run(rendered, plan=plan, transcript=transcript).check("caption-coverage")
    # Captions cover [0, 1.2) and [1.4, 4.0) s; five0 (1.25-1.45 s, midpoint 1.35) falls between.
    assert [d["word"] for d in entry.details] == ["five0"]


def test_short_without_captions_fails_longform_is_skipped(tmp_path):
    plan, transcript, rendered = _captioned(tmp_path, [], kind="short")
    short = run(rendered, plan=plan, transcript=transcript).check("caption-coverage")
    assert short.status == "fail" and short.measured["uncovered"] == 16
    plan, transcript, rendered = _captioned(tmp_path, [], kind="longform")
    long = run(rendered, kind="longform", plan=plan, transcript=transcript).check(
        "caption-coverage"
    )
    assert long.status == "skipped" and long.skip_reason == "no-captions"
    no_words = run(rendered, plan=plan).check("caption-coverage")
    assert no_words.skip_reason == "no-transcript"


# ---------------------------------------------------------------- hostile inputs


@pytest.mark.parametrize("name", ["-starts-with-dash.wav", "space and ünïcødé ✓.wav"])
def test_hostile_names_check_cleanly(tmp_path, name):
    report = run(sine(tmp_path / name, -14.5))
    assert report.status == "pass"
    assert report.file.endswith(name)


def test_relative_dash_name(tmp_path, monkeypatch):
    sine(tmp_path / "-i.wav", -14.5)
    monkeypatch.chdir(tmp_path)
    assert run("-i.wav").status == "pass"


@pytest.mark.parametrize(
    ("content", "message"),
    [(b"", "Empty file"), (b"not media at all\n", "Not readable media"), (None, "File not found")],
)
def test_unreadable_renders_fail_and_skip_the_rest(tmp_path, content, message):
    path = tmp_path / "bad.mp4"
    if content is not None:
        path.write_bytes(content)
    report = run(path)
    decode = report.check("decode")
    assert decode.status == "fail"
    assert message.lower() in decode.message.lower()
    assert report.status == "fail" and not report.ok
    assert report.media is None
    assert report.counts["pass"] == 0


def test_render_without_audio_fails_loudness(tmp_path):
    report = run(render(tmp_path / "v.mp4", frames=30, audio="none"))
    assert report.check("decode").status == "pass"
    loud = report.check("loudness")
    assert loud.status == "fail" and "no audio stream" in loud.message
    assert report.check("clipping").skip_reason == "no-audio"
    assert report.check("longest-silence").skip_reason == "no-audio"
    assert not report.ok


def test_fifo_does_not_hang(tmp_path):
    if not hasattr(os, "mkfifo"):
        pytest.skip("no FIFOs here")
    fifo = tmp_path / "render.mp4"
    os.mkfifo(fifo)
    started = time.monotonic()
    report = run(fifo)
    assert time.monotonic() - started < 3
    assert "not a regular file" in report.check("decode").message.lower()


@posix_only
def test_stalled_ffprobe_times_out_into_a_failure(tmp_path, monkeypatch):
    path = sine(tmp_path / "s.wav", -14.5)
    monkeypatch.setenv("AAR_FFPROBE", str(fake_tool(tmp_path, "ffprobe", "sleep 30")))
    started = time.monotonic()
    report = run(path, timeout_s=0.5)
    assert time.monotonic() - started < 5
    decode = report.check("decode")
    assert decode.status == "fail" and "timed out after 0.5 s" in decode.message
    assert report.check("loudness").skip_reason == "render-unreadable"


@posix_only
def test_stalled_ffmpeg_times_out_into_failures(tmp_path, monkeypatch):
    path = sine(tmp_path / "s.wav", -14.5)
    monkeypatch.setenv("AAR_FFMPEG", str(fake_tool(tmp_path, "ffmpeg", "sleep 30")))
    started = time.monotonic()
    report = run(path, timeout_s=0.5)
    assert time.monotonic() - started < 10
    for check_id in ("decode", "loudness", "clipping", "longest-silence"):
        entry = report.check(check_id)
        assert entry.status == "fail", check_id
        assert "timed out" in entry.message, check_id


def test_missing_tools_raise_instead_of_reporting(tmp_path, monkeypatch):
    monkeypatch.setenv("AAR_FFMPEG", str(tmp_path / "nope"))
    with pytest.raises(MediaToolMissing):
        run(tmp_path / "anything.wav")


def test_bad_arguments(tmp_path):
    path = tmp_path / "x.wav"
    with pytest.raises(ValueError):
        check(path, kind="master", profile=CCN)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        check(path, kind="short", profile=CCN, silence_db=0)
    with pytest.raises(ValueError):
        check(path, kind="short", profile=CCN, timeout_s=0)


def test_a_cut_past_the_end_of_its_source_fails(tmp_path):
    source = sine(tmp_path / "a.wav", -6.0, seconds=1.0)
    plan = _join_plan(tmp_path, source, src_in=60, src_out=80)  # 1.2 s into a 1.0 s file
    entry = run(sine(tmp_path / "r.wav", -14.5), plan=plan, media_root=tmp_path).check("join-quiet")
    assert entry.status == "fail"
    assert entry.details[0]["reason"] == "src_in is at or past the end of the source audio"


@pytest.mark.skipif(os.name != "posix", reason="POSIX file names may hold a newline")
def test_messages_stay_on_one_line_for_hostile_names(tmp_path):
    good = run(sine(tmp_path / "two\nlines.wav", -14.5))
    assert good.status == "pass"
    missing = run(tmp_path / "gone\nfile.wav")
    message = missing.check("decode").message
    assert "\n" not in message and "gone file.wav" in message


# ---------------------------------------------------------------- not checked is not all clear


def test_a_transcript_of_another_recording_checks_no_join_and_warns(tmp_path):
    plan, transcript, truth, rendered = _speech(tmp_path)
    other = transcript.model_copy(update={"recording": "another-night"})
    fake = FakeTranscriber(truth)
    report = run(rendered, plan=plan, transcript=other, transcriber=fake)
    entry = report.check("join-retranscribe")
    assert entry.status == "warn" and entry.skip_reason is None and entry.incomplete
    assert entry.measured == {
        "cuts": 1,
        "failed": 0,
        "unchecked": 1,
        "unchecked_reason": "transcript-does-not-cover",
    }
    assert "does not cover source 'a'" in entry.details[0]["reason"]
    assert entry.message.startswith("None of the 1 cuts could be checked")
    assert fake.calls == []  # nothing to compare against, so the transcriber is not run
    captions = report.check("caption-coverage")
    assert captions.status == "warn" and captions.incomplete
    assert captions.measured["segments_unchecked"] == 2
    assert "covers none of the plan's 2 segments" in captions.message
    assert {"join-retranscribe", "caption-coverage"} <= set(report.unchecked)
    assert "join-quiet" not in report.unchecked  # it ran in full (and fails: a plain tone)


def test_a_transcript_covering_some_segments_marks_captions_incomplete(tmp_path):
    plan, transcript, _, rendered = _speech(tmp_path)
    data = plan.model_dump(mode="json")
    data["sources"].append({"id": "x", "media": "x.wav", "recording_id": "other"})
    data["segments"][1]["source"] = "x"
    data["captions"] = [{"lines": ["ALL"], "record_in": 0, "record_out": 200}]
    entry = run(rendered, plan=plan_of(data), transcript=transcript).check("caption-coverage")
    assert entry.status == "warn" and entry.incomplete
    assert entry.measured["segments_unchecked"] == 1 and entry.measured["kept_words"] == 8
    assert "1 of 2 segments come from a recording the transcript does not cover" in entry.message


def test_report_unchecked_lists_skips_and_incomplete_checks(tmp_path):
    plan, transcript, truth, rendered = _speech(tmp_path)
    data = plan.model_dump(mode="json")
    data["captions"] = [{"lines": ["ALL"], "record_in": 0, "record_out": 200}]
    full = run(
        rendered,
        plan=plan_of(data),
        transcript=transcript,
        transcriber=FakeTranscriber(truth),
    )
    # Only the size cap (not a review render) is left out; everything else ran in full.
    assert full.unchecked == ["size-cap"]
    assert full.check("join-quiet").status == "fail"  # the fixture's source is a plain tone
    bare = run(rendered)
    assert bare.status == "pass"
    assert bare.unchecked == [
        "size-cap",
        "duration",
        "join-quiet",
        "join-retranscribe",
        "caption-coverage",
    ]
    review = run(render(tmp_path / "v.mp4", frames=30, audio="none"), kind="review")
    assert review.unchecked == [
        "clipping",
        "longest-silence",
        "duration",
        "join-quiet",
        "join-retranscribe",
        "caption-coverage",
    ]


@pytest.mark.parametrize(("render_s", "status", "diff"), [(1.5, "pass", 0), (1.0, "fail", -25)])
def test_a_v2_outro_counts_towards_the_duration_but_is_not_a_cut(tmp_path, render_s, status, diff):
    source = gapped(tmp_path / "a.wav", before_s=0.5, gap_s=0.1, after_s=0.5)
    data = plan_dict(
        [
            {"id": "a", "media": source.name},
            {"id": "out", "media": "outro.mov", "duration": 25},
        ],
        [{"track": 1, "source": "a", "src_in": 0, "src_out": 50}],
    )
    data.update(version=2, outro={"source": "out"})
    plan = plan_of(data)
    assert plan.total_frames() == 75
    report = run(sine(tmp_path / "r.wav", -14.5, seconds=render_s), plan=plan, media_root=tmp_path)
    duration = report.check("duration")
    assert duration.status == status
    assert duration.target["plan_frames"] == 75
    assert duration.measured["diff_frames"] == pytest.approx(diff, abs=0.01)
    # The content -> outro step is the outro's transition, not a content cut.
    assert report.check("join-quiet").measured == {"cuts": 0}
