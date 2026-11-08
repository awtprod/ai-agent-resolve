"""End-to-end QA on generated media with a CCN-shaped plan (S13b F3, adopted from the
validator's point-2 probe).

A1-A4 muted and routed to the A5 bus, A5 the bus, A6 the bounce. The two content segments come
from different recordings, so their join is a real cut. A track carries mixed audio when it is
unmuted OR routed to a bus, so the CCN dialogue join is checked, not silently passed. A plan
whose only cut sits on a muted, unrouted track reports the check as skipped, never pass.
"""

from __future__ import annotations

from pathlib import Path

from media_support import gapped, needs_ffmpeg, plan_dict, sine

from aar.plan.model import EditPlan
from aar.plan.profile import load_profile
from aar.qa import check

pytestmark = needs_ffmpeg
CCN = load_profile("ccn")

CCN_TRACKS = [
    {"name": "A1", "mute": True, "bus": "A5"},
    {"name": "A2", "mute": True, "bus": "A5"},
    {"name": "A3", "mute": True, "bus": "A5"},
    {"name": "A4", "mute": True, "bus": "A5"},
    {"name": "A5", "mute": False},
    {"name": "A6", "mute": False},
]


def _plan(tmp_path: Path, audio_tracks: list[dict]) -> EditPlan:
    other = sine(tmp_path / "other.wav", -14.5, seconds=1.0)
    incoming = gapped(tmp_path / "a.wav", before_s=0.5, gap_s=0.03, after_s=0.5)  # 30 ms: fails
    data = plan_dict(
        [{"id": "o", "media": str(other)}, {"id": "a", "media": incoming.name}],
        [
            {"track": 1, "source": "o", "src_in": 0, "src_out": 10},
            {"track": 1, "source": "a", "src_in": 26, "src_out": 46},
        ],
        fps="50",
    )
    data["version"] = 2
    data["audio_tracks"] = audio_tracks
    return EditPlan.model_validate(data)


def _join(tmp_path: Path, audio_tracks: list[dict]):
    report = check(
        sine(tmp_path / "r.wav", -14.5),
        kind="short",
        profile=CCN,
        plan=_plan(tmp_path, audio_tracks),
        media_root=tmp_path,
    )
    return report, report.check("join-quiet")


def test_control_no_audio_tracks_join_is_checked_and_fails(tmp_path: Path) -> None:
    _, entry = _join(tmp_path, [])
    assert entry.status == "fail" and entry.measured.get("cuts", 1) != 0


def test_ccn_topology_dialogue_join_is_still_checked(tmp_path: Path) -> None:
    _, entry = _join(tmp_path, CCN_TRACKS)
    assert entry.status == "fail", (entry.status, entry.message)


def test_plain_unmuted_unrouted_tracks_join_is_still_checked(tmp_path: Path) -> None:
    _, entry = _join(tmp_path, [{"name": "A1"}, {"name": "A2"}])
    assert entry.status == "fail", (entry.status, entry.message)


def test_all_muted_unrouted_join_is_skipped_not_passed(tmp_path: Path) -> None:
    # The only cut sits on a muted, unrouted, non-bus track: it carries no mixed audio, so the
    # check is skipped with a reason and listed in unchecked -- never a silent pass (F3).
    report, entry = _join(tmp_path, [{"name": "A1", "mute": True}, {"name": "A2", "mute": True}])
    assert entry.status == "skipped"
    assert entry.skip_reason == "no-mixed-audio"
    assert "join-quiet" in report.unchecked
