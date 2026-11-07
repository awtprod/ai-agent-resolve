"""S13b F1 pin: QA join enumeration (``aar.qa.planmap.cuts``) skips joins on a track that
carries no mixed audio -- a muted or unrouted audio track per the plan's ``audio_tracks`` --
because that source audio never reaches the render, so a join-quiet or join-retranscribe check
there would judge audio the viewer never hears. A plan with no ``audio_tracks`` (every v1 plan)
is checked exactly as before.
"""

from __future__ import annotations

from aar.plan.model import EditPlan
from aar.qa.planmap import carries_mixed_audio, cuts


def _plan(audio_tracks: list[dict], *, tracks: tuple[int, int] = (1, 1)) -> EditPlan:
    """Two segments from two different recordings -> one non-continuous join (a cut)."""
    return EditPlan.model_validate(
        {
            "version": 2,
            "project": "P",
            "timeline": "T",
            "template": "Tpl",
            "kind": "longform",
            "fps": {"num": 24, "den": 1},
            "resolution": {"width": 1920, "height": 1080},
            "profile": "ccn",
            "sources": [
                {"id": "a", "media": "a.mov", "recording_id": "rec1"},
                {"id": "b", "media": "b.mov", "recording_id": "rec2"},
            ],
            "segments": [
                {"track": tracks[0], "source": "a", "src_in": 0, "src_out": 48},
                {"track": tracks[1], "source": "b", "src_in": 0, "src_out": 48},
            ],
            "audio_tracks": audio_tracks,
        }
    )


def test_v1_plan_with_no_audio_tracks_is_still_checked() -> None:
    assert len(cuts(_plan([]), 1)) == 1


def test_muted_track_join_is_skipped() -> None:
    # The CCN topology: content on video track 1, A1 muted (feeds the bus but not the mix).
    plan = _plan([{"name": "A1", "mute": True, "bus": "A5"}])
    assert cuts(plan, 1) == []
    assert carries_mixed_audio(plan, 1) is False


def test_unrouted_track_join_is_skipped() -> None:
    # Not muted, but routed to no bus -> carries no mix (e.g. an A6 bounce track).
    plan = _plan([{"name": "A6", "mute": False, "bus": None}])
    assert cuts(plan, 1) == []


def test_bus_track_itself_is_checked() -> None:
    # A5 has no bus of its own but is the bus other tracks feed, so it carries the mix.
    plan = _plan([{"name": "A5", "mute": False, "bus": None}, {"name": "A1", "bus": "A5"}])
    assert len(cuts(plan, 1)) == 1
    assert carries_mixed_audio(plan, 1) is True


def test_routed_unmuted_track_is_checked() -> None:
    plan = _plan([{"name": "A1", "mute": False, "bus": "A5"}])
    assert len(cuts(plan, 1)) == 1


def test_short_audio_tracks_list_falls_back_to_checking() -> None:
    # Content on video track 2 but only one audio_tracks entry: no info -> checked.
    plan = _plan([{"name": "A1", "mute": True}], tracks=(2, 2))
    assert len(cuts(plan, 1)) == 1


def test_mixed_tracks_only_the_muted_one_is_skipped() -> None:
    # One join on a muted track 1, one join on a routed track 2.
    plan = EditPlan.model_validate(
        {
            "version": 2,
            "project": "P",
            "timeline": "T",
            "template": "Tpl",
            "kind": "longform",
            "fps": {"num": 24, "den": 1},
            "resolution": {"width": 1920, "height": 1080},
            "profile": "ccn",
            "sources": [
                {"id": "a", "media": "a.mov", "recording_id": "r1"},
                {"id": "b", "media": "b.mov", "recording_id": "r2"},
                {"id": "c", "media": "c.mov", "recording_id": "r3"},
                {"id": "d", "media": "d.mov", "recording_id": "r4"},
            ],
            "segments": [
                {"track": 1, "source": "a", "src_in": 0, "src_out": 48},
                {"track": 1, "source": "b", "src_in": 0, "src_out": 48},
                {"track": 2, "source": "c", "src_in": 0, "src_out": 48},
                {"track": 2, "source": "d", "src_in": 0, "src_out": 48},
            ],
            "audio_tracks": [
                {"name": "A1", "mute": True, "bus": "A5"},
                {"name": "A2", "mute": False, "bus": "A5"},
            ],
        }
    )
    kept = cuts(plan, 1)
    assert [c.incoming for c in kept] == [3]  # only the track-2 join survives
    assert [c.number for c in kept] == [1]  # numbering stays contiguous over kept cuts
