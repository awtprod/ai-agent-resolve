"""S13b F3 pin: a QA join sits on a track that carries mixed audio when that track is unmuted,
routed to a bus, or is itself a bus. The CCN A1-A4 (muted but bussed to A5) therefore carry the
dialogue to the mix and are checked; only a muted, unrouted, non-bus track is dropped. A plan
with no ``audio_tracks`` (every v1 plan) is checked, as before.
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


def test_v1_plan_with_no_audio_tracks_is_checked() -> None:
    assert len(cuts(_plan([]), 1)) == 1


def test_ccn_muted_but_bussed_track_is_checked() -> None:
    # The CCN topology: A1 is muted but routed to the A5 bus, so it carries the dialogue.
    plan = _plan([{"name": "A1", "mute": True, "bus": "A5"}])
    assert carries_mixed_audio(plan, 1) is True
    assert len(cuts(plan, 1)) == 1


def test_unmuted_track_is_checked() -> None:
    plan = _plan([{"name": "A1", "mute": False, "bus": None}])
    assert carries_mixed_audio(plan, 1) is True
    assert len(cuts(plan, 1)) == 1


def test_bus_track_itself_is_checked() -> None:
    # A5 has no bus of its own but is the bus other tracks feed, so it carries the mix.
    plan = _plan([{"name": "A5", "mute": False, "bus": None}, {"name": "A1", "bus": "A5"}])
    assert carries_mixed_audio(plan, 1) is True
    assert len(cuts(plan, 1)) == 1


def test_muted_bus_track_still_carries_the_mix() -> None:
    # Even a muted, unrouted track carries the mix if it IS the bus another track feeds.
    plan = _plan([{"name": "A5", "mute": True, "bus": None}, {"name": "A1", "bus": "A5"}])
    assert carries_mixed_audio(plan, 1) is True
    assert len(cuts(plan, 1)) == 1


def test_muted_unrouted_non_bus_track_carries_no_mix_and_is_filtered() -> None:
    plan = _plan([{"name": "A1", "mute": True, "bus": None}])
    assert carries_mixed_audio(plan, 1) is False
    assert cuts(plan, 1) == []
    # But only_mixed=False still sees the join, so the check reports skipped, not "no cuts".
    assert len(cuts(plan, 1, only_mixed=False)) == 1


def test_short_audio_tracks_list_falls_back_to_checking() -> None:
    plan = _plan([{"name": "A1", "mute": True}], tracks=(2, 2))
    assert len(cuts(plan, 1)) == 1


def test_mixed_tracks_only_the_no_mix_one_is_filtered() -> None:
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
            # track 1 muted+unrouted (no mix), track 2 muted+bussed (carries mix).
            "audio_tracks": [
                {"name": "A1", "mute": True, "bus": None},
                {"name": "A2", "mute": True, "bus": "A5"},
            ],
        }
    )
    kept = cuts(plan, 1)
    assert [c.incoming for c in kept] == [3]  # only the track-2 join survives
    assert [c.number for c in kept] == [1]  # numbering stays contiguous over kept cuts
