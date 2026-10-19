"""timeline.append_ranges against the fake's append model, one guard per known trap."""

from __future__ import annotations

from typing import Any

from aar.node.ops import make_hot_op
from aar.node.ops.timebase import converted_length

from .fake_resolve import FakeMediaPoolItem
from .ops_support import AGENT, Show, media

START = 108000  # the fake timelines start at 01:00:00:00 of a 30-nominal clock


def rng(clip: FakeMediaPoolItem, src_in: int, src_out: int, **kw: Any) -> dict[str, Any]:
    return {"media": media(clip), "src_in": src_in, "src_out": src_out, **kw}


def append(show: Show, *ranges: dict[str, Any], **params: Any) -> Any:
    return show.ok(
        "timeline.append_ranges",
        {"timeline": show.ref("Main [agent]"), "ranges": list(ranges), **params},
    )


def spans(show: Show, kind: str = "video", track: int = 1) -> list[tuple[int, int]]:
    return [(i.start - START, i.end - START) for i in show.branch.items(kind, track)]


def appends(show: Show) -> int:
    return sum(1 for call in show.fake.calls if call[0] == "AppendToTimeline")


def test_same_rate_ranges_land_exactly_and_end_frame_goes_through_exclusive() -> None:
    show = Show()
    result = append(show, rng(show.cam_a, 100, 130), rng(show.cam_a, 400, 460))

    # trap append-endframe-exclusive: [100, 130) is 30 frames with endFrame=130. Passing 129
    # would land 29 frames and need a repair.
    first, second = result.ranges
    assert (first.status, first.end_frame_used, first.planned_frames) == ("appended", 130, 30)
    assert first.video is not None
    assert (first.video.record_in, first.video.record_out) == (0, 30)
    assert (first.video.source_start, first.video.source_end) == (100, 130)
    assert second.video is not None and (second.video.record_in, second.video.duration) == (30, 60)
    assert spans(show) == [(0, 30), (30, 90)]
    assert result.gaps == [] and result.done and result.next_cursor is None
    assert show.main.items("video") == []  # the agent branch was made current, not Main
    assert show.gw.get_context().timeline == "Main"  # and the human's timeline came back


def test_rate_converted_range_one_frame_short_is_repaired_with_end_plus_one() -> None:
    show = Show()
    assert converted_length(7, 24.0, 29.97) == 9  # 8.74 timeline frames, rounded
    result = append(
        show, rng(show.cam_24, 0, 7), rng(show.cam_24, 100, 110), rng(show.cam_a, 0, 30)
    )

    repaired = result.ranges[0]
    assert repaired.status == "repaired" and repaired.end_frame_used == 8
    assert [n.trap_id for n in repaired.notes] == ["append-fps-short"]
    assert repaired.video is not None and repaired.video.duration == 9
    assert [r.status for r in result.ranges[1:]] == ["appended", "appended"]
    assert spans(show) == [(0, 9), (9, 21), (21, 51)]
    assert result.gaps == []


def test_without_the_repair_a_short_range_is_an_error_and_leaves_a_gap() -> None:
    show = Show()
    error = show.fails(
        "timeline.append_ranges",
        {
            "timeline": show.ref("Main [agent]"),
            "ranges": [rng(show.cam_24, 0, 7), rng(show.cam_a, 0, 30)],
            "repair_short": False,
        },
    )
    assert error.code == "length_mismatch"
    assert error.detail["trap_id"] == "append-fps-short"
    assert error.detail["resume_cursor"] == 0
    assert spans(show) == [(0, 8)]  # one frame short of the next range's planned record_in 9


def test_a_repair_that_overshoots_stops_with_the_item_left_for_inspection() -> None:
    show = Show()
    # 12 frames of 24p fill 14.99 timeline frames: planned 15, lands 14, and 13 source
    # frames land 16.
    error = show.fails(
        "timeline.append_ranges",
        {
            "timeline": show.ref("Main [agent]"),
            "ranges": [rng(show.cam_a, 0, 30), rng(show.cam_24, 0, 12)],
        },
    )
    assert error.code == "repair_overshoot"
    assert error.detail["trap_id"] == "append-fps-short"
    assert error.detail["resume_cursor"] == 1
    assert [r["status"] for r in error.detail["completed"]] == ["appended"]
    assert spans(show) == [(0, 30), (30, 46)]


def test_braw_av_append_finds_the_audio_the_api_did_not_return() -> None:
    show = Show()
    result = append(show, rng(show.cam_b, 1000, 1100))
    placed = result.ranges[0]
    assert placed.video is not None and len(placed.audio) == 1
    audio = placed.audio[0]
    assert (audio.track_type, audio.record_in, audio.duration) == ("audio", 0, 100)
    assert "av-append-video-only" in {n.trap_id for n in placed.notes}
    # non-BRAW: the API returned both items. A new call plans from frame 0 unless told.
    plain = append(show, rng(show.cam_a, 0, 50, record_in=200))
    assert len(plain.ranges[0].audio) == 1
    assert "av-append-video-only" not in {n.trap_id for n in plain.ranges[0].notes}


def test_video_only_and_audio_only_ranges() -> None:
    show = Show()
    result = append(
        show,
        rng(show.cam_a, 0, 40, media_type="video"),
        rng(show.music, 0, 90, media_type="audio"),
    )
    video, music = result.ranges
    assert video.video is not None and video.audio == []
    assert music.video is None and [a.duration for a in music.audio] == [90]
    assert spans(show, "audio") == [(0, 90)]


def test_stills_are_refused_before_anything_is_appended() -> None:
    show = Show()
    error = show.fails(
        "timeline.append_ranges",
        {
            "timeline": show.ref("Main [agent]"),
            "ranges": [rng(show.cam_a, 0, 30), rng(show.still, 0, 48)],
        },
    )
    assert error.code == "still_refused"
    assert error.detail["trap_id"] == "append-still-ignores-range"
    assert appends(show) == 0 and spans(show) == []


def test_multicam_clips_are_refused() -> None:
    show = Show()
    error = show.fails(
        "timeline.append_ranges",
        {"timeline": show.ref("Main [agent]"), "ranges": [rng(show.multicam, 0, 30)]},
    )
    assert (error.code, error.detail["trap_id"]) == ("multicam_refused", "multicam-angles")
    assert appends(show) == 0


def test_resume_cursor_batches_and_reruns_are_idempotent() -> None:
    show = Show()
    ranges = [rng(show.cam_a, i * 100, i * 100 + 25) for i in range(4)]
    first = append(show, *ranges, max_ranges=2)
    assert ([r.index for r in first.ranges], first.next_cursor, first.done) == ([0, 1], 2, False)
    second = append(show, *ranges, cursor=2, max_ranges=2)
    assert ([r.index for r in second.ranges], second.done) == ([2, 3], True)
    before = appends(show)
    again = append(show, *ranges, max_ranges=10)
    assert [r.status for r in again.ranges] == ["present"] * 4
    assert appends(show) == before
    assert spans(show) == [(0, 25), (25, 50), (50, 75), (75, 100)]


def test_a_short_item_left_by_an_earlier_attempt_is_repaired_on_rerun() -> None:
    show = Show()
    show.branch.place("video", 1, show.cam_24, START, 3, src_start=0)  # 1 short of 4
    result = append(show, rng(show.cam_24, 0, 3))
    assert result.ranges[0].status == "repaired"
    assert spans(show) == [(0, 4)]


def test_an_occupied_slot_is_never_overwritten() -> None:
    show = Show()
    show.branch.place("video", 1, show.cam_b, START + 10, 50)
    error = show.fails(
        "timeline.append_ranges",
        {"timeline": show.ref("Main [agent]"), "ranges": [rng(show.cam_a, 0, 30)]},
    )
    assert error.code == "slot_occupied"
    assert appends(show) == 0


def test_record_in_places_ranges_and_gaps_are_measured() -> None:
    show = Show()
    result = append(show, rng(show.cam_a, 0, 30), rng(show.cam_a, 100, 130, record_in=50))
    assert spans(show) == [(0, 30), (50, 80)]
    gaps = [(g.track_type, g.track, g.record_in, g.record_out, g.frames) for g in result.gaps]
    assert gaps == [("audio", 1, 30, 50, 20), ("video", 1, 30, 50, 20)]


def test_bad_ranges_are_refused_up_front() -> None:
    show = Show()
    beyond = show.fails(
        "timeline.append_ranges",
        {"timeline": show.ref("Main [agent]"), "ranges": [rng(show.cam_a, 19990, 20010)]},
    )
    assert beyond.code == "range_beyond_media"
    no_track = show.fails(
        "timeline.append_ranges",
        {"timeline": show.ref("Main [agent]"), "ranges": [rng(show.cam_a, 0, 10, track=3)]},
    )
    assert no_track.code == "track_missing"
    assert appends(show) == 0


def test_source_readback_one_frame_off_is_noted_two_frames_is_an_error() -> None:
    show = Show()
    show.fake.source_readback_skew = -1
    ok = append(show, rng(show.cam_a, 100, 130))
    assert "source-frame-readback-rounding" in {n.trap_id for n in ok.ranges[0].notes}
    assert ok.ranges[0].video is not None and ok.ranges[0].video.source_start == 99
    show.fake.source_readback_skew = 2
    error = show.fails(
        "timeline.append_ranges",
        {
            "timeline": show.ref("Main [agent]"),
            "ranges": [rng(show.cam_a, 100, 130), rng(show.cam_a, 500, 530)],
        },
    )
    assert error.code == "source_mismatch" and error.detail["resume_cursor"] == 1


def test_appended_items_are_flagged_ungraded() -> None:
    show = Show()
    result = append(show, rng(show.cam_a, 0, 30))
    assert [(n.code, n.trap_id) for n in result.notes] == [("items_ungraded", "append-ungraded")]


def test_append_leases_and_selects_the_named_timeline() -> None:
    show = Show()
    hot = make_hot_op(
        "timeline.append_ranges",
        {"timeline": show.ref("Main [agent]"), "ranges": [rng(show.cam_a, 0, 30)]},
        target=show.locator(),
        holder=AGENT,
        require_branch=False,
    )
    assert hot.timeline == "Main [agent]" and hot.label == "timeline.append_ranges"


def test_ntsc_rates_are_converted_exactly() -> None:
    # 1501 frames at 24 fps fill 1500.4995... frames at 23.976 (24000/1001) but only 1499.4995
    # at a rounded 23.976: the exact-rate table decides between 1500 and 1499.
    assert converted_length(1501, 24.0, 23.976) == 1500
    show = Show()
    show.project.settings["timelineFrameRate"] = 23.976
    result = append(show, rng(show.cam_24, 0, 1501))
    placed = result.ranges[0]
    assert placed.planned_frames == 1500
    assert placed.video is not None and placed.video.duration == 1500
    assert spans(show) == [(0, 1500)]


def test_a_short_range_at_the_end_of_its_media_is_not_deleted_to_be_repaired() -> None:
    show = Show()
    tail = show.project.media_pool.add_clip(("Footage",), "short 24p", fps=24.0, frames=7)
    error = show.fails(
        "timeline.append_ranges",
        {"timeline": show.ref("Main [agent]"), "ranges": [rng(tail, 0, 7)]},
    )
    assert (error.code, error.detail["trap_id"]) == ("repair_beyond_media", "append-fps-short")
    assert spans(show) == [(0, 8)]  # the short item stays for the agent to decide
    assert not [c for c in show.fake.calls if c[0] == "DeleteClips"]


def test_a_short_item_that_will_not_delete_stops_the_repair() -> None:
    show = Show()
    show.fake.faults.add("DeleteClips:no-op")
    error = show.fails(
        "timeline.append_ranges",
        {"timeline": show.ref("Main [agent]"), "ranges": [rng(show.cam_24, 0, 7)]},
    )
    assert (error.code, error.detail["trap_id"]) == ("repair_delete_failed", "append-fps-short")
    assert spans(show) == [(0, 8)]  # never appended a second copy on top
    assert appends(show) == 1
