"""The pure parts of ``analyze.laughs``: the strict parser, the envelope cross-check, the
onset/end refinement and the merge. No ffmpeg, no network."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from aar.analyze.laughs import (
    BadResponse,
    Check,
    Found,
    LaughSettings,
    cross_check,
    merge,
    parse_items,
    parse_time,
    refine,
    response_text,
)

# ---------------------------------------------------------------- times


@pytest.mark.parametrize(
    ("value", "parsed"),
    [
        ("0:00", 0.0),
        ("1:05", 65.0),
        ("12:59", 779.0),
        ("1:02:03", 3723.0),
        ("0:07.5", 7.5),
        ("0:07.25", 7.25),
        ("0:07.125", 7.125),
        (75, "bare-number"),
        (75.5, "bare-number"),
        ("75", "bare-number"),
        ("75.5", "bare-number"),
        (True, "bad-time"),
        (None, "bad-time"),
        ("1:60", "bad-time"),  # seconds must be below 60
        ("1:75", "bad-time"),
        ("1:60:00", "bad-time"),  # minutes below 60 when hours are given
        ("1:5", "bad-time"),
        ("01:02:03:04", "bad-time"),
        ("0:07.1234", "bad-time"),
        ("\uff10:\uff10\uff17", "bad-time"),  # full-width digits
        ("0\uff1a07", "bad-time"),  # full-width colon
        ("\u0660:\u0660\u0667", "bad-time"),  # Arabic-Indic digits
        ("0:07\n", "bad-time"),
        (" 0:07", "bad-time"),
        ("-0:07", "bad-time"),
        ("", "bad-time"),
        ("about 0:07", "bad-time"),
    ],
)
def test_parse_time(value, parsed):
    assert parse_time(value) == parsed


# ---------------------------------------------------------------- items


def item(start="0:10", end="0:12", level=2, kind="laugh", **extra):
    return {"start": start, "end": end, "level": level, "kind": kind, **extra}


@pytest.mark.parametrize(
    ("raw", "reason"),
    [
        (item(start=10), "bare-number"),
        (item(end="12"), "bare-number"),
        (item(start="0:1O"), "bad-time"),
        (item(level=True), "bad-level"),
        (item(level="2"), "bad-level"),
        (item(level=2.0), "bad-level"),
        (item(level=0), "bad-level"),
        (item(level=4), "bad-level"),
        (item(kind="cheer"), "unknown-kind"),
        (item(kind="Laugh"), "unknown-kind"),
        (item(start="0:12", end="0:10"), "reversed"),
        (item(start="1:40", end="1:41"), "out-of-range"),  # starts at the duration (100 s)
        (item(start="1:39", end="1:41.5"), "out-of-range"),  # ends 1.5 s past it
        ({"start": "0:10", "end": "0:12", "level": 2}, "bad-item"),
        (["0:10", "0:12", 2, "laugh"], "bad-item"),
        ("laugh at 0:10", "bad-item"),
    ],
)
def test_every_drop_has_its_index_reason_and_excerpt(raw, reason):
    good = item()
    kept, drops = parse_items(json.dumps([good, raw]), 100.0)
    assert [k.id for k in kept] == [0]
    assert drops == [{"index": 1, "reason": reason, "raw": json.dumps(raw)[:200]}]


def test_kept_items_point_clamped_and_strength():
    raw = [
        item("0:10", "0:10", level=1),  # a point: end = start + 1 s
        item("1:39", "1:40.8", level=3),  # ends 0.8 s past the 100 s duration: clamped
        item("0:20", "0:25", level=2, kind="applause"),
    ]
    kept, drops = parse_items(json.dumps(raw), 100.0)
    assert drops == []
    point, clamped, applause = kept
    assert (point.start, point.end, point.flags, point.strength) == (10.0, 11.0, ["point"], 0.333)
    assert (clamped.end, clamped.flags, clamped.strength) == (100.0, ["clamped"], 1.0)
    assert (applause.kind, applause.strength) == ("applause", 0.667)


def test_a_long_item_excerpt_is_cut_at_200_characters():
    raw = item(kind="x" * 500)
    _, [drop] = parse_items(json.dumps([raw]), 100.0)
    assert len(drop["raw"]) == 200 and drop["reason"] == "unknown-kind"


@pytest.mark.parametrize(
    "text",
    ['```json\n[{"start": "0:10", "end": "0:12", "level": 2, "kind": "laugh"}]\n```', "[]"],
)
def test_one_json_fence_may_be_stripped(text):
    kept, drops = parse_items(text, 100.0)
    assert drops == [] and len(kept) == (0 if text == "[]" else 1)


@pytest.mark.parametrize(
    "text",
    [
        '{"start": "0:10"}',
        "Here are the laughs: []",
        "```\n[]\n```",  # only a json fence is stripped
        "```json\n```json\n[]\n```\n```",
        "",
        "null",
    ],
)
def test_anything_but_an_array_is_a_bad_response(text):
    with pytest.raises(BadResponse) as error:
        parse_items(text, 100.0)
    assert error.value.type == "bad-response" and error.value.exit_code == 1


def test_response_text_joins_the_first_candidate_parts():
    reply = {"candidates": [{"content": {"parts": [{"text": "[1"}, {"text": "]"}]}}]}
    assert response_text(reply) == "[1]"
    for bad in ({}, {"candidates": []}, {"candidates": [{"content": {"parts": []}}]}, []):
        with pytest.raises(BadResponse):
            response_text(bad)


# ---------------------------------------------------------------- cross-check

S = LaughSettings()


def env_with_window(values: list[float], baseline: float = -40.0) -> np.ndarray:
    """A 20 s, 10 Hz envelope at ``baseline`` whose frames 90.. hold ``values``.

    A laugh at [10.0, 10.1] has the window [9.0, 12.1): exactly 31 frames from frame 90, so
    its 90th percentile is the 28th smallest value."""
    env = np.full(200, baseline)
    env[90 : 90 + len(values)] = values
    return env


def window(top: list[float], fill: float = -40.0) -> list[float]:
    return sorted([fill] * (31 - len(top)) + top)


@pytest.mark.parametrize(("p90", "status"), [(-34.0, "confirmed"), (-34.01, "unconfirmed")])
def test_bump90_threshold_is_inclusive(p90, status):
    env = env_with_window(window([p90] * 4))  # sorted[27] is the first of the four
    check = cross_check(np, env, 10.0, 10.1, 20.0, LaughSettings(confirm_frac=0.0))
    assert check.baseline_db == -40.0
    assert check.bump90_db == pytest.approx(p90 + 40.0)
    assert check.status == status


@pytest.mark.parametrize(("above", "status"), [(3, "confirmed"), (2, "unconfirmed")])
def test_frac6_threshold_is_inclusive(above, status):
    env = env_with_window(window([-20.0] * above))
    settings = LaughSettings(confirm_bump_db=0.0, confirm_frac=3 / 31)
    check = cross_check(np, env, 10.0, 10.1, 20.0, settings)
    assert check.frac6 == above / 31
    assert check.status == status


def test_frac6_counts_frames_strictly_above_baseline_plus_6():
    env = env_with_window(window([-34.0] * 10))  # exactly +6: not above
    assert cross_check(np, env, 10.0, 10.1, 20.0, S).frac6 == 0.0


def test_a_clear_laugh_is_confirmed_and_a_flat_window_is_not():
    loud = cross_check(np, env_with_window(window([-25.0] * 12)), 10.0, 10.1, 20.0, S)
    assert loud.status == "confirmed" and loud.bump90_db == 15.0 and loud.frac6 == 12 / 31
    flat = cross_check(np, env_with_window(window([])), 10.0, 10.1, 20.0, S)
    assert flat.status == "unconfirmed" and flat.bump90_db == 0.0 and flat.frac6 == 0.0


def test_the_baseline_is_the_median_of_the_8_s_before_the_window():
    env = np.full(200, -40.0)
    env[:10] = 0.0  # more than 8 s before the window: ignored
    env[10:50] = -50.0  # 4 s of the 8 s baseline at -50, 4 s at -40: median -45
    env[90:121] = -30.0  # the window itself never counts
    check = cross_check(np, env, 10.0, 10.1, 20.0, S)
    assert check.baseline_db == -45.0 and check.bump90_db == 15.0


@pytest.mark.parametrize(("start", "status"), [(2.9, "unchecked"), (3.0, "confirmed")])
def test_less_than_2_s_of_baseline_is_unchecked(start, status):
    env = np.full(200, -40.0)
    env[20:60] = -20.0
    check = cross_check(np, env, start, start + 1.0, 20.0, S)
    assert check.status == status
    if status == "unchecked":
        assert check.reason == "no-baseline" and check.bump90_db is None


def test_silence_is_unchecked():
    env = np.full(200, -120.0)
    check = cross_check(np, env, 10.0, 11.0, 20.0, S)
    assert (check.status, check.reason) == ("unchecked", "silent")
    quiet_room = np.full(200, -85.0)
    quiet_room[90:121] = -60.0  # a laugh in a very quiet room still counts
    assert cross_check(np, quiet_room, 10.0, 10.1, 20.0, S).status == "confirmed"


# ---------------------------------------------------------------- refinement


def env50(n: int = 1000, base: float = -40.0) -> np.ndarray:
    return np.full(n, base)


def test_the_onset_needs_200_ms_sustained_above_plus_3():
    env = env50()
    env[500:505] = -30.0  # a 100 ms burst at 10.0 s, then back to the baseline
    env[520:545] = -30.0  # the laugh: 500 ms from 10.4 s
    start, end, flags = refine(np, env, 10.5, 11.0, -40.0, 20.0)
    assert start == pytest.approx(10.4) and "onset-unrefined" not in flags
    assert end == pytest.approx(10.9)  # the last frame above +3 ends at 545 * 0.02


@pytest.mark.parametrize(("frames", "found"), [(10, True), (9, False)])
def test_exactly_200_ms_is_enough(frames, found):
    env = env50()
    env[520 : 520 + frames] = -30.0
    start, _, flags = refine(np, env, 10.5, 11.0, -40.0, 20.0)
    if found:
        assert start == pytest.approx(10.4)
    else:
        assert start == 10.5 and "onset-unrefined" in flags


def test_the_onset_frame_must_be_above_plus_6_and_the_rest_above_plus_3():
    env = env50()
    env[520:540] = -35.0  # +5: sustained, but never +6
    env[530] = -33.0  # +7 at 10.6 s, and the next 200 ms stay above +3
    start, _, _ = refine(np, env, 10.5, 11.0, -40.0, 20.0)
    assert start == pytest.approx(10.6)


def test_the_onset_search_window_is_a_minus_1_to_a_plus_1_5():
    env = env50()
    env[425:445] = -30.0  # 8.5-8.9 s: wholly more than 1 s before a = 10
    start, _, flags = refine(np, env, 10.0, 12.0, -40.0, 20.0)
    assert start == 10.0 and "onset-unrefined" in flags
    env = env50()
    env[450:480] = -30.0  # 9.0 s: exactly 1 s before
    assert refine(np, env, 10.0, 12.0, -40.0, 20.0)[0] == pytest.approx(9.0)
    env = env50()
    env[576:600] = -30.0  # 11.52 s: past a + 1.5
    assert refine(np, env, 10.0, 12.0, -40.0, 20.0)[0] == 10.0


def test_no_baseline_means_no_refinement():
    start, end, flags = refine(np, env50(), 10.0, 11.0, None, 20.0)
    assert (start, end) == (10.0, 11.0)
    assert flags == ["onset-unrefined", "end-unrefined"]


def test_start_stays_below_end():
    # Only +4 dB from 9.6 to 9.8 s: no onset (never +6), so the start stays at a = 10.0, but
    # the end search finds 9.8 s. The end then falls back to b, keeping start < end.
    env = env50()
    env[480:490] = -36.0
    start, end, flags = refine(np, env, 10.0, 10.5, -40.0, 20.0)
    assert (start, end) == (10.0, 10.5) and "onset-unrefined" in flags


# ---------------------------------------------------------------- merge


def found(start, end, status="confirmed", level=2, ident=0, bump=10.0):
    return Found([ident], start, end, (start, end), level, Check(status, bump90_db=bump), [])


@pytest.mark.parametrize(("gap", "merged"), [(0.25, True), (0.26, False), (-0.5, True)])
def test_laughs_within_0_25_s_merge(gap, merged):
    out, merges = merge([found(10.0, 11.0, ident=0), found(11.0 + gap, 12.5, ident=1)])
    assert (len(out) == 1) is merged
    if merged:
        assert out[0].ids == [0, 1] and out[0].start == 10.0 and out[0].end == 12.5
        assert merges == [{"into": 0, "ids": [0, 1], "file_start_s": 10.0, "file_end_s": 12.5}]
    else:
        assert merges == []


def test_a_merge_keeps_the_larger_strength_and_the_best_status():
    out, _ = merge(
        [
            found(10.0, 11.0, "unchecked", level=3, ident=0),
            found(11.1, 12.0, "confirmed", level=1, ident=1),
            found(12.1, 13.0, "unconfirmed", level=2, ident=2),
        ]
    )
    [laugh] = out
    assert laugh.level == 3 and laugh.strength == 1.0
    assert laugh.check.status == "confirmed"
    assert laugh.ids == [0, 1, 2]
    assert math.isclose(laugh.end, 13.0)
