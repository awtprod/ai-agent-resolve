"""Timeline import (S20) fix round: the validator's findings H1-H4, M1-M3 and the smaller ones, each
as a test that fails on the code before the fix (aar-evidence/s20-val/VERDICT.md)."""

from __future__ import annotations

import dataclasses
from dataclasses import replace
from typing import Any

import pytest
from export_support import MEDIA_MAP, load_plan, permissive
from test_timeline_import import (
    RATE_2997,
    TC_2997,
    codes,
    layout_rows,
    ranges,
    run,
    simple,
    simple_data,
    statuses,
)
from timeline_import_support import Row, build_timeline, content, ripple

from aar.drp.model import Issue
from aar.plan.cli import _import_text
from aar.plan.export import media_map_from_mapping
from aar.plan.model import EditPlan
from aar.plan.timeline_import import (
    READER_WARNING_CODES,
    ImportResult,
    TimelineImportError,
    assign_items,
    import_timeline,
)

# -- the validator's twelve misattribution seeds (p_conserve.py), as data ------------------------
# Sources as in the validator's generator; rows are (track, record_in, duration, src_in, file,
# speed, tag), the tag naming the original segment a row came from.

SEED_FILES = {
    "CamA.mov": "E:\\Show\\CamA.mov",
    "CamB.mov": "E:\\Show\\CamB.mov",
    "CamC.MOV": "F:\\Other\\CamC.MOV",
    "Outro.mov": "E:\\Assets\\Outro.mov",
    "Fire.mov": "G:\\Stock\\Fire.mov",
}

SEED_CASES: dict[int, dict[str, Any]] = {
    345: {
        "plan": {
            "fps": 25,
            "segments": [
                {"source": "B", "src_in": 23, "src_out": 67, "gain_db": -3.0, "label": "L"},
                {"source": "C", "src_in": 612, "src_out": 686},
                {"track": 2, "source": "C", "src_in": 5, "src_out": 40, "record_in": 52},
                {"source": "C", "src_in": 612, "src_out": 686},
                {"source": "A", "src_in": 3217, "src_out": 3376},
                {"source": "B", "src_in": 3369, "src_out": 3591, "gain_db": -3.0, "label": "L"},
                {
                    "source": "B",
                    "src_in": 2584,
                    "src_out": 2655,
                    "record_in": 590,
                    "gain_db": -3.0,
                    "label": "L",
                },
                {"source": "A", "src_in": 3291, "src_out": 3476, "gain_db": -3.0, "label": "L"},
            ],
            "head": {"video": "dissolve", "video_frames": 6},
            "outro": {"source": "O", "audio_track": "A1"},
        },
        "rows": [
            ("V1", 4, 44, 23, "CamB.mov", 1.0, "segments[0]"),
            ("V1", 70, 74, 612, "CamC.MOV", 1.0, "segments[1]"),
            ("V1", 144, 74, 612, "CamC.MOV", 1.0, "segments[3]"),
            ("V1", 227, 190, 3186, "CamA.mov", 1.0, "segments[4]"),
            ("V1", 417, 222, 3369, "CamB.mov", 1.0, "segments[5]"),
            ("V1", 639, 71, 2584, "CamB.mov", 1.0, "segments[6]"),
            ("V1", 720, 70, 3291, "CamA.mov", 1.0, "segments[7]"),
            ("V1", 790, 60, 3361, "CamA.mov", 1.0, "segments[7]"),
            ("V1", 850, 39, 10, "Fire.mov", 1.0, "foreign"),
            ("V1", 889, 41, 3421, "CamA.mov", 1.0, "segments[7]"),
            ("V1", 940, 14, 3462, "CamA.mov", 1.0, "segments[7]"),
            ("V1", 954, 90, 0, "Outro.mov", 1.0, "outro"),
            ("V2", 52, 35, 5, "CamC.MOV", 1.0, "segments[2]"),
        ],
    },
    459: {
        "plan": {
            "fps": "30000/1001",
            "segments": [
                {"source": "C", "src_in": 613, "src_out": 717, "gain_db": -3.0, "label": "L"},
                {"source": "B", "src_in": 3577, "src_out": 3664},
                {"track": 2, "source": "C", "src_in": 5, "src_out": 40, "record_in": 37},
                {"source": "C", "src_in": 2532, "src_out": 2729},
                {"source": "A", "src_in": 2250, "src_out": 2310, "gain_db": -3.0, "label": "L"},
                {"source": "C", "src_in": 632, "src_out": 730},
                {"source": "A", "src_in": 2859, "src_out": 2992},
            ],
            "outro": {"source": "O", "audio_track": "A1"},
        },
        "rows": [
            ("V1", 0, 28, 689, "CamC.MOV", 1.0, "segments[0]"),
            ("V1", 28, 6, 3577, "CamB.mov", 1.0, "segments[1]"),
            ("V1", 34, 34, 3583, "CamB.mov", 1.0, "segments[1]"),
            ("V1", 70, 47, 3617, "CamB.mov", 1.0, "segments[1]"),
            ("V1", 117, 197, 2532, "CamC.MOV", 0.5, "segments[3]@retimed"),
            ("V1", 314, 60, 2250, "CamA.mov", 1.0, "segments[4]"),
            ("V1", 374, 21, 632, "CamC.MOV", 1.0, "segments[5]"),
            ("V1", 395, 60, 670, "CamC.MOV", 1.0, "segments[5]"),
            ("V1", 455, 133, 2859, "CamA.mov", 1.0, "segments[6]"),
            ("V1", 588, 90, 0, "Outro.mov", 1.0, "outro"),
            ("V2", 37, 35, 5, "CamC.MOV", 1.0, "segments[2]"),
        ],
    },
    638: {
        "plan": {
            "fps": "30000/1001",
            "segments": [
                {"source": "A", "src_in": 1937, "src_out": 2124},
                {"source": "C", "src_in": 521, "src_out": 603},
                {"source": "B", "src_in": 3114, "src_out": 3260},
                {
                    "source": "B",
                    "src_in": 1027,
                    "src_out": 1101,
                    "reframe": [{"time": 0, "x": 0.3}, {"time": 58, "x": 0.7, "zoom": 1.5}],
                },
                {"track": 2, "source": "C", "src_in": 5, "src_out": 40, "record_in": 15},
                {
                    "source": "A",
                    "src_in": 1908,
                    "src_out": 2057,
                    "reframe": [{"time": 0, "x": 0.3}, {"time": 48, "x": 0.7, "zoom": 1.5}],
                    "gain_db": -3.0,
                    "label": "L",
                },
                {
                    "source": "A",
                    "src_in": 1908,
                    "src_out": 2057,
                    "reframe": [{"time": 0, "x": 0.3}, {"time": 48, "x": 0.7, "zoom": 1.5}],
                    "gain_db": -3.0,
                    "label": "L",
                },
            ],
            "head": {"video": "dissolve", "video_frames": 6},
        },
        "rows": [
            ("V1", 0, 216, 1908, "CamA.mov", 1.0, "segments[0]"),
            ("V1", 216, 149, 1908, "CamA.mov", 1.0, "segments[5]"),
            ("V1", 365, 171, 3114, "CamB.mov", 1.0, "segments[2]"),
            ("V1", 555, 74, 1027, "CamB.mov", 1.0, "segments[3]"),
            ("V1", 629, 82, 521, "CamC.MOV", 1.0, "segments[1]"),
            ("V1", 711, 95, 1908, "CamA.mov", 1.0, "segments[6]"),
            ("V1", 806, 54, 2003, "CamA.mov", 1.0, "segments[6]"),
            ("V2", 15, 35, 5, "CamC.MOV", 1.0, "segments[4]"),
        ],
    },
    785: {
        "plan": {
            "fps": 24,
            "segments": [
                {"source": "A", "src_in": 3696, "src_out": 3745},
                {"source": "A", "src_in": 577, "src_out": 792},
                {"source": "B", "src_in": 1641, "src_out": 1707},
                {
                    "source": "A",
                    "src_in": 680,
                    "src_out": 696,
                    "reframe": [{"time": 0, "x": 0.3}, {"time": 84, "x": 0.7, "zoom": 1.5}],
                },
                {"source": "B", "src_in": 2057, "src_out": 2202},
            ],
            "outro": {"source": "O", "audio_track": "A1"},
        },
        "rows": [
            ("V1", 18, 49, 3696, "CamA.mov", 1.0, "segments[0]"),
            ("V1", 67, 137, 577, "CamA.mov", 1.0, "segments[1]"),
            ("V1", 204, 20, 772, "CamA.mov", 1.0, "segments[1]"),
            ("V1", 224, 66, 1641, "CamB.mov", 1.0, "segments[2]"),
            ("V1", 290, 38, 10, "CamA.mov", 1.0, "samename"),
            ("V1", 336, 10, 686, "CamA.mov", 1.0, "segments[3]"),
            ("V1", 346, 145, 2057, "CamB.mov", 1.0, "segments[4]"),
            ("V1", 491, 90, 0, "Outro.mov", 1.0, "outro"),
        ],
    },
    938: {
        "plan": {
            "fps": 25,
            "segments": [
                {"source": "B", "src_in": 1519, "src_out": 1750, "record_in": 30},
                {"source": "B", "src_in": 1344, "src_out": 1581},
                {"source": "B", "src_in": 1344, "src_out": 1581},
                {"source": "A", "src_in": 1413, "src_out": 1583},
            ],
            "head": {"video": "dissolve", "video_frames": 6},
        },
        "rows": [
            ("V1", 0, 245, 1519, "CamB.mov", 1.0, "segments[0]"),
            ("V1", 245, 237, 1344, "CamB.mov", 1.0, "segments[1]"),
            ("V1", 482, 71, 10, "Fire.mov", 1.0, "foreign"),
            ("V1", 553, 120, 1344, "CamB.mov", 1.0, "segments[2]"),
            ("V1", 673, 13, 1568, "CamB.mov", 1.0, "segments[2]"),
            ("V1", 686, 170, 1413, "CamA.mov", 1.0, "segments[3]"),
        ],
    },
    968: {
        "plan": {
            "fps": 24,
            "segments": [
                {"source": "A", "src_in": 1178, "src_out": 1378},
                {"source": "A", "src_in": 1178, "src_out": 1378},
                {"source": "B", "src_in": 1183, "src_out": 1259},
                {"source": "B", "src_in": 1183, "src_out": 1259},
                {"source": "A", "src_in": 1101, "src_out": 1334, "gain_db": -3.0, "label": "L"},
                {
                    "source": "A",
                    "src_in": 2948,
                    "src_out": 3003,
                    "reframe": [{"time": 0, "x": 0.3}, {"time": 85, "x": 0.7, "zoom": 1.5}],
                },
                {"source": "B", "src_in": 3630, "src_out": 3772},
            ],
            "head": {"video": "dissolve", "video_frames": 6},
            "outro": {"source": "O", "audio_track": "A1"},
        },
        "rows": [
            ("V1", 18, 55, 1178, "CamA.mov", 1.0, "segments[0]"),
            ("V1", 73, 200, 1178, "CamA.mov", 0.5, "segments[1]@retimed"),
            ("V1", 273, 76, 1183, "CamB.mov", 1.0, "segments[2]"),
            ("V1", 349, 76, 1183, "CamB.mov", 1.0, "segments[3]"),
            ("V1", 425, 26, 1101, "CamA.mov", 1.0, "segments[4]"),
            ("V1", 451, 1, 1333, "CamA.mov", 1.0, "segments[4]"),
            ("V1", 452, 63, 10, "Fire.mov", 1.0, "foreign"),
            ("V1", 515, 52, 2951, "CamA.mov", 1.0, "segments[5]"),
            ("V1", 567, 142, 3630, "CamB.mov", 1.0, "segments[6]"),
            ("V1", 709, 90, 0, "Outro.mov", 1.0, "outro"),
        ],
    },
    974: {
        "plan": {
            "fps": "30000/1001",
            "segments": [
                {"source": "C", "src_in": 2465, "src_out": 2666},
                {"source": "B", "src_in": 1958, "src_out": 2060},
                {"source": "C", "src_in": 2435, "src_out": 2603},
            ],
            "outro": {"source": "O", "audio_track": "A1"},
        },
        "rows": [
            ("V1", 0, 54, 2006, "CamB.mov", 1.0, "segments[1]"),
            ("V1", 54, 80, 2465, "CamC.MOV", 1.0, "segments[0]"),
            ("V1", 134, 100, 2503, "CamC.MOV", 1.0, "segments[2]"),
            ("V1", 234, 14, 10, "CamA.mov", 1.0, "samename"),
            ("V1", 248, 90, 0, "Outro.mov", 1.0, "outro"),
        ],
    },
    2053: {
        "plan": {
            "fps": 25,
            "segments": [
                {"source": "C", "src_in": 3330, "src_out": 3529, "gain_db": -3.0, "label": "L"},
                {"source": "C", "src_in": 3354, "src_out": 3480},
                {"source": "C", "src_in": 3354, "src_out": 3480},
            ],
            "outro": {"source": "O", "audio_track": "A1"},
        },
        "rows": [
            ("V1", 0, 70, 3330, "CamC.MOV", 1.0, "segments[0]"),
            ("V1", 70, 14, 10, "Fire.mov", 1.0, "foreign"),
            ("V1", 84, 4, 3476, "CamC.MOV", 1.0, "segments[1]"),
            ("V1", 88, 126, 3354, "CamC.MOV", 1.0, "segments[2]"),
            ("V1", 241, 64, 10, "Fire.mov", 1.0, "foreign"),
            ("V1", 305, 90, 0, "Outro.mov", 1.0, "outro"),
        ],
    },
    2062: {
        "plan": {
            "fps": 24,
            "segments": [
                {"source": "C", "src_in": 1819, "src_out": 1955},
                {"source": "C", "src_in": 1836, "src_out": 1842},
                {"source": "A", "src_in": 1908, "src_out": 2121, "gain_db": -3.0, "label": "L"},
            ],
        },
        "rows": [
            ("V1", 0, 21, 549, "CamB.mov", 1.0, "new"),
            ("V1", 23, 36, 1836, "CamC.MOV", 1.0, "segments[1]"),
            ("V1", 69, 220, 1901, "CamA.mov", 1.0, "segments[2]"),
        ],
    },
    2699: {
        "plan": {
            "fps": 25,
            "segments": [
                {"source": "B", "src_in": 995, "src_out": 1237},
                {"source": "B", "src_in": 1687, "src_out": 1692, "gain_db": -3.0, "label": "L"},
                {"source": "C", "src_in": 2347, "src_out": 2484},
                {
                    "source": "A",
                    "src_in": 2365,
                    "src_out": 2443,
                    "reframe": [{"time": 0, "x": 0.3}, {"time": 81, "x": 0.7, "zoom": 1.5}],
                },
                {"source": "B", "src_in": 1659, "src_out": 1754},
            ],
        },
        "rows": [
            ("V1", 0, 242, 995, "CamB.mov", 1.0, "segments[0]"),
            ("V1", 242, 20, 10, "CamA.mov", 1.0, "samename"),
            ("V1", 262, 59, 10, "CamA.mov", 1.0, "samename"),
            ("V1", 326, 78, 2365, "CamA.mov", 1.0, "segments[3]"),
            ("V1", 404, 137, 2347, "CamC.MOV", 1.0, "segments[2]"),
            ("V1", 541, 25, 1667, "CamB.mov", 1.0, "segments[1]"),
            ("V1", 566, 109, 1645, "CamB.mov", 1.0, "segments[4]"),
        ],
    },
    2817: {
        "plan": {
            "fps": 25,
            "segments": [
                {"source": "C", "src_in": 1520, "src_out": 1732},
                {"source": "A", "src_in": 471, "src_out": 667},
                {"source": "B", "src_in": 2840, "src_out": 2888},
                {"source": "B", "src_in": 2048, "src_out": 2284},
                {"source": "A", "src_in": 524, "src_out": 529, "gain_db": -3.0, "label": "L"},
                {"source": "C", "src_in": 919, "src_out": 1163},
                {
                    "source": "A",
                    "src_in": 699,
                    "src_out": 860,
                    "reframe": [{"time": 0, "x": 0.3}, {"time": 83, "x": 0.7, "zoom": 1.5}],
                    "gain_db": -3.0,
                    "label": "L",
                },
            ],
            "head": {"video": "dissolve", "video_frames": 6},
            "outro": {"source": "O", "audio_track": "A1"},
        },
        "rows": [
            ("V1", 13, 212, 1520, "CamC.MOV", 1.0, "segments[0]"),
            ("V1", 225, 69, 471, "CamA.mov", 1.0, "segments[1]"),
            ("V1", 294, 54, 540, "CamA.mov", 1.0, "segments[1]"),
            ("V1", 348, 73, 594, "CamA.mov", 1.0, "segments[1]"),
            ("V1", 421, 48, 2840, "CamB.mov", 1.0, "segments[2]"),
            ("V1", 469, 236, 2048, "CamB.mov", 1.0, "segments[3]"),
            ("V1", 705, 10, 524, "CamA.mov", 1.0, "segments[4]"),
            ("V1", 715, 244, 919, "CamC.MOV", 1.0, "segments[5]"),
            ("V1", 959, 161, 699, "CamA.mov", 1.0, "segments[6]"),
            ("V1", 1120, 90, 0, "Outro.mov", 1.0, "outro"),
        ],
    },
    3268: {
        "plan": {
            "fps": "30000/1001",
            "segments": [
                {"source": "B", "src_in": 452, "src_out": 556},
                {"source": "A", "src_in": 3275, "src_out": 3452},
                {"source": "A", "src_in": 3275, "src_out": 3452},
                {"source": "C", "src_in": 981, "src_out": 1183},
                {"source": "A", "src_in": 3528, "src_out": 3595},
                {"source": "A", "src_in": 3475, "src_out": 3650, "gain_db": -3.0, "label": "L"},
                {"source": "A", "src_in": 3475, "src_out": 3650, "gain_db": -3.0, "label": "L"},
            ],
            "head": {"video": "dissolve", "video_frames": 6},
        },
        "rows": [
            ("V1", 13, 46, 452, "CamB.mov", 1.0, "segments[0]"),
            ("V1", 59, 177, 3275, "CamA.mov", 1.0, "segments[1]"),
            ("V1", 236, 177, 3275, "CamA.mov", 1.0, "segments[2]"),
            ("V1", 442, 202, 981, "CamC.MOV", 1.0, "segments[3]"),
            ("V1", 644, 98, 3497, "CamA.mov", 1.0, "segments[4]"),
            ("V1", 742, 175, 3475, "CamA.mov", 1.0, "segments[5]"),
            ("V1", 917, 53, 10, "Fire.mov", 1.0, "foreign"),
        ],
    },
}

#: Seeds whose every tagged item lands in its own original (or an identical twin).
SEEDS_RESOLVED = (345, 459, 638, 785, 938, 974, 2053)
#: Seeds where overlap cannot tell (an extended small segment, a copy of another's range): the
#: item lands elsewhere and the diff says which other original it also overlaps.
SEEDS_REPORTED = (968, 2062, 2699, 2817, 3268)


def seed_plan(case: dict[str, Any]) -> EditPlan:
    data: dict[str, Any] = {
        "version": 2,
        "project": "P",
        "timeline": "T",
        "kind": "longform",
        "resolution": {"width": 1920, "height": 1080},
        "profile": "ccn",
        "sources": [
            {
                "id": sid,
                "media": SEED_FILES[f"Cam{sid}.{'MOV' if sid == 'C' else 'mov'}"],
                "duration": 4000,
                "start_timecode": 0,
                "frame_size": [1920, 1080],
            }
            for sid in "ABC"
        ]
        + [{"id": "O", "media": "E:\\Assets\\Outro.mov", "duration": 90, "start_timecode": 0}],
        "audio_tracks": [{"name": "A1"}, {"name": "A2"}],
        **case["plan"],
    }
    return EditPlan.model_validate(data)


def seed_rows(case: dict[str, Any]) -> list[Row]:
    rows = []
    for track, rin, dur, src, name, speed, tag in case["rows"]:
        path = "G:\\Stock\\CamA.mov" if tag == "samename" else SEED_FILES[name]
        sound = f"A{track[1:]}" if speed == 1.0 else None
        role = "outro" if tag == "outro" else ""
        rows.append(Row(track, "video", rin, dur, src, path, sound=sound, speed=speed, role=role))
    return rows


def owners(case: dict[str, Any], result: ImportResult) -> list[tuple[int, int, list[int]]]:
    """(true original, the row the item landed in, that piece's also_overlaps) per tagged item."""
    out = []
    for track, rin, _, src, _, speed, tag in case["rows"]:
        if not tag.startswith("segments[") or speed != 1.0:
            continue
        true = int(tag[len("segments[") : -1])
        hits = [
            (row["index"], piece["also_overlaps"])
            for row in result.diff["segments"]
            for piece in row["new"]
            if row["track"] == int(track[1:])
            and (piece["record_in"], piece["src_in"]) == (rin, src)
        ]
        assert len(hits) == 1, (tag, rin)
        out.append((true, *hits[0]))
    return out


def twins(plan: EditPlan, a: int, b: int) -> bool:
    sa, sb = plan.segments[a], plan.segments[b]
    return (sa.source, sa.src_in, sa.src_out) == (sb.source, sb.src_in, sb.src_out)


@pytest.mark.parametrize("seed", SEEDS_RESOLVED + SEEDS_REPORTED)
def test_validator_seeds_attribute_items_or_say_they_cannot(seed: int) -> None:
    case = SEED_CASES[seed]
    plan = seed_plan(case)
    rate = plan.fps.num / plan.fps.den
    result = run(plan, seed_rows(case), rate=rate)
    found = owners(case, result)
    wrong = [(t, o, also) for t, o, also in found if t != o and not twins(plan, t, o)]
    if seed in SEEDS_RESOLVED:
        assert wrong == []
    else:
        assert wrong and all(t in also for t, _, also in wrong)
        listed = {(u["index"], tuple(u["also_overlaps"])) for u in result.diff["uncertain_matches"]}
        assert listed


# -- H2: assignment as a whole ---------------------------------------------------------------------


def recap_plan() -> EditPlan:
    data = simple_data()
    data["sources"][0]["frame_size"] = [1920, 1080]
    data["segments"] = [
        {"source": "camA", "src_in": 900, "src_out": 1300, "label": "main"},
        {"source": "camB", "src_in": 0, "src_out": 100, "label": "cut"},
        {
            "source": "camA",
            "src_in": 1000,
            "src_out": 1100,
            "label": "recap",
            "reframe": [{"time": 0, "zoom": 2.0}],
        },
    ]
    data["titles"], data["markers"] = [], []
    return EditPlan.model_validate(data)


def test_a_trimmed_recap_keeps_its_own_fields() -> None:
    """Validator H2: main a 900..1300 and recap a 1000..1100 both trimmed; the recap item overlaps
    both equally (80) and is the recap's, not a piece of the main take."""
    plan = recap_plan()
    rows = layout_rows(plan)
    c = content(rows)
    main = replace(c[0], src_in=1000, duration=300)
    recap = replace(c[2], src_in=1020, duration=80)
    result = run(plan, ripple(rows, [main, c[1], recap]))
    assert statuses(result) == ["trimmed", "moved", "trimmed"]
    assert [(s.label, s.src_in, s.src_out) for s in result.plan.segments] == [
        ("main", 1000, 1300),
        ("cut", 0, 100),
        ("recap", 1020, 1100),
    ]
    assert [k.zoom for k in result.plan.segments[2].reframe] == [2.0]
    # the recap item overlapped the main take too: the diff says so
    assert result.diff["uncertain_matches"] == [
        {"index": 0, "assigned_to": 0, "why": "overlap", "also_overlaps": [2]},
        {"index": 2, "assigned_to": 2, "why": "overlap", "also_overlaps": [0]},
    ]


def test_assign_items_rules() -> None:
    # exact range, at its own position first (a range used twice)
    originals = [(0, "a", 0, 100, 0), (1, "a", 0, 100, 300)]
    assert assign_items([("a", 0, 100, 300), ("a", 0, 100, 0)], originals)[0] == {0: 1, 1: 0}
    # larger overlap wins over nearness
    originals = [(0, "a", 0, 100, 0), (1, "a", 50, 300, 500)]
    assert assign_items([("a", 40, 300, 0)], originals)[0] == {0: 1}
    # a tie goes to the original no item holds yet, then the nearer position
    originals = [(0, "a", 0, 100, 0), (1, "a", 0, 100, 900)]
    got, also = assign_items([("a", 10, 90, 0), ("a", 20, 80, 10)], originals)
    assert got == {0: 0, 1: 1} and also == {0: [1], 1: [0]}
    # an original matched exactly is not split by another item that overlaps it
    originals = [(0, "a", 0, 100, 0), (1, "a", 50, 120, 100)]
    assert assign_items([("a", 0, 100, 0), ("a", 40, 100, 100)], originals)[0] == {0: 0, 1: 1}
    # an item that only abuts an original (zero overlap) is not its piece
    assert assign_items([("a", 100, 150, 0)], [(0, "a", 0, 100, 0)])[0] == {}


def test_an_item_overlapping_an_exactly_matched_segment_takes_the_other() -> None:
    data = simple_data()
    data["segments"] = [
        {"source": "camA", "src_in": 0, "src_out": 100, "label": "one"},
        {"source": "camA", "src_in": 50, "src_out": 120, "label": "two"},
    ]
    data["titles"], data["markers"] = [], []
    plan = EditPlan.model_validate(data)
    rows = layout_rows(plan)
    c = content(rows)
    result = run(plan, ripple(rows, [c[0], replace(c[1], src_in=40, duration=60)]))
    assert statuses(result) == ["unchanged", "trimmed"]
    assert result.plan.segments[1].label == "two"


def test_an_abutting_item_is_added_not_a_trim() -> None:
    plan = simple()
    rows = layout_rows(plan)
    c = content(rows)
    after = replace(c[0], src_in=200, duration=50)
    result = run(plan, ripple(rows, [after, *c[1:]]))
    assert statuses(result)[0] == "removed"
    assert [(a["source"], a["src_in"]) for a in result.diff["added"]] == [("camA", 200)]


# -- L1: joined neighbours ---------------------------------------------------------------------


def test_joined_neighbours_are_reported_as_joined() -> None:
    data = simple_data()
    data["segments"] = [
        {"source": "camA", "src_in": 0, "src_out": 100, "label": "one"},
        {"source": "camA", "src_in": 100, "src_out": 200, "label": "two"},
        {"source": "camB", "src_in": 0, "src_out": 100},
    ]
    data["titles"], data["markers"] = [], []
    plan = EditPlan.model_validate(data)
    rows = layout_rows(plan)
    c = content(rows)
    joined = replace(c[0], duration=190)
    result = run(plan, ripple(rows, [joined, c[2]]))
    rows_ = result.diff["segments"]
    assert statuses(result) == ["joined", "joined", "moved"]
    assert rows_[0]["joined"] == rows_[1]["joined"] == {"segments": [0, 1], "fields_from": 0}
    assert (rows_[0]["head_frames"], rows_[0]["tail_frames"]) == (0, 10)
    assert rows_[1]["new"] == []
    assert [(s.label, s.src_in, s.src_out) for s in result.plan.segments] == [
        ("one", 0, 190),
        (None, 0, 100),
    ]


# -- H1: media matching ------------------------------------------------------------------------


def card_plan() -> EditPlan:
    data = simple_data()
    data["sources"][0]["media"] = "E:\\Show\\CardA\\C0001.MP4"
    data["titles"], data["markers"] = [], []
    return EditPlan.model_validate(data)


def test_a_different_file_with_a_sources_name_is_foreign() -> None:
    """Validator H1: camera cards reuse names; CardB\\C0001.MP4 is not CardA\\C0001.MP4."""
    plan = card_plan()
    rows = layout_rows(plan)
    c = content(rows)
    other = replace(c[0], path="E:\\Show\\CardB\\C0001.MP4", src_in=10, duration=80)
    result = run(plan, ripple(rows, [other, *c[1:]]))
    assert statuses(result)[0] == "removed"
    assert [(f["track"], f["media"], f["reason"]) for f in result.diff["foreign"]] == [
        ("V1", "E:\\Show\\CardB\\C0001.MP4", "same_name_different_file")
    ]
    assert result.plan.segments[0].label is None and result.plan.segments[0].source == "camB"
    assert result.diff["matched_by_file_name"] == []


def test_a_same_name_file_past_the_sources_length_is_foreign_not_a_refusal() -> None:
    plan = card_plan()
    rows = layout_rows(plan)
    c = content(rows)
    other = Row("V1", "video", 0, 40, 4990, "E:\\Show\\CardB\\C0001.MP4", sound="A1")
    result = run(plan, ripple(rows, [*c, other]))
    assert [f["reason"] for f in result.diff["foreign"]] == ["same_name_different_file"]
    assert set(statuses(result)) == {"unchanged"}


def test_file_name_matching_only_for_a_source_with_no_edit_pc_path_and_listed() -> None:
    data = simple_data()
    data["sources"][0]["media"] = "/srv/media/day1/A.mov"  # a hub path
    data["titles"], data["markers"] = [], []
    plan = EditPlan.model_validate(data)
    rows = layout_rows(plan, media_map_from_mapping(MEDIA_MAP))  # exported with the map
    result = run(plan, rows)  # imported without it: camA's edit-PC path is unknown
    assert set(statuses(result)) == {"unchanged"}
    by_name = result.diff["matched_by_file_name"]
    assert {(e["track"], e["record_in"], e["source"]) for e in by_name} == {
        ("V1", 0, "camA"),
        ("V1", 400, "camA"),
        ("A1", 0, "camA"),
        ("A1", 400, "camA"),
    }
    assert result.diff["segments"][0]["new"][0]["match"] == "file-name"
    assert result.diff["media_matched_by"] == ["file-name", "path"]
    text = _import_text(result.diff, "o.json", "r.json")  # type: ignore[arg-type]
    assert "matched by file name only (4): V1 0 E:\\Media\\day1\\A.mov -> camA" in text


def test_a_file_name_two_unplaced_sources_share_is_foreign() -> None:
    data = simple_data()
    data["sources"][0]["media"] = "/srv/media/day1/A.mov"
    data["sources"].append({"id": "dup", "media": "/srv/media/day2/A.mov", "duration": 5000})
    data["titles"], data["markers"] = [], []
    plan = EditPlan.model_validate(data)
    result = run(plan, layout_rows(plan, media_map_from_mapping(MEDIA_MAP)))
    assert {f["reason"] for f in result.diff["foreign"]} == {"ambiguous_file_name"}
    assert statuses(result) == ["removed", "unchanged", "removed", "unchanged"]


@pytest.mark.parametrize(
    ("item_path", "matches"),
    [
        ("E:\\Show\\Late\\A.mov", True),  # the export's path with \\ for /
        ("e:/show/late/a.MOV", True),  # Windows paths compare without case
        ("E:\\Show\\Late\\A.mov.bak", False),
        ("E:\\Show\\Late2\\A.mov", False),  # the same name in another folder
        ("\\\\nas\\Show\\Late\\A.mov", False),
    ],
)
def test_path_matching_by_literal_examples(item_path: str, matches: bool) -> None:
    """The expected answers are written out here, not computed with the product's media_key."""
    data = simple_data()
    data["sources"][0]["media"] = "E:/Show/Late/A.mov"
    data["titles"], data["markers"] = [], []
    plan = EditPlan.model_validate(data)
    rows = layout_rows(plan)
    moved = [replace(r, path=item_path) if r.path == "E:/Show/Late/A.mov" else r for r in rows]
    result = run(plan, moved)
    if matches:
        assert set(statuses(result)) == {"unchanged"}
        assert result.diff["foreign"] == []
    else:
        assert statuses(result) == ["removed", "unchanged", "removed", "unchanged"]
        assert len(result.diff["foreign"]) == 2


def test_two_sources_on_one_file_pick_by_overlap_then_plan_order() -> None:
    data = simple_data()
    data["sources"].append({"id": "alias", "media": "E:\\Show\\A.mov", "duration": 5000})
    data["segments"][2]["source"] = "alias"
    data["titles"], data["markers"] = [], []
    plan = EditPlan.model_validate(data)
    rows = layout_rows(plan)
    c = content(rows)
    # overlaps camA 0..200 by 50 and alias 500..700 by 50: the first in plan order, camA
    both = Row("V1", "video", 0, 400, 150, "E:\\Show\\A.mov", sound="A1")
    # overlaps neither: the first source on the file in plan order, camA
    neither = Row("V1", "video", 0, 50, 3000, "E:\\Show\\A.mov", sound="A1")
    result = run(plan, ripple(rows, [c[1], c[3], both, neither]))
    assert [(a["source"], a["src_in"]) for a in result.diff["added"]] == [("camA", 3000)]
    assert result.diff["segments"][0]["new"][0]["src_in"] == 150


# -- H3: reader issues -------------------------------------------------------------------------


def test_reader_issue_codes_that_can_hide_items_are_not_warnings() -> None:
    for code in (
        "scan-truncated",
        "scan-unparsed-markup",
        "missing-container",
        "missing-member",
        "unreadable-member",
        "blob-decode",
        "unresolved-media",
        "unknown-track",
        "path-from-later-stream",
        "scan-budget-exceeded",
    ):
        assert code not in READER_WARNING_CODES
    assert {"uninterpreted", "linked-files", "stale-media-file-path"} <= READER_WARNING_CODES


def test_reader_warnings_are_passed_through_to_the_diff() -> None:
    plan = simple()
    timeline, media = build_timeline(layout_rows(plan), rate=25.0)
    warnings = [Issue("info", "uninterpreted", "2 payload(s) not interpreted")]
    result = import_timeline(plan, timeline, media, profile=permissive(), reader_warnings=warnings)
    assert result.diff["reader_warnings"] == {
        "count": 1,
        "listed": [
            {"severity": "info", "code": "uninterpreted", "message": "2 payload(s) not interpreted"}
        ],
    }


# -- H4: timing off the primary track ----------------------------------------------------------


def broll_plan() -> EditPlan:
    """The validator's p_v2ripple plan: b-roll over B on V2, a title and a caption over B."""
    return EditPlan.model_validate(
        {
            "version": 2,
            "project": "P",
            "timeline": "T",
            "kind": "longform",
            "fps": 25,
            "resolution": {"width": 1920, "height": 1080},
            "profile": "ccn",
            "sources": [
                {"id": s, "media": f"E:\\Show\\{s}.mov", "duration": 9000, "start_timecode": 0}
                for s in "ABG"
            ],
            "segments": [
                {"source": "A", "src_in": 0, "src_out": 200},
                {"source": "B", "src_in": 0, "src_out": 200},
                {
                    "source": "G",
                    "track": 2,
                    "src_in": 0,
                    "src_out": 50,
                    "record_in": 250,
                    "label": "broll over B",
                },
            ],
            "titles": [{"lines": ["name of B"], "record_in": 220, "record_out": 300}],
            "captions": [{"lines": ["words in B"], "record_in": 260, "record_out": 290}],
            "markers": [{"record": 210, "name": "B starts"}],
        }
    )


def test_a_ripple_delete_moves_broll_titles_captions_and_markers() -> None:
    """Validator H4: Resolve ripples every track; the plan follows the timeline for items and
    the content for captions and markers, and lists every move."""
    plan = broll_plan()
    rows = layout_rows(plan)
    first = content(rows)[0]
    edited = [replace(r, record_in=r.record_in - 200) for r in rows if r is not first]
    result = run(plan, edited)
    assert statuses(result) == ["removed", "moved", "moved"]
    assert result.diff["segments"][2]["new"][0]["record_in"] == 50
    assert [
        (s.source, s.track, start)
        for s, (start, _) in zip(result.plan.segments, result.plan.record_ranges(), strict=True)
    ] == [
        ("B", 1, 0),
        ("G", 2, 50),
    ]
    assert result.plan.segments[1].label == "broll over B"
    assert [(t.record_in, t.record_out) for t in result.plan.titles] == [(20, 100)]
    assert [(c.record_in, c.record_out) for c in result.plan.captions] == [(60, 90)]
    assert [m.record for m in result.plan.markers] == [10]
    assert [
        (e["location"], e["by"], e["new"]["record_in"]) for e in result.diff["repositioned"]
    ] == [
        ("titles[0]", "content", 20),
        ("captions[0]", "content", 60),
        ("markers[0]", "content", 10),
    ]
    assert result.diff["not_carried"] == [] and result.diff["dropped"] == []


def test_captions_and_markers_over_removed_content_are_dropped() -> None:
    plan = broll_plan()
    rows = layout_rows(plan)
    second = content(rows)[1]
    # B deleted with a ripple: V2 b-roll and title deleted with it
    edited = [r for r in rows if r is not second and r.track == "V1"]
    result = run(plan, edited)
    assert codes(result, "dropped") == [
        "title_removed",
        "caption_content_gone",
        "marker_content_gone",
    ]
    assert statuses(result) == ["unchanged", "removed", "removed"]
    assert result.plan.captions == [] and result.plan.markers == [] and result.plan.titles == []


def test_a_caption_across_a_cut_whose_sides_were_swapped_is_dropped() -> None:
    data = simple_data()
    data["captions"] = [{"lines": ["across"], "record_in": 190, "record_out": 210}]
    data["titles"], data["markers"] = [], []
    plan = EditPlan.model_validate(data)
    rows = layout_rows(plan)
    c = content(rows)
    swapped = run(plan, ripple(rows, [c[1], c[0], *c[2:]]))
    assert codes(swapped, "dropped") == ["caption_content_gone"]
    kept = run(plan, ripple(rows, [replace(c[0], src_in=50, duration=150), *c[1:]]))
    assert [(x.record_in, x.record_out) for x in kept.plan.captions] == [(140, 160)]


def test_a_marker_whose_last_frame_is_cut_away_is_dropped() -> None:
    data = simple_data()
    data["markers"] = [{"record": 550, "name": "long", "duration": 60}]
    data["titles"] = []
    plan = EditPlan.model_validate(data)
    rows = layout_rows(plan)
    c = content(rows)
    result = run(plan, ripple(rows, c[:3]))  # the fourth segment, under 600..609, is gone
    assert codes(result, "dropped") == ["marker_content_gone"]


def test_an_added_clip_on_a_planned_track_is_added_there() -> None:
    plan, rows = load_plan(TC_2997), layout_rows(load_plan(TC_2997))
    extra = Row("V2", "video", 700, 40, 300, "\\\\nas\\media\\late\\crowd.mov", sound="A2")
    result = run(plan, [*rows, extra], rate=RATE_2997)
    assert [(a["track"], a["source"], a["record_in"]) for a in result.diff["added"]] == [
        (2, "broll", 700)
    ]
    added = result.plan.segments[result.diff["added"][0]["index"]]
    assert (added.track, added.src_in, added.record_in) == (2, 300, 700)


def test_a_deleted_broll_clip_is_removed_from_the_plan() -> None:
    plan = load_plan(TC_2997)
    rows = [r for r in layout_rows(plan) if r.track != "V2"]
    result = run(plan, rows, rate=RATE_2997)
    assert result.diff["segments"][3]["status"] == "removed"
    assert [s.track for s in result.plan.segments] == [1, 1, 1]


def test_items_the_plan_cannot_place_are_not_carried_or_foreign() -> None:
    plan = load_plan(TC_2997)
    rows = layout_rows(plan)
    known_on_v6 = Row("V6", "video", 100, 20, 50, "E:/Show/Late/A.mov")
    unknown_on_v2 = Row("V2", "video", 600, 20, 0, "H:\\Else\\X.mov")
    title = Row("V3", "title", 700, 30)
    result = run(plan, [*rows, known_on_v6, unknown_on_v2, title], rate=RATE_2997)
    picture = [e for e in result.diff["not_carried"] if e["code"] != "audio_only_edit"]
    assert [(e["code"], e["track"]) for e in picture] == [
        ("title_added", "V3"),
        ("other_picture_track", "V6"),
    ]
    assert [(f["track"], f["reason"]) for f in result.diff["foreign"]] == [("V2", "unknown_file")]


def test_a_title_moved_in_resolve_takes_its_new_place() -> None:
    plan = load_plan(TC_2997)
    rows = layout_rows(plan)
    title = next(r for r in rows if r.kind == "title" and r.record_in == 30)
    moved = [replace(r, record_in=60) if r is title else r for r in rows]
    result = run(plan, moved, rate=RATE_2997)
    assert [(t.record_in, t.record_out) for t in result.plan.titles] == [(60, 180)]
    assert result.plan.titles[0].lines == plan.titles[0].lines
    assert [e["location"] for e in result.diff["repositioned"]] == ["titles[0]"]


def test_a_title_ending_on_the_last_frame_stays_one_past_it_is_dropped() -> None:
    plan = simple()
    rows = layout_rows(plan)
    c = content(rows)
    total = 600 + 100  # three segments and the outro
    title = next(r for r in rows if r.kind == "title")
    on_end = [replace(r, record_in=total - 90) if r is title else r for r in ripple(rows, c[:3])]
    assert [(t.record_out) for t in run(plan, on_end).plan.titles] == [total]
    past = [replace(r, record_in=total - 89) if r is title else r for r in ripple(rows, c[:3])]
    assert codes(run(plan, past), "dropped")[0] == "title_past_end"


# -- M1, M3, L5: items a plan cannot hold ------------------------------------------------------


def test_stills_are_not_carried() -> None:
    plan = simple()
    rows = layout_rows(plan)
    c = content(rows)
    photo = Row("V1", "video", 0, 48, 86400, "E:\\Show\\Photo.PNG")
    result = run(plan, ripple(rows, [c[0], photo, *c[1:]]))
    assert codes(result) == ["still_image"]
    # a .mov the media pool says has one frame is a still too, not a retime
    timeline, media = build_timeline(ripple(rows, [c[0], photo, *c[1:]]), rate=25.0)
    media = {
        k: dataclasses.replace(m, frames=1) if m.path == "E:\\Show\\B.mov" else m
        for k, m in media.items()
    }
    result = import_timeline(plan, timeline, media, profile=permissive())
    # the photo, and both camB pictures with their sounds
    assert codes(result).count("still_image") == 5
    # each camB still stands where its plan segment was: the plan keeps its own version
    assert statuses(result) == ["unchanged", "kept", "moved", "kept"]
    assert [result.diff["segments"][i]["kept_because"] for i in (1, 3)] == ["still_image"] * 2


@pytest.mark.parametrize(
    ("change", "code"),
    [
        ({"src_in": -5}, "negative_source_in"),
        ({"record_in": -10}, "outside_timeline"),
        ({"duration": 0}, "zero_length"),
    ],
)
def test_hostile_item_values_are_not_carried(change: dict[str, int], code: str) -> None:
    plan = simple()
    rows = layout_rows(plan)
    first = content(rows)[0]
    edited = [replace(r, sound=None, **change) if r is first else r for r in rows]
    result = run(plan, edited)
    assert codes(result)[0] == code
    assert statuses(result)[0] == "removed"
    assert result.plan.segments[0].src_in >= 0


def test_a_speed_other_than_100_percent_is_retimed_whatever_the_timemap_says() -> None:
    plan = simple()
    timeline, media = build_timeline(layout_rows(plan), rate=25.0)
    v1 = next(t for t in timeline.tracks if t.label == "V1")
    v1.items[1] = dataclasses.replace(v1.items[1], speed=0.5, timemap_form=None)
    result = import_timeline(plan, timeline, media, profile=permissive())
    assert codes(result)[0] == "retimed"
    assert statuses(result)[1] == "kept"


def test_unreadable_items_are_not_carried() -> None:
    plan = simple()
    timeline, media = build_timeline(layout_rows(plan), rate=25.0)
    v1 = next(t for t in timeline.tracks if t.label == "V1")
    v1.items[1] = dataclasses.replace(v1.items[1], record_start=None)
    v1.items[2] = dataclasses.replace(v1.items[2], source_in=None)
    result = import_timeline(plan, timeline, media, profile=permissive())
    assert codes(result)[:2] == ["unreadable_item", "unreadable_item"]
    assert statuses(result)[1:3] == ["removed", "removed"]


# -- M2: the export's layout runs on the edited plan --------------------------------------------


def test_an_edited_plan_the_export_refuses_sets_export_errors() -> None:
    data = simple_data()
    data["head"] = {"video": "dissolve", "video_frames": 6}
    data["titles"], data["markers"] = [], []
    plan = EditPlan.model_validate(data)
    rows = layout_rows(plan)
    c = content(rows)
    result = run(plan, ripple(rows, [replace(c[0], duration=3), *c[1:]]))
    assert result.diff["lint_errors"] is False
    assert result.diff["export_errors"] is True and result.has_errors
    assert result.diff["export_check"]["error"]["code"] == "transitions_overlap"
    clean = run(plan, rows)
    assert clean.diff["export_check"] == {"ok": True, "error": None} and not clean.has_errors


# -- the outro and the diff's own flags -------------------------------------------------------


def test_an_outro_after_a_gap_is_reported_and_a_missing_one_kept() -> None:
    plan = simple()
    rows = layout_rows(plan)
    late = [replace(r, record_in=r.record_in + 12) if r.role == "outro" else r for r in rows]
    result = run(plan, late)
    assert codes(result) == ["outro_position"]
    assert result.plan.outro == plan.outro
    gone = [r for r in rows if r.role != "outro"]
    result = run(plan, gone)
    assert codes(result) == ["outro_missing"] and result.diff["outro"]["status"] == "missing"
    assert result.plan.outro == plan.outro


def test_an_outro_cut_to_exactly_its_sting_is_trimmed() -> None:
    plan = load_plan(TC_2997)  # sting 10 + 60 frames
    rows = layout_rows(plan)
    exact = [replace(r, duration=70) if r.role == "outro" else r for r in rows]
    assert run(plan, exact, rate=RATE_2997).diff["outro"]["status"] == "trimmed"
    short = [replace(r, duration=69) if r.role == "outro" else r for r in rows]
    assert run(plan, short, rate=RATE_2997).diff["outro"]["status"] == "kept"


def test_a_not_carried_entry_alone_makes_the_diff_not_unchanged() -> None:
    plan = simple()
    music = Row("A2", "audio", 0, 300, 0, "E:\\Music\\bed.wav")
    result = run(plan, [*layout_rows(plan), music])
    assert set(statuses(result)) == {"unchanged"} and result.diff["unchanged"] is False


def test_a_source_range_ending_on_the_sources_last_frame_is_accepted() -> None:
    plan = simple()
    rows = layout_rows(plan)
    c = content(rows)
    at_end = replace(c[3], src_in=4900, duration=100)
    assert ranges(run(plan, ripple(rows, [*c[:3], at_end])))[3][2] == 5000
    with pytest.raises(TimelineImportError) as caught:
        run(plan, ripple(rows, [*c[:3], replace(c[3], src_in=4901, duration=100)]))
    assert caught.value.code == "source_range_outside"


# -- M4, L2: a bounded stdout --------------------------------------------------------------------


def test_the_text_summary_groups_and_bounds_repeated_entries() -> None:
    plan = simple()
    rows = layout_rows(plan)
    c = content(rows)
    pieces = [replace(c[1], src_in=200 + 5 * n, duration=5) for n in range(40)]
    sounds = [Row(f"A{2 + n % 2}", "audio", 10 * n, 5, 0, "E:\\Music\\bed.wav") for n in range(9)]
    result = run(plan, [*ripple(rows, [c[0], *pieces, *c[2:]]), *sounds])
    text = _import_text(result.diff, "o.json", "r.json")  # type: ignore[arg-type]
    assert "segments[1] split: 200..400 -> [1] 200..205, [2] 205..210, [3] 210..215, " in text
    assert "... 40 pieces in all" in text
    assert "audio_only_edit x9 (A2 x5, A3 x4)" in text
    assert len(text.splitlines()) < 20 and max(len(line) for line in text.splitlines()) < 400
    assert len(result.diff["segments"][1]["new"]) == 40  # the JSON keeps every piece
