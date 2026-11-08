"""Shared helpers for the FCPXML export tests: fixtures, a permissive profile, a seeded random
plan generator, and the round-trip check."""

from __future__ import annotations

import itertools
import random
from pathlib import Path
from typing import Any

from plan_support import FIXTURES

from aar.plan.export import ExportResult, export_fcpxml, media_map_from_mapping
from aar.plan.export.reader import read_fcpxml
from aar.plan.export.timebase import SUPPORTED_RATES
from aar.plan.export.verify import VerifyResult, verify_fcpxml
from aar.plan.model import EditPlan
from aar.plan.profile import Profile, load_profile, parse_profile

EXPORT_FIXTURES = FIXTURES / "export"
#: name -> plan file of every golden export fixture.
GOLDEN_PLANS: dict[str, Path] = {
    # The S13h2 fixture with frame sizes (that file is pinned by the compiler's phase results).
    "ccn_longform_v2": EXPORT_FIXTURES / "ccn_longform_v2.plan.json",
    "short_reframe": EXPORT_FIXTURES / "short_reframe.plan.json",
    "ccn_2997": EXPORT_FIXTURES / "ccn_2997.plan.json",
}
#: Golden plans that also get a ``<name>.transition`` golden (audio-only crossfades).
TRANSITION_GOLDENS = ("ccn_longform_v2", "ccn_2997")
MEDIA_MAP = {"/srv/media": "E:\\Media", "/srv/media/ccn": "\\\\nas\\ccn"}
_QA = {"longform_lufs": -22.0, "short_lufs": -14.5, "lufs_tolerance": 0.5, "review_mp4_max_mb": 20}


def load_plan(path: Path) -> EditPlan:
    return EditPlan.model_validate_json(path.read_text(encoding="utf-8"))


def ccn() -> Profile:
    return load_profile("ccn")


def permissive(timeline_start: str = "01:00:00:00") -> Profile:
    """Only ``ranges-valid``: random plans need not follow the CCN house rules."""
    return parse_profile(
        {
            "name": "export-test",
            "qa": _QA,
            "rules": {"ranges-valid": {}},
            "export": {"timeline_start": timeline_start},
        }
    )


def export(plan: EditPlan, profile: Profile | None = None, **kwargs: Any) -> ExportResult:
    return export_fcpxml(plan, profile or permissive(), **kwargs)


def reverify(result: ExportResult) -> VerifyResult:
    """Verify the written text again against a layout rebuilt from the plan."""
    from aar.plan.export.layout import build_layout

    layout = result.layout
    fresh = build_layout(
        layout.plan,
        media_map=media_map_from_mapping(MEDIA_MAP),
        timeline_start=layout.timeline_start,
        audio_crossfade=layout.audio_crossfade,
    )
    return verify_fcpxml(result.fcpxml, fresh)


def structure_problems(text: str) -> list[str]:
    return read_fcpxml(text).problems


# -- the random plan generator (property test) -------------------------------------------------

_RATES = sorted(SUPPORTED_RATES)
_SIZES = [(1920, 1080), (1080, 1920), (3840, 2160), (1280, 720), (2048, 858)]
#: Source picture sizes: the timeline sizes, other aspects, and a face track's sampled size.
_FRAME_SIZES = [
    (1920, 1080),
    (3840, 2160),
    (1080, 1920),
    (1440, 1080),
    (1080, 1080),
    (640, 360),
    (4096, 1716),
    (720, 1280),
]


def crossfade_form(seed: int) -> str:
    """The audio-crossfade form a random plan is exported with: both, by seed parity."""
    return "split" if seed % 2 == 0 else "transition"


_TEXT = ["Jane & Joe", "<b>Loud</b>", 'say "hi"', "it's", "Café ✨", "tab\there", "a\nb"]


def _media(rng: random.Random, index: int) -> str:
    return rng.choice(
        [
            f"E:/Media/Ep 1/clip {index}.mov",
            f"E:\\Show\\cam{index}#1.braw",
            f"\\\\nas\\share\\c{index}.mxf",
            f"file:///E:/M/c{index}%20x.mov",
            f"/srv/media/day1/c{index}.mov",
            f"/srv/media/ccn/c{index}.mov",
        ]
    )


def random_plan(seed: int) -> tuple[dict[str, Any], str]:
    """A random plan that lints clean under :func:`permissive` and can be exported: handles
    and lengths are chosen so every transition fits. Returns (plan data, timeline start)."""
    rng = random.Random(seed)
    rate = rng.choice(_RATES)
    width, height = rng.choice(_SIZES)
    sources: list[dict[str, Any]] = []
    for index in range(rng.randint(1, 4)):
        source: dict[str, Any] = {"id": f"s{index}", "media": _media(rng, index)}
        if rng.random() < 0.85:
            source["duration"] = rng.randint(3000, 6000)
        if rng.random() < 0.85:
            source["frame_size"] = list(rng.choice(_FRAME_SIZES))
        sources.append(source)
    known = [s for s in sources if "duration" in s]

    def pick_source(need_duration: bool = False) -> dict[str, Any]:
        return rng.choice(known if need_duration and known else sources)

    segments: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    cursor = 0
    v1: list[int] = []
    for _ in range(rng.randint(0, 6)):
        if rng.random() < 0.2:
            cursor += rng.randint(1, 40)  # a hole on V1
        src = pick_source()
        length = rng.randint(45, 160)
        src_in = rng.choice([0, rng.randint(0, 25), rng.randint(30, 400)])
        seg: dict[str, Any] = {
            "track": 1,
            "source": src["id"],
            "src_in": src_in,
            "src_out": src_in + length,
            "record_in": cursor,
        }
        if "duration" in src:
            seg["src_out"] = min(seg["src_out"], src["duration"] - rng.choice([0, 5, 40]))
            seg["src_in"] = seg["src_out"] - length
            if seg["src_in"] < 0:
                seg["src_in"], seg["src_out"] = 0, length
        _decorate(rng, seg)
        segments.append(seg)
        v1.append(len(segments) - 1)
        cursor += length
    for other_track in range(2, rng.randint(2, 4)):
        start = rng.randint(0, max(cursor, 60))
        for _ in range(rng.randint(0, 2)):
            src = pick_source()
            length = rng.randint(25, 90)
            src_in = rng.randint(0, 200)
            if "duration" in src:
                src_in = min(src_in, src["duration"] - length)
            seg = {
                "track": other_track,
                "source": src["id"],
                "src_in": src_in,
                "src_out": src_in + length,
                "record_in": start,
            }
            _decorate(rng, seg)
            segments.append(seg)
            start += length + rng.randint(0, 30)
    sources_by_id = {s["id"]: s for s in sources}
    for seg in segments:  # a moved clip needs its source's size (source_size_required)
        if "frame_size" not in sources_by_id[seg["source"]]:
            seg.pop("transform", None)
            seg.pop("reframe", None)
    for a, b in itertools.pairwise(v1):
        out_seg, in_seg = segments[a], segments[b]
        if out_seg["record_in"] + out_seg["src_out"] - out_seg["src_in"] != in_seg["record_in"]:
            continue  # not abutting: a transition here would be reported, not exported
        if rng.random() < 0.55:
            d = rng.randint(1, 20)
            dur = sources_by_id[out_seg["source"]].get("duration")
            tail = None if dur is None else dur - out_seg["src_out"]
            head = in_seg["src_in"]
            fits = (tail is not None and tail >= d - d // 2 and head >= d // 2) or (
                (tail is not None and tail >= d) or head >= d
            )
            if not fits:
                continue
            video = rng.random() < 0.4
            tx: dict[str, Any] = {"after": a, "video": "dissolve" if video else "cut"}
            tx["video_frames"] = d if video else 0
            if not video or rng.random() < 0.7:
                tx["audio"] = {"type": "Cross Fade 0 dB", "frames": d}
            transitions.append(tx)
    data: dict[str, Any] = {
        "version": 2,
        "project": rng.choice(["CCN", "Show & Tell"]),
        "timeline": f"Random {seed}",
        "kind": "longform",
        "fps": {"num": rate.numerator, "den": rate.denominator},
        "resolution": {"width": width, "height": height},
        "profile": "export-test",
        "sources": sources,
        "segments": segments,
        "transitions": transitions,
    }
    tracks = [
        {
            "name": f"A{i + 1}",
            "gain_db": rng.choice([0.0, -1.0, 2.5, rng.uniform(-12, 6)]),
            "mute": rng.random() < 0.3,
            **({"bus": "A5"} if rng.random() < 0.3 else {}),
        }
        for i in range(rng.randint(0, 6))
    ]
    data["audio_tracks"] = tracks
    if segments and rng.random() < 0.5:
        data["head"] = {
            "video": rng.choice(["cut", "dissolve", "dip"]),
            "audio_fade_frames": rng.randint(0, 20),
        }
        data["head"]["video_frames"] = 0 if data["head"]["video"] == "cut" else rng.randint(1, 20)
    if tracks and known and rng.random() < 0.6:
        src = rng.choice(known)
        src_in = rng.randint(20, 30)  # >= every outro dissolve, so "left" always fits
        outro: dict[str, Any] = {
            "source": src["id"],
            "src_in": src_in,
            "src_out": src_in + rng.randint(40, 120),
            "track": rng.choice([1, 1, 1, 2]),
            "audio_track": rng.choice(tracks)["name"],
        }
        if rng.random() < 0.6:
            outro["transition"] = {"video": "dissolve", "video_frames": rng.randint(1, 20)}
        if rng.random() < 0.5:
            outro["sting"] = {"lines": [rng.choice(_TEXT)], "start": 2, "frames": 30}
        data["outro"] = outro
    if rng.random() < 0.5:
        data["end_fade_frames"] = rng.randint(0, 3)
    total = _total(data)
    if total > 0:
        data["markers"] = [
            {
                "record": rng.randint(0, total - 1),
                "name": rng.choice(["", *_TEXT]),
                **({"note": rng.choice(_TEXT)} if rng.random() < 0.5 else {}),
                **({"color": "Red"} if rng.random() < 0.3 else {}),
            }
            for _ in range(rng.randint(0, 3))
        ]
        data["titles"] = []
        for _ in range(rng.randint(0, 2)):
            start = rng.randint(0, total - 1)
            data["titles"].append(
                {
                    "kind": rng.choice(["title", "lower-third"]),
                    "lines": [rng.choice(_TEXT) for _ in range(rng.randint(1, 3))],
                    "record_in": start,
                    "record_out": rng.randint(start + 1, total),
                    **({"template": "CCN Lower Third"} if rng.random() < 0.5 else {}),
                }
            )
    start = rng.choice(["01:00:00:00", "00:00:00:00", "10:00:00:00", "00:59:58:05"])
    return data, start


def _decorate(rng: random.Random, seg: dict[str, Any]) -> None:
    if rng.random() < 0.4:
        seg["gain_db"] = rng.choice([-3.0, 1.5, rng.uniform(-20, 10)])
    if rng.random() < 0.3:
        seg["transform"] = {
            "zoom": rng.uniform(0.5, 2.0),
            "pan": rng.uniform(-300, 300),
            "tilt": rng.uniform(-200, 200),
            "rotation_deg": rng.choice([0.0, 90.0, rng.uniform(-45, 45)]),
            "flip_h": rng.random() < 0.2,
            "flip_v": rng.random() < 0.1,
        }
    if rng.random() < 0.3:
        length = seg["src_out"] - seg["src_in"]
        times = sorted(rng.sample(range(0, length + 1), rng.randint(1, 4)))
        seg["reframe"] = [
            {
                "time": t,
                "x": rng.uniform(0, 1),
                "y": rng.uniform(0, 1),
                "zoom": rng.uniform(0.8, 2.0),
            }
            for t in times
        ]
    if rng.random() < 0.2:
        seg["label"] = rng.choice(_TEXT)
    if rng.random() < 0.2:
        seg["notes"] = rng.choice(_TEXT)


def _total(data: dict[str, Any]) -> int:
    plan = EditPlan.model_validate(data)
    return plan.total_frames()
