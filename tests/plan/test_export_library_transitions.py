"""S25e: ``aar plan export --format library`` writes a plan's audio-only crossfades as sound
transitions, from the FCPXML export's own layout, and still refuses what it cannot carry.

Synthetic 29.97 seed (holding a crossfade, the transition template), template and facts; two
camera files of 3,000 frames. A round-trip property: random cuts with crossfades on their
joins, written, verified, imported back: equal to the plan."""

from __future__ import annotations

import random
import sqlite3
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "drp"))

from libwrite_support import (
    crossfade,
    facts_lines,
    linked_pair,
    make_library,
    mp4_facts,
    mp4_template,
    mp4_template_facts,
    write_seed,
)

from aar.drp.library import LibraryOptions
from aar.drp.mediapool.gate import load_facts
from aar.plan.export.errors import ExportError
from aar.plan.export.library import export_library
from aar.plan.model import EditPlan
from aar.plan.profile import parse_profile
from aar.plan.timeline_import import import_timeline, read_timeline

pytest.importorskip("zstandard")

OPTS = LibraryOptions(settle_seconds=0.0)
CLIP = mp4_template()
CAM_A = "X:\\Show\\Footage\\camera-a.MP4"
CAM_C = "X:\\Show\\Footage\\camera-c.MP4"
FRAMES = 3000
START = 108000
PROFILE = {
    "name": "export-test",
    "qa": {
        "longform_lufs": -22.0,
        "short_lufs": -14.5,
        "lufs_tolerance": 0.5,
        "review_mp4_max_mb": 20,
    },
    "rules": {"ranges-valid": {}},
    "export": {"timeline_start": "01:00:00:00"},
}


def _facts(path: str) -> dict[str, Any]:
    data = mp4_facts(
        path=path,
        video_width=1920,
        video_height=1080,
        video_coded_width=1920,
        video_coded_height=1080,
        video_r_frame_rate="30000/1001",
        video_avg_frame_rate="30000/1001",
        video_time_base="1/30000",
        video_duration_ts=FRAMES * 1001,
        video_nb_frames=str(FRAMES),
        audio_duration_ts=FRAMES * 1001 * 48000 // 30000,
        data_time_base="1/30000",
        data_duration_ts=FRAMES * 1001,
    )
    data["ffprobe"]["format"]["duration"] = f"{FRAMES * 1001 / 30000:.6f}"
    return data


def _plan(segments: list[dict[str, Any]], transitions: list[dict[str, Any]], **change: Any) -> Any:
    data: dict[str, Any] = {
        "version": 2,
        "project": "Show",
        "timeline": "Cut 1",
        "kind": "longform",
        "fps": "30000/1001",
        "resolution": {"width": 1920, "height": 1080},
        "profile": "export-test",
        "sources": [
            {"id": "a", "media": CAM_A, "duration": FRAMES},
            {"id": "c", "media": CAM_C, "duration": FRAMES},
        ],
        "segments": segments,
        "transitions": transitions,
    }
    data.update(change)
    return EditPlan.model_validate(data)


THREE = [
    {"track": 1, "source": "a", "src_in": 100, "src_out": 160},
    {"track": 1, "source": "c", "src_in": 300, "src_out": 372},
    {"track": 1, "source": "a", "src_in": 900, "src_out": 950},
]


def _fade(after: int, frames: int = 2, kind: str = "Cross Fade 0 dB") -> dict[str, Any]:
    return {"after": after, "video": "cut", "audio": {"type": kind, "frames": frames}}


def _setup(tmp_path: Path, *, template: bool = True) -> dict[str, Path]:
    seeds = tmp_path / "seeds"
    seeds.mkdir(exist_ok=True)
    seed = write_seed(
        seeds / "seed.db",
        rate=30000 / 1001,
        start_frame=START,
        rate_text="29.97",
        resolution=(1920, 1080),
        clips=[CLIP],
        items=crossfade(CLIP, 0xB000, start=START)
        if template
        else linked_pair(CLIP, 0xB000, start=START, duration=48, src_in=0),
    )
    facts = facts_lines(seeds / "facts.jsonl", _facts(CAM_A), _facts(CAM_C), mp4_template_facts())
    return {"seed": seed, "facts": facts, "library": make_library(tmp_path / "lib")}


def _export(paths: dict[str, Path], plan: EditPlan, **kwargs: Any) -> Any:
    kwargs.setdefault("project_name", "Show cut")
    return export_library(
        plan,
        parse_profile(PROFILE),
        library=paths["library"],
        seed=paths["seed"],
        facts=load_facts(paths["facts"]),
        templates=[paths["seed"]],
        options=OPTS,
        allow_unopened=True,
        allow_unverified_media=True,
        **kwargs,
    )


def _transitions(path: Path) -> list[tuple[int, int, str, int]]:
    conn = sqlite3.connect(path)
    try:
        rows = conn.execute(
            "SELECT Start, Duration, PrettyType, AlignmentType FROM Sm2TiItem "
            "WHERE DbType = 'Sm2TiTransition'"
        ).fetchall()
    finally:
        conn.close()
    return sorted((int(a) - START, int(b), str(c), int(d)) for a, b, c, d in rows)


def test_a_plan_with_crossfades_is_placed_and_imports_back_unchanged(tmp_path: Path) -> None:
    paths = _setup(tmp_path)
    plan = _plan(THREE, [_fade(0), _fade(1, 4, "Cross Fade +3 dB")])
    out = _export(paths, plan)
    report = out.report
    assert report["exported"]["sound_transitions"] == 2
    assert "add_sound_transitions" in report["operations_not_opened"]
    path = Path(report["project"]["path"])
    # Centred on the cuts at 60 and 132: one frame before the first, two before the second.
    assert _transitions(path) == [
        (59, 2, "Cross Fade 0DB", 2),
        (130, 4, "Cross Fade +3DB", 2),
    ]
    # The +3 dB kind is carried: the FCPXML's note that it becomes the one audio crossfade is
    # not in the library report.
    assert not [i for i in report["not_carried"] if i["feature"] == "audio_transition_type"]
    timeline = read_timeline(path, library_options=OPTS)
    assert timeline.sound_transitions
    result = import_timeline(
        plan, timeline.timeline, timeline.media, sound_transitions=timeline.sound_transitions
    )
    assert result.diff["unchanged"] is True and not result.diff["not_carried"]
    assert result.plan == plan


@pytest.mark.parametrize(
    ("transitions", "change", "words"),
    [
        ([_fade(0, 3)], {}, "odd length"),
        ([_fade(0, 2, "Cross Fade -3 dB")], {}, "'Cross Fade -3 dB' sound transition"),
        ([_fade(0, 20)], {}, "20-frame"),
        (
            [{"after": 0, "video": "dissolve", "video_frames": 6}],
            {},
            "a picture transition",
        ),
        ([], {"head": {"video": "dissolve", "video_frames": 4}}, "picture transition"),
        ([], {"head": {"audio_fade_frames": 6}}, "an audio fade"),
        ([], {"end_fade_frames": 6}, "an audio fade"),
    ],
)
def test_what_a_library_project_cannot_carry_is_refused(
    tmp_path: Path, transitions: list[dict[str, Any]], change: dict[str, Any], words: str
) -> None:
    paths = _setup(tmp_path)
    with pytest.raises(ExportError) as info:
        _export(paths, _plan(THREE, transitions, **change))
    assert info.value.code == "unsupported_plan_feature"
    assert any(words in f["feature"] for f in info.value.details["features"]), info.value.details
    projects = paths["library"] / "Resolve Projects" / "Users" / "guest" / "Projects"
    assert list(projects.iterdir()) == []


def test_a_crossfade_without_media_before_the_cut_is_refused_never_shortened(
    tmp_path: Path,
) -> None:
    """The incoming clip starts at its file's first frame: no centred crossfade has the media,
    so the layout aligns it to start on the cut, which no example in whole frames supports."""
    paths = _setup(tmp_path)
    segments = [dict(s) for s in THREE]
    segments[1] = {"track": 1, "source": "c", "src_in": 0, "src_out": 72}
    with pytest.raises(ExportError) as info:
        _export(paths, _plan(segments, [_fade(0, 4)]))
    assert info.value.code == "unsupported_plan_feature"
    assert "aligned right" in info.value.details["features"][0]["feature"]


def test_a_crossfade_on_a_timeline_of_another_rate_is_refused(tmp_path: Path) -> None:
    paths = _setup(tmp_path)
    plan = _plan(THREE, [_fade(0)], fps="24000/1001")
    with pytest.raises(ExportError) as info:
        _export(paths, plan)
    assert info.value.code == "unsupported_plan_feature"
    assert "29.97" in info.value.details["features"][0]["feature"]


def test_a_crossfade_at_a_gap_is_refused(tmp_path: Path) -> None:
    """Not a join: the FCPXML reports it, and the import of a library project would drop it."""
    paths = _setup(tmp_path)
    segments = [dict(s) for s in THREE]
    segments[1]["record_in"] = 70  # a 10-frame gap after the first segment
    with pytest.raises(ExportError) as info:
        _export(paths, _plan(segments, [_fade(0)]))
    assert info.value.code == "unsupported_plan_feature"
    [feature] = info.value.details["features"]
    assert feature["location"] == "transitions[0]" and "not on a join" in feature["feature"]


def test_without_a_transition_template_the_crossfades_are_refused(tmp_path: Path) -> None:
    paths = _setup(tmp_path, template=False)
    with pytest.raises(ExportError) as info:
        _export(paths, _plan(THREE, [_fade(0)]))
    assert info.value.code == "item_refused"
    assert "no sound transition" in info.value.message


@pytest.mark.parametrize("seed_number", range(10))
def test_random_cuts_with_crossfades_round_trip(tmp_path: Path, seed_number: int) -> None:
    rng = random.Random(1000 + seed_number)
    segments: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    for index in range(rng.randint(2, 6)):
        source = rng.choice(["a", "c"])
        length = rng.randint(20, 120)
        src_in = rng.randint(20, FRAMES - length - 20)
        segments.append(
            {"track": 1, "source": source, "src_in": src_in, "src_out": src_in + length}
        )
        if index and rng.random() < 0.8:
            transitions.append(
                _fade(
                    index - 1,
                    rng.choice([2, 4, 6, 8, 16]),
                    rng.choice(["Cross Fade 0 dB", "Cross Fade +3 dB"]),
                )
            )
    plan = _plan(segments, transitions)
    paths = _setup(tmp_path)
    out = _export(paths, plan)
    assert out.report["exported"]["sound_transitions"] == len(transitions)
    path = Path(out.report["project"]["path"])
    cuts = []
    record = 0
    for seg in segments:
        record += seg["src_out"] - seg["src_in"]
        cuts.append(record)
    want = sorted(
        (
            cuts[t["after"]] - t["audio"]["frames"] // 2,
            t["audio"]["frames"],
            t["audio"]["type"].replace(" dB", "DB"),
            2,
        )
        for t in transitions
    )
    assert _transitions(path) == want
    timeline = read_timeline(path, library_options=OPTS)
    result = import_timeline(
        plan, timeline.timeline, timeline.media, sound_transitions=timeline.sound_transitions
    )
    assert result.diff["unchanged"] is True and not result.diff["not_carried"]
    assert result.plan == plan


def test_sound_transitions_are_compared_only_for_a_library_project(tmp_path: Path) -> None:
    """A library project's sound transitions stand for the plan's audio-only crossfades; a
    .drt of an imported FCPXML holds those as split edits, so there they stay uncompared (the
    limit is unchanged for a .drt)."""
    paths = _setup(tmp_path)
    plan = _plan(THREE, [_fade(0), _fade(1)])
    path = Path(_export(paths, plan).report["project"]["path"])
    conn = sqlite3.connect(path)
    conn.execute(
        "UPDATE Sm2TiItem SET Duration = '4', Start = CAST(CAST(Start AS INTEGER) - 1 AS TEXT) "
        "WHERE DbType = 'Sm2TiTransition' AND CAST(Start AS INTEGER) = ?",
        (START + 59,),
    )
    conn.commit()
    conn.close()
    timeline = read_timeline(path, library_options=OPTS)
    assert timeline.kind == "library"
    compared = import_timeline(plan, timeline.timeline, timeline.media, sound_transitions=True)
    codes = sorted((n["code"], n.get("location", "")) for n in compared.diff["not_carried"])
    assert codes == [("transition_added", ""), ("transition_missing", "transitions[0]")]
    assert any("compared with the plan's audio-only" in x for x in compared.diff["limits"])
    ignored = import_timeline(plan, timeline.timeline, timeline.media)
    assert not ignored.diff["not_carried"] and ignored.diff["unchanged"] is True
    assert (
        "audio_transitions: transition items on audio tracks are not compared"
        in (ignored.diff["limits"])
    )
