"""S25d: the export step's library hand-off (:mod:`aar.recipes.ccn.library_handoff`) and its
params, without the recipe: every fault of the library, the seed, the facts or the paths is a
typed refused record (never an exception), the params refuse path tricks at the boundary, and a
taken name moves to ``-2``. The synthetic seed and plan are the S25c2 export tests' (23.976,
3840x2160, two camera files of a verified kind)."""

from __future__ import annotations

import json
import os
import random
import string
import sys
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "drp"))

from libwrite_support import (
    facts_lines,
    linked_pair,
    make_library,
    mp4_facts,
    mp4_template,
    mp4_template_facts,
    write_seed,
)

from aar.media.probe import timecode_frames
from aar.plan.model import EditPlan
from aar.plan.profile import parse_profile
from aar.recipes.ccn.handoff import handoff_notes, library_lines, refusal_lines
from aar.recipes.ccn.library_handoff import hand_over, project_name
from aar.recipes.ccn.params import LibraryParam

pytest.importorskip("zstandard")

CLIP = mp4_template()
CAM_A = "X:\\Show\\Footage\\camera-a.MP4"
CAM_C = "X:\\Show\\Footage\\camera-c.MP4"
PROFILE = parse_profile(
    {
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
)
SHA = "a" * 64


def _plan(**source_change: Any) -> EditPlan:
    a = {"id": "a", "media": CAM_A, "duration": 3000, **source_change}
    return EditPlan.model_validate(
        {
            "version": 2,
            "project": "Show",
            "timeline": "Cut 1",
            "kind": "longform",
            "fps": "24000/1001",
            "resolution": {"width": 3840, "height": 2160},
            "profile": "export-test",
            "sources": [a, {"id": "c", "media": CAM_C, "duration": 3000}],
            "segments": [
                {"track": 1, "source": "a", "src_in": 0, "src_out": 48},
                {"track": 1, "source": "c", "src_in": 100, "src_out": 160},
            ],
        }
    )


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    media = tmp_path / "media"
    (media / "seeds").mkdir(parents=True)
    monkeypatch.setenv("AAR_MEDIA_ROOT", str(media))
    monkeypatch.setenv("AAR_WORK_DIR", str(tmp_path / "work"))
    return media


def _lib(root: Path, **change: Any) -> LibraryParam:
    seed = root / "seeds" / "Project.db"
    if not seed.exists():
        write_seed(
            seed,
            clips=[CLIP],
            items=linked_pair(CLIP, 0xB000, start=86400, duration=48, src_in=0),
        )
    facts = root / "facts.jsonl"
    if not facts.exists():
        facts_lines(facts, mp4_facts(), mp4_facts(path=CAM_C), mp4_template_facts())
    library = root / "Library"
    if not library.exists():
        make_library(library)
    data: dict[str, Any] = {
        "folder": str(library),
        "seed": str(seed),
        "facts": str(facts),
        "allow_unopened_operations": True,
    }
    data.update(change)
    return LibraryParam.model_validate(data)


def _hand_over(lib: LibraryParam, plan: EditPlan | None = None, round_: int = 1) -> dict[str, Any]:
    record, _report = hand_over(
        lib,
        plan=plan or _plan(),
        plan_sha256=SHA,
        profile=PROFILE,
        media_map={"/srv/none": "E:\\none"},
        project="Show",
        round_=round_,
        transcript=None,
        laughs=None,
    )
    json.dumps(record)  # a step output: JSON only
    return record


def _code(record: dict[str, Any]) -> str:
    assert record["status"] == "refused", record
    assert record["error"]["hint"]
    return record["error"]["code"]


def test_a_placed_record_names_everything_the_pickup_needs(root: Path) -> None:
    record = _hand_over(_lib(root))
    assert record["status"] == "placed"
    path = Path(record["library"]).joinpath(*record["parts"])
    assert record["parts"] == ["Resolve Projects", "Users", "guest", "Projects", "Show - Cut 1 r1"]
    assert (
        path.name == record["folder"] == record["project_name"] == project_name("Show", "Cut 1", 1)
    )
    import hashlib

    assert hashlib.sha256((path / "Project.db").read_bytes()).hexdigest() == record["placed_sha256"]
    assert record["plan_sha256"] == SHA and record["timeline"] == "Cut 1"
    assert record["operations_not_opened"] == ["rename_timeline", "add_media"]
    assert record["user"] == "guest" and record["label"] == "Library"
    # The same round again (a retry, a reopen): never over the first; the next free name.
    again = _hand_over(_lib(root))
    assert again["folder"] == "Show - Cut 1 r1-2" and again["project_id"] != record["project_id"]


def test_a_plan_that_states_its_start_timecode_is_checked_against_the_facts(root: Path) -> None:
    # mp4_facts: 02:10:20:12 at 23.976 (S25c2 compared the frame count with the text and
    # refused every plan that states a start).
    start = timecode_frames("02:10:20:12", 24000 / 1001)  # type: ignore[arg-type]
    assert _hand_over(_lib(root), _plan(start_timecode=start))["status"] == "placed"
    wrong = _hand_over(_lib(root), _plan(start_timecode=start + 1), round_=2)
    assert _code(wrong) == "facts_mismatch" and "02:10:20:12" in wrong["error"]["message"]


@pytest.mark.parametrize(
    ("damage", "code"),
    [
        ("library_missing", "not_a_library"),
        ("not_a_library", "not_a_library"),
        ("facts_missing", "facts_unreadable"),
        ("facts_malformed", "bad_facts"),
        ("facts_of_other_files", "facts_missing"),
        ("facts_failed_lines", "facts_missing"),
        ("facts_fifo", "bad_facts"),
        ("seed_missing", "bad_seed"),
        ("seed_not_sqlite", "bad_seed"),
        ("library_link_out", "path_outside_media_root"),
        ("library_link_to_sibling", "path_outside_media_root"),
        ("facts_link_to_sibling", "path_outside_media_root"),
        ("template_outside", "path_outside_media_root"),
        ("read_only", "library_unwritable"),
    ],
)
def test_every_fault_at_export_time_is_a_typed_refusal(root: Path, damage: str, code: str) -> None:
    lib = _lib(root)
    library, facts, seed = Path(lib.folder), Path(lib.facts), Path(lib.seed)
    if damage == "library_missing":
        library.rename(library.with_name("gone"))
    elif damage == "not_a_library":
        (library / "Resolve Projects" / "Users").rename(library / "elsewhere")
    elif damage == "facts_missing":
        facts.unlink()
    elif damage == "facts_malformed":
        facts.write_text('{"path": "X:\\\\a.MP4", "exists": true, "size": "1"}\n')
    elif damage == "facts_of_other_files":
        facts_lines(facts, mp4_facts(path="X:\\Other\\x.MP4"), mp4_template_facts())
    elif damage == "facts_failed_lines":
        failed = {"path": CAM_A, "exists": True, "error": {"code": "timeout", "message": "slow"}}
        facts.write_text(json.dumps(failed) + "\n")
    elif damage == "facts_fifo":
        if not hasattr(os, "mkfifo"):
            pytest.skip("no FIFOs here")
        facts.unlink()
        os.mkfifo(facts)
    elif damage == "seed_missing":
        seed.unlink()
    elif damage == "seed_not_sqlite":
        seed.write_bytes(b"not a database at all" * 100)
    elif damage == "library_link_out":
        outside = root.parent / "outside"
        library.rename(outside)
        library.symlink_to(outside, target_is_directory=True)
    elif damage in ("library_link_to_sibling", "facts_link_to_sibling", "template_outside"):
        # A folder beside the media root whose name starts with the root's: outside it.
        sibling = root.with_name(root.name + "-old")
        sibling.mkdir()
        if damage == "library_link_to_sibling":
            library.rename(sibling / "Library")
            library.symlink_to(sibling / "Library", target_is_directory=True)
        elif damage == "facts_link_to_sibling":
            facts.rename(sibling / "facts.jsonl")
            facts.symlink_to(sibling / "facts.jsonl")
        else:
            template = root / "template.db"
            seed_copy = sibling / "Project.db"
            seed_copy.write_bytes(seed.read_bytes())
            template.symlink_to(seed_copy)
            lib = lib.model_copy(update={"templates": [str(template)]})
    elif damage == "read_only":
        projects = library / "Resolve Projects"
        for folder in [projects, *projects.rglob("*")]:
            folder.chmod(0o555)
        if os.access(projects, os.W_OK):
            pytest.skip("running with privileges that ignore folder modes")
    try:
        record = _hand_over(lib)
    finally:
        for folder in library.rglob("*"):
            if folder.is_dir() and not folder.is_symlink():
                folder.chmod(0o755)
    assert _code(record) == code, record["error"]
    if damage == "facts_failed_lines":
        assert "could not measure" in record["error"]["message"]
        assert "timeout: slow" in record["error"]["message"]
    projects = library / "Resolve Projects" / "Users" / "guest" / "Projects"
    if projects.is_dir():
        assert list(projects.iterdir()) == []


def test_the_opt_ins_default_off(root: Path) -> None:
    lib = _lib(root, allow_unopened_operations=False)
    assert (lib.allow_unopened_operations, lib.allow_unverified_media) == (False, False)
    assert _code(_hand_over(lib)) == "operation_not_opened"
    assert (
        LibraryParam.model_validate(
            {"folder": lib.folder, "seed": lib.seed, "facts": lib.facts}
        ).allow_unopened_operations
        is False
    )


@pytest.mark.parametrize(
    "change",
    [
        {"folder": "relative/lib"},
        {"folder": "/media/../etc"},
        {"folder": "/media/./lib"},
        {"folder": "//media//lib"},
        {"folder": "/media/lib\x00x"},
        {"folder": "/media/li\nb"},
        {"folder": "/elsewhere/lib"},
        {"folder": "/media2/Library"},
        {"folder": "/media-old"},
        {"seed": "/media2/seeds/Project.db"},
        {"facts": "/mediafacts.jsonl"},
        {"templates": ["/media2/t.db"]},
        {"seed": "/media/seeds/Project.txt"},
        {"seed": "C:\\seed\\Project.db"},
        {"facts": "/media/../facts.jsonl"},
        {"templates": ["/media/t.drp"]},
        {"templates": ["/media/t.db"] * 9},
        {"user": "../guest"},
        {"user": "a/b"},
        {"user": "guest."},
        {"user": ""},
        {"label": "a\nb"},
        {"seed_resolution": "1920x"},
        {"seed_resolution": "0x1080"},
        {"allow_unopened_operations": "true"},
        {"allow_unverified_media": 1},
        {"surprise": True},
        {"facts": "/media/Library"},
    ],
)
def test_params_refuse_path_tricks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: dict[str, Any]
) -> None:
    monkeypatch.setenv("AAR_MEDIA_ROOT", "/media")
    data = {
        "folder": "/media/Library",
        "seed": "/media/seeds/Project.db",
        "facts": "/media/facts.jsonl",
        **change,
    }
    with pytest.raises(ValidationError):
        LibraryParam.model_validate(data)


def test_params_fuzz_only_ever_refuse_with_a_validation_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AAR_MEDIA_ROOT", "/media")
    rng = random.Random(2511)
    alphabet = string.printable + "\x00\u202e\u200b\udcff/\\.."
    fields = ["folder", "seed", "facts", "user", "label", "seed_resolution"]
    accepted = 0
    for _ in range(3000):
        data: dict[str, Any] = {
            "folder": "/media/Library",
            "seed": "/media/s/Project.db",
            "facts": "/media/f.jsonl",
        }
        for name in rng.sample(fields, rng.randint(1, 3)):
            pick = rng.random()
            if pick < 0.6:
                value: Any = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 40)))
                if rng.random() < 0.5 and name in ("folder", "seed", "facts"):
                    value = "/media/" + value
            elif pick < 0.8:
                value = rng.choice([None, 1, True, [], {}, 1.5, b"x"])
            else:
                value = "/media/" + "/".join(
                    rng.choice(["..", ".", "", "a", "b.db"]) for _ in range(4)
                )
            data[name] = value
        try:
            lib = LibraryParam.model_validate(data)
        except ValidationError:
            continue
        accepted += 1
        for path in (lib.folder, lib.seed, lib.facts):
            parts = path.split("/")
            assert path.startswith("/media/") and ".." not in parts and "." not in parts
            assert "\x00" not in path
    assert accepted > 0


def test_the_notes_quote_a_placed_and_a_refused_hand_off() -> None:
    placed = {
        "status": "placed",
        "label": "Shared Library",
        "user": "guest",
        "project_name": "Show - Cut 1 r1",
        "timeline": "Cut 1",
        "operations_not_opened": ["add_media"],
        "unverified_media": [],
        "not_carried": 0,
    }
    refused = {
        "status": "refused",
        "label": "Shared Library",
        "error": {"code": "not_a_library", "message": "no Users folder", "hint": "give the folder"},
    }
    common: dict[str, Any] = dict(
        project="Show", timeline="Cut 1", round_=1, report={}, report_id="r", loudness=None
    )
    text = handoff_notes(**common, library=library_lines(placed))
    assert '"Shared Library" as the project "Show - Cut 1 r1"' in text
    assert "Saving is all that" in text and "Not yet seen working in Resolve" in text
    # the library user, when the record names one
    assert 'choose the project library "Shared Library" (user "guest").' in text
    assert "(user" not in "\n".join(library_lines({**placed, "user": None}))
    # nothing listed that is not there
    for absent in ("without full evidence", "leaves out", "will NOT pick up"):
        assert absent not in text
    text = handoff_notes(**common, library=library_lines(refused))
    assert "LIBRARY PROJECT: NOT WRITTEN" in text and "no Users folder [not_a_library]" in text
    assert "To fix: give the folder" in text
    # without a library the notes are what they were
    assert "LIBRARY" not in handoff_notes(**common)
    assert handoff_notes(**common) == handoff_notes(**common, library=None)
    # a value from the run cannot forge a heading: one line, quoted
    forged = dict(placed, project_name="x\r\nSOUND\r\n")
    forged_text = handoff_notes(**common, library=library_lines(forged))
    assert "\r\nSOUND\r\n\r\n" not in forged_text.split("OR OPEN")[1]


def test_the_notes_name_unverified_media_details_left_out_and_pickup_off() -> None:
    record = {
        "status": "placed",
        "label": "Shared Library",
        "project_name": "Show - Cut 1 r1",
        "timeline": "Cut 1",
        "operations_not_opened": [],
        "unverified_media": ["E:\\Show\\a.mp4", "E:\\Show\\b.mp4"],
        "not_carried": 3,
        "pickup": "off: the placed Project.db could not be hashed (see cleanup)",
    }
    lines = library_lines(record)
    assert (
        "  Clips written without full evidence for their kind (check they play and link): "
        "E:\\Show\\a.mp4, E:\\Show\\b.mp4."
    ) in lines
    assert (
        "  The library project leaves out 3 plan detail(s) (gain, framing, labels, notes, "
        "markers, captions): redo them in Resolve as for the FCPXML."
    ) in lines
    assert (
        "  The hub will NOT pick up saves of it: off: the placed Project.db could not be hashed "
        "(see cleanup)."
    ) in lines
    assert not any("Not yet seen working" in line for line in lines)
    # the owner's note line: the FCPXML stays the dependable route
    assert "  Until library projects have been confirmed in Resolve, the FCPXML stays the " in (
        "\n".join(lines)
    )
    assert refusal_lines("Lib", "c", "m", "h") == library_lines(
        {"status": "refused", "label": "Lib", "error": {"code": "c", "message": "m", "hint": "h"}}
    )
