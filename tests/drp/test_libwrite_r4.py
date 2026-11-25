"""S25c2, review round 4: a sound reads no further than its clip's sound, and the gates never
compare an item or a clip with a second read of itself.

1. A sound placement is refused past the clip's own audio length (its first audio track's
   samples over its sample rate, in frames at the item's rate, rounded up), in the author, in
   the cut and in the verifier of the written project.
2. The item gate and the media gate read each project once (by its resolved path), and leave
   out the target by (project, item or clip id), so a project given twice, or through an alias
   path, is still leave-one-out.
"""

from __future__ import annotations

import dataclasses
import os
import sys
from fractions import Fraction
from math import ceil
from pathlib import Path
from typing import Any

import mediapool_support as ms
import pytest
from libwrite_support import linked_pair, mp4_template, write_seed
from mediapool_support import replace_row, track, uid, wav_template
from test_libwrite import CLIP, OPTS, _make, _seed
from test_mediapool_templates import FAST, answer, project_with

import aar.drp.items.author as author_module
import aar.drp.libwrite as libwrite
from aar.drp.items import ItemError, ItemPlacement, author_item, gate
from aar.drp.items.media import media_view
from aar.drp.items.rows import read_items
from aar.drp.libwrite import Authored, ProjectWriteError
from aar.drp.libwrite.cut import Cut, author_cut, pick_item_templates
from aar.drp.mediapool import facts_from_json
from aar.drp.mediapool import gate as media_gate

pytest.importorskip("zstandard")

RATE = Fraction(24000, 1001)


def _short(seconds_samples: int = 48000) -> Any:
    """A camera clip of 240 picture frames whose sound is ``seconds_samples`` at 48 kHz."""
    clip = mp4_template(base=0x400, name="short-sound.MP4")
    return replace_row(
        clip,
        "audio0",
        TracksBA=track(
            uid(0x409),
            start=86400 / float(RATE),
            rate=48000,
            channels=2,
            samples=seconds_samples,
            codec="Linear PCM",
            depth=1,
        ),
    )


def _sound_template(tmp_path: Path) -> tuple[Any, Any]:
    items = read_items(_seed(tmp_path), options=OPTS)
    return next(r for r in items.items if r.kind == "sound"), items.clips


def _sound(src_in: int, duration: int) -> ItemPlacement:
    return ItemPlacement("sound", uid(0xA006), 0, 86400, duration, src_in, (uid(0xD101),))


# -- 1. a sound reads no further than its clip's sound ---------------------------------------


def test_the_sound_length_is_the_samples_in_frames_rounded_up() -> None:
    view = media_view(_short())
    assert view.frames == 240
    # one second at 24000/1001 is 23.976 frames: the last, partial frame is the 24th
    assert view.sound_frames == ceil(Fraction(48000, 48000) * RATE) == 24
    assert media_view(CLIP).sound_frames == 240  # 480480 samples: exactly 240 frames
    # a sound file: at the rate of the project it was imported into (30000/1001)
    assert media_view(wav_template()).sound_frames == ceil(
        Fraction(240000, 48000) * Fraction(30000, 1001)
    )


@pytest.mark.parametrize(
    ("src_in", "duration", "past"),
    [(0, 24, False), (23, 1, False), (0, 25, True), (24, 1, True), (5000, 10, True)],
)
def test_a_sound_placement_past_the_clips_sound_is_refused(
    tmp_path: Path, src_in: int, duration: int, past: bool
) -> None:
    template, clips = _sound_template(tmp_path)
    clip = _short()
    if past:
        with pytest.raises(ItemError, match="past the media's 24 frames of sound") as info:
            author_item(clip, template, clips, _sound(src_in, duration), item_id=uid(0xD100))
        assert info.value.code == "bad_placement"
    else:
        author_item(clip, template, clips, _sound(src_in, duration), item_id=uid(0xD100))


def test_a_sound_of_a_sound_file_is_bounded_by_its_samples(tmp_path: Path) -> None:
    template, clips = _sound_template(tmp_path)
    wav = wav_template()
    last = media_view(wav).sound_frames
    assert last is not None
    author_item(wav, template, clips, _sound(last - 1, 1), item_id=uid(0xD100))
    with pytest.raises(ItemError, match="of sound") as info:
        author_item(wav, template, clips, _sound(last, 1), item_id=uid(0xD100))
    assert info.value.code == "bad_placement"


def _cut_seed(tmp_path: Path) -> Path:
    short = _short()
    return write_seed(
        tmp_path / "seed.db",
        clips=[CLIP, short],
        items=linked_pair(CLIP, 0xB000, start=86400, duration=48, src_in=0),
    )


def test_a_cut_whose_sound_runs_past_the_clips_sound_is_refused(tmp_path: Path) -> None:
    seed_path = _cut_seed(tmp_path)
    picks = pick_item_templates([read_items(seed_path, options=OPTS)])
    seed = libwrite.read_seed(seed_path, options=OPTS)
    short = next(c for c in seed.clips if c.media_id == _short().media_id)
    with pytest.raises(ItemError, match="of sound") as info:
        author_cut(seed, [Cut(short, 0, 48, 0)], picks)
    assert info.value.code == "bad_placement"
    # the picture may run on: a split edit whose sound stops at the sound's end is accepted
    items, _ = author_cut(seed, [Cut(short, 0, 48, 0, sound_duration=24)], picks)
    assert [i.kind for i in items] == ["picture", "sound"]


def test_the_verifier_refuses_a_sound_past_its_clips_sound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed_path = _cut_seed(tmp_path)
    picks = pick_item_templates([read_items(seed_path, options=OPTS)])
    short_id = _short().media_id

    def author(s: Any) -> Authored:
        clip = next(c for c in s.clips if c.media_id == short_id)
        cuts = [Cut(clip, 0, 48, 0)]
        items, specs = author_cut(s, cuts, picks)
        return Authored([], items, specs, cuts=cuts)

    # The author's own bound switched off, and the reader's read-back too: the verifier alone.
    real = author_module.media_view
    monkeypatch.setattr(
        author_module, "media_view", lambda c: dataclasses.replace(real(c), sound_frames=10**9)
    )
    monkeypatch.setattr(libwrite, "check_written", lambda *a, **k: None)
    with pytest.raises(ProjectWriteError, match="past the end of its clip's sound") as info:
        _make(tmp_path, seed_path, author=author)
    assert info.value.code == "self_check_failed"
    assert info.value.details["field"] == "In"


# -- 2. the gates never use a second read of the target as its template ------------------------


def _one_pair(tmp_path: Path) -> Path:
    folder = tmp_path / "only"
    folder.mkdir()
    return write_seed(
        folder / "Project.db",
        clips=[CLIP],
        items=linked_pair(CLIP, 0xB000, start=86400, duration=48, src_in=0),
    )


def _alias(tmp_path: Path, project: Path) -> Path:
    link = tmp_path / "alias"
    try:
        os.symlink(project.parent, link, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("this system cannot make symbolic links")
    return link / project.name


def test_an_item_gate_project_with_one_item_of_each_kind_has_no_template(tmp_path: Path) -> None:
    project = _one_pair(tmp_path)
    board = gate.run([project], options=OPTS)
    assert board.items_checked == 0
    assert sum(g.refused for g in board.groups) == 2


@pytest.mark.parametrize("how", ["twice", "dot-dot", "symlink"])
def test_the_item_gate_reads_a_project_given_twice_once(tmp_path: Path, how: str) -> None:
    project = _one_pair(tmp_path)
    if how == "twice":
        again = project
    elif how == "dot-dot":
        again = project.parent / ".." / project.parent.name / project.name
    else:
        if sys.platform == "win32":
            pytest.skip("symbolic links need privileges on Windows")
        again = _alias(tmp_path, project)
    board = gate.run([project, again], options=OPTS)
    # still no other item of either kind: the second read of the same item is not a template
    assert board.items_checked == 0
    assert sum(g.refused for g in board.groups) == 2
    assert board.extents_checked == 1


def test_the_item_gate_excludes_the_target_by_project_and_id_not_by_object(
    tmp_path: Path,
) -> None:
    project = _one_pair(tmp_path)
    items = read_items(project, options=OPTS)
    again = read_items(project, options=OPTS)
    key = str(project.resolve())
    rows = next(r for r in items.items if r.kind == "picture")
    copy = next(r for r in again.items if r.kind == "picture")
    shape = gate.shape_of(rows, items.clips)
    target = gate._Target(key, items, rows, shape, True)
    twin = gate._Target(key, again, copy, shape, True)
    assert twin is not target
    assert gate._template(target, [target, twin], other_project=False) is None


def _wav_pair() -> tuple[Any, Any, dict[str, Any]]:
    first = ms.wav_facts()
    second = ms.wav_facts(path="X:\\Show\\Sound\\take-two.wav", stream_duration_ts=96000)
    clip_a = answer(ms.wav_template(), first, 0x5000)
    clip_b = answer(ms.wav_template(), second, 0x6000)
    facts = {f["path"].lower(): facts_from_json(f) for f in (first, second)}
    return clip_a, clip_b, facts


@pytest.mark.parametrize("how", ["twice", "dot-dot", "symlink"])
def test_the_media_gate_reads_a_project_given_twice_once(tmp_path: Path, how: str) -> None:
    clip_a, _, facts = _wav_pair()
    db = project_with(tmp_path, "p", clip_a)
    if how == "twice":
        again = db
    elif how == "dot-dot":
        again = db.parent / ".." / db.parent.name / db.name
    else:
        if sys.platform == "win32":
            pytest.skip("symbolic links need privileges on Windows")
        again = _alias(tmp_path, db)
    board = media_gate.run([db, again], facts, options=FAST)
    [score] = board.kinds
    # one member: self-checked only, never checked against a second read of itself
    assert (score.members, score.checked, score.status) == (1, 0, "single")


def test_the_media_gate_tells_projects_apart_by_full_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clip_a, clip_b, facts = _wav_pair()
    first = project_with(tmp_path, "first", clip_a, clip_b)
    second = project_with(tmp_path, "second", clip_a, clip_b)
    assert first.name == second.name == "Project.db"
    pairs: list[tuple[str, str]] = []
    real = media_gate._check

    def spy(score: Any, kind: Any, target: Any, template: Any) -> None:
        pairs.append((target.project, template.project))
        real(score, kind, target, template)

    monkeypatch.setattr(media_gate, "_check", spy)
    board = media_gate.run([first, second], facts, options=FAST)
    assert board.kinds[0].checked == 4 and len(pairs) == 4
    # the template comes from the target's own project (the one in the same folder)
    assert all(t == s for t, s in pairs), pairs
    assert {t for t, _ in pairs} == {str(first.resolve()), str(second.resolve())}


# -- 3. a cut entry's own numbers ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("change", "fragment"),
    [
        ({"record_in": -1}, "record_in -1"),
        ({"record_in": True}, "record_in True"),
        ({"record_in": 1.0}, "record_in 1.0"),
        ({"sound_record_in": -24}, "sound_record_in -24"),
        ({"sound": False, "sound_src_in": 0}, "has no sound"),
        ({"picture": 1}, "True or False"),
    ],
)
def test_a_cut_entry_with_a_bad_record_frame_is_refused(
    tmp_path: Path, change: dict[str, Any], fragment: str
) -> None:
    seed_path = _seed(tmp_path)
    picks = pick_item_templates([read_items(seed_path, options=OPTS)])
    seed = libwrite.read_seed(seed_path, options=OPTS)
    cut = dataclasses.replace(Cut(seed.clips[0], 0, 24, 0), **change)
    with pytest.raises(ProjectWriteError, match=fragment) as info:
        author_cut(seed, [cut], picks)
    assert info.value.code == "bad_input"


def test_a_cut_entry_that_is_not_a_cut_is_refused(tmp_path: Path) -> None:
    seed_path = _seed(tmp_path)
    picks = pick_item_templates([read_items(seed_path, options=OPTS)])
    seed = libwrite.read_seed(seed_path, options=OPTS)
    with pytest.raises(ProjectWriteError, match="not a Cut") as info:
        author_cut(seed, [("clip", 0, 24, 0)], picks)  # type: ignore[list-item]
    assert info.value.code == "bad_input"


# -- 4. a file imported twice is not its own template ------------------------------------------


def test_the_item_gate_takes_a_template_of_another_file_over_the_same_file_twice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    again = mp4_template(base=0x400)  # the same file as CLIP, imported a second time
    other = mp4_template(base=0x500, name="camera-d.MP4")
    assert media_view(again).path == media_view(CLIP).path != media_view(other).path
    items = []
    for index, clip in enumerate((CLIP, again, other)):
        for row, join in linked_pair(
            clip, 0xB000 + 16 * index, start=86400 + 48 * index, duration=48, src_in=0, index=index
        ):
            items.append(({**row, "MediaFilePath": media_view(clip).path}, join))
    project = write_seed(tmp_path / "Project.db", clips=[CLIP, again, other], items=items)
    pairs: list[tuple[object, object]] = []
    real = gate._check

    def spy(group: Any, target: Any, template: Any) -> None:
        pairs.append(
            (target.rows.item.get("MediaFilePath"), template.rows.item.get("MediaFilePath"))
        )
        real(group, target, template)

    monkeypatch.setattr(gate, "_check", spy)
    board = gate.run([project], options=OPTS)
    assert board.items_checked == 6 and len(pairs) == 6
    assert all(t != s for t, s in pairs), pairs


def test_the_media_gate_takes_a_template_of_another_file_over_the_same_file_twice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clip_a, clip_b, facts = _wav_pair()
    first = ms.wav_facts()
    clip_again = answer(ms.wav_template(), first, 0x7000)  # clip_a's file, imported again
    db = project_with(tmp_path, "p", clip_a, clip_again, clip_b)
    pairs: list[tuple[str, str]] = []
    real = media_gate._check

    def spy(score: Any, kind: Any, target: Any, template: Any) -> None:
        pairs.append((target.facts.path, template.facts.path))
        real(score, kind, target, template)

    monkeypatch.setattr(media_gate, "_check", spy)
    board = media_gate.run([db], facts, options=FAST)
    assert board.kinds[0].members == 3 and len(pairs) == 3
    assert all(t != s for t, s in pairs), pairs


@pytest.mark.parametrize("start", [float("nan"), float("inf"), float("-inf")])
def test_a_sound_file_whose_start_is_not_a_finite_number_is_refused(start: float) -> None:
    wav = replace_row(
        wav_template(),
        "audio0",
        TracksBA=track(
            uid(0x104),
            start=start,
            rate=48000,
            channels=2,
            samples=240000,
            codec="Linear PCM",
            depth=2,
        ),
    )
    with pytest.raises(ItemError, match="StartTime") as info:
        media_view(wav)
    assert info.value.code == "bad_media"
