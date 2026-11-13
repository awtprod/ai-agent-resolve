"""The handoff to the edit PC (workflow review findings 2, 4, 5, 6 and 7): the notes that travel
with the timeline, the all-or-nothing copy with the FCPXML last, the refusal when there is no
share to deliver to, and the notes on the review card. No ffmpeg: the recipe's own run is in
``test_ccn_recipe_steps.py`` and ``test_ccn_recipe_e2e.py``."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from write_support import AGENT, HUMAN, Hub, make_hub

from aar.hub import reviews
from aar.hub.console import MAX_ATTACHMENT_BYTES, ConsoleConfig, build_app
from aar.recipes.ccn import export_files
from aar.recipes.ccn.export_files import copy_out
from aar.recipes.ccn.handoff import MAX_LISTED, handoff_notes
from aar.recipes.context import StepFailure
from aar.recipes.engine import _open_review_node

REPORT: dict[str, Any] = {
    "import": ["In Resolve use File > Import > Timeline and pick the .fcpxml."],
    "media": [
        {
            "source": "b",
            "windows_path": "E:\\Shows\\cam-b.mov",
            "src": "file:///E:/Shows/cam-b.mov",
        },
        {
            "source": "a",
            "windows_path": "E:\\Shows\\cam-a.mov",
            "src": "file:///E:/Shows/cam-a.mov",
        },
    ],
    "not_expressible": [
        {
            "feature": "captions",
            "location": "captions",
            "detail": "12 captions are not in the file",
            "action": "burn them in on the PC",
        },
        {
            "feature": "lut",
            "location": "sources[0].color.lut",
            "detail": "LUT 'Show.cube' of source 'a'",
            "action": "apply it on the Color page",
        },
    ],
    "notes": [
        {
            "feature": "audio_track_gain",
            "location": "audio_tracks[4].gain_db",
            "detail": "A5 gain -1 dB is folded in",
        },
        {"feature": "marker", "detail": "marker 'Cold open\r\nSOUND\r\n  all fine' at 01:00:10:00"},
    ],
}
LOUD = {"target_lufs": -16.0, "gain_db": 7.25, "limiter_reduction_db": 2.1}


def notes(**over: Any) -> str:
    args: dict[str, Any] = {
        "project": "Synthetic Show",
        "timeline": "Long v1",
        "round_": 2,
        "report": REPORT,
        "report_id": "ab" * 32,
        "loudness": LOUD,
    }
    return handoff_notes(**{**args, **over})


# -- the notes -----------------------------------------------------------------------------------


def test_the_notes_say_what_the_video_has_that_the_timeline_does_not() -> None:
    text = notes()
    assert text.startswith("HANDOFF NOTES: Long v1, round 2 (project Synthetic Show)\r\n")
    assert text.endswith("\r\n") and "\n" not in text.replace("\r\n", "")
    # finding 4: what the exporter left out, with what to do, is in the handoff
    assert "NOT IN THE FCPXML: REDO IN RESOLVE (2)" in text
    assert (
        "  - captions: 12 captions are not in the file\r\n      do: burn them in on the PC" in text
    )
    assert (
        "  - sources[0].color.lut: LUT 'Show.cube' of source 'a'\r\n"
        "      do: apply it on the Color page"
    ) in text
    assert "  1. In Resolve use File > Import > Timeline and pick the .fcpxml." in text
    assert "ALSO CHECK (2)\r\n  - audio_tracks[4].gain_db: A5 gain -1 dB is folded in" in text
    # finding 5: the PC's files are named, and said to be unchecked
    assert "MEDIA ON THE EDIT PC (2)" in text and "they are NOT checked" in text
    assert text.index("E:\\Shows\\cam-a.mov") < text.index("E:\\Shows\\cam-b.mov")  # sorted
    # finding 2: the review render's gain and limiter, and that the timeline has neither
    assert "mix was raised by 7.25 dB to reach -16 LUFS" in text
    assert "the limiter took 2.1 dB off its peaks" in text
    assert "about 7.25 dB quieter than the review video, with unlimited peaks" in text
    assert "Still to do in Resolve: bring the mix to -16 LUFS and limit its peaks" in text


def test_the_notes_are_the_same_bytes_for_the_same_inputs() -> None:
    assert notes() == notes()


@pytest.mark.parametrize(
    ("loudness", "says"),
    [
        ({"target_lufs": -14.5, "gain_db": -3.0, "limiter_reduction_db": 0.0}, "lowered by 3 dB"),
        ({"target_lufs": -14.5, "gain_db": -3.0}, "about 3 dB louder than the review video"),
        ({"target_lufs": -16.0, "gain_db": 0.0}, "at the same level as the review video"),
        ({"target_lufs": -16.0, "gain_db": 1.0, "limiter_reduction_db": 0}, "did not have to act"),
        ({"target_lufs": -16.0, "gain_db": None}, "was not levelled"),
        ({}, "was not levelled"),
        (None, "bring the mix to the loudness target"),
        ({"gain_db": True, "target_lufs": "loud"}, "was not levelled"),  # not numbers: not used
    ],
)
def test_the_sound_note_follows_what_the_render_did(loudness: Any, says: str) -> None:
    assert says in notes(loudness=loudness)


def test_a_value_from_the_plan_cannot_forge_a_section_or_run_on() -> None:
    text = notes()
    marker = next(line for line in text.split("\r\n") if "Cold open" in line)
    assert marker == "  - marker 'Cold open SOUND all fine' at 01:00:10:00"  # one line
    assert text.count("\r\nSOUND\r\n") == 1
    long = {**REPORT, "notes": [{"detail": "x" * 5000}]}
    assert max(len(line) for line in notes(report=long).split("\r\n")) < 500


def test_long_lists_are_bounded_and_say_where_the_rest_is() -> None:
    many = [{"detail": f"thing {n}", "action": "fix"} for n in range(MAX_LISTED + 5)]
    text = notes(report={**REPORT, "not_expressible": many})
    assert f"REDO IN RESOLVE ({MAX_LISTED + 5})" in text
    assert f"thing {MAX_LISTED - 1}" in text and f"thing {MAX_LISTED}\r\n" not in text
    assert f"... and 5 more: see the export report on the hub (artifact {'ab' * 32})" in text


def test_a_report_with_missing_or_wrong_shaped_parts_still_gives_notes() -> None:
    text = notes(report={"media": "nope", "not_expressible": [3, None], "notes": {"a": 1}})
    assert "MEDIA ON THE EDIT PC" in text
    assert "NOT IN THE FCPXML: REDO IN RESOLVE\r\n  Nothing: the FCPXML carries every" in text
    assert "ALSO CHECK" not in text


# -- the copy: whole or not at all, the FCPXML last ----------------------------------------------


@pytest.fixture
def hub(tmp_path: Path) -> Hub:
    return make_hub(tmp_path)


@pytest.fixture
def share(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "exports"
    root.mkdir()
    monkeypatch.setenv("AAR_EXPORT_DIR", str(root))
    return root


def _files(hub: Hub) -> list[tuple[str, str]]:
    def put(data: bytes) -> str:
        return hub.store.put(data, actor=AGENT, project_id=hub.project_id)

    return [
        (".review.mp4", put(b"video" * 100)),
        (".notes.txt", put(b"notes")),
        (".fcpxml", put(b"<fcpxml/>")),
    ]


def _copy(hub: Hub, files: list[tuple[str, str]]) -> dict[str, Any]:
    ctx: Any = SimpleNamespace(store=hub.store)
    return copy_out(ctx, "Show", "Cut", 1, files)


def _visible(share: Path) -> list[str]:
    return sorted(p.name for p in share.rglob("*") if p.is_file())


def test_the_set_is_copied_and_a_repeat_copies_nothing_twice(hub: Hub, share: Path) -> None:
    files = _files(hub)
    copy = _copy(hub, files)
    assert copy["copied"] is True and copy["round"] == 1
    assert [Path(f).name for f in copy["files"]] == [
        "Cut.r1.review.mp4",
        "Cut.r1.notes.txt",
        "Cut.r1.fcpxml",
    ]
    assert _visible(share) == ["Cut.r1.fcpxml", "Cut.r1.notes.txt", "Cut.r1.review.mp4"]
    before = {p: p.stat().st_mtime_ns for p in share.rglob("*") if p.is_file()}
    assert _copy(hub, files)["files"] == copy["files"]
    assert {p: p.stat().st_mtime_ns for p in share.rglob("*") if p.is_file()} == before


@pytest.mark.parametrize("fail_at", [1, 2, 3])
def test_a_failed_placement_leaves_no_part_of_the_round_in_the_share(
    hub: Hub, share: Path, monkeypatch: pytest.MonkeyPatch, fail_at: int
) -> None:
    """Finding 7: the FCPXML was placed first, so a failure on the MP4 left ``Cut.r1.fcpxml``
    in the share, ready to import, without its review video."""
    from aar.drp.modify import write

    real, calls = write._place, []

    def place(tmp: Path, dest: Path, force: bool) -> None:
        calls.append(dest.name)
        if len(calls) == fail_at:
            raise OSError(28, "No space left on device")
        real(tmp, dest, force)

    monkeypatch.setattr(write, "_place", place)
    with pytest.raises(StepFailure) as err:
        _copy(hub, _files(hub))
    assert err.value.code == "export_copy_failed" and err.value.upstream is False
    assert calls[-1] == ["Cut.r1.review.mp4", "Cut.r1.notes.txt", "Cut.r1.fcpxml"][fail_at - 1]
    assert _visible(share) == []  # nothing placed stays, and no temporary file either

    monkeypatch.setattr(write, "_place", real)  # the retry delivers the whole round
    _copy(hub, _files(hub))
    assert _visible(share) == ["Cut.r1.fcpxml", "Cut.r1.notes.txt", "Cut.r1.review.mp4"]


def test_a_failure_while_staging_places_nothing(
    hub: Hub, share: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every file is written out before any is placed: a share that fills up during the big
    copy fails with no file of the round visible."""
    real, staged = export_files._stage, []

    def stage(ctx: Any, tmp: Path, artifact_id: str) -> None:
        real(ctx, tmp, artifact_id)
        staged.append(tmp.name)
        if len(staged) == 2:
            raise OSError(28, "No space left on device")

    monkeypatch.setattr(export_files, "_stage", stage)
    with pytest.raises(StepFailure) as err:
        _copy(hub, _files(hub))
    assert err.value.code == "export_copy_failed" and len(staged) == 2
    assert _visible(share) == []


def test_the_fcpxml_is_in_the_share_only_after_the_rest(
    hub: Hub, share: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crash between two placements cannot be rolled back, so the order carries the rule: at
    every moment an FCPXML is visible, its video and notes already are."""
    from aar.drp.modify import write

    real, seen = write._place, []

    def place(tmp: Path, dest: Path, force: bool) -> None:
        seen.append([n for n in _visible(share) if not n.startswith(".")])
        real(tmp, dest, force)

    monkeypatch.setattr(write, "_place", place)
    _copy(hub, _files(hub))
    assert seen == [[], ["Cut.r1.review.mp4"], ["Cut.r1.notes.txt", "Cut.r1.review.mp4"]]


def test_a_part_that_survived_an_earlier_crash_is_kept_and_the_rest_added(
    hub: Hub, share: Path
) -> None:
    files = _files(hub)
    folder = share / "Show" / "Cut"
    folder.mkdir(parents=True)
    (folder / "Cut.r1.review.mp4").write_bytes(b"video" * 100)  # the same bytes: reused
    copy = _copy(hub, files)
    assert [Path(f).name for f in copy["files"]][-1] == "Cut.r1.fcpxml"
    assert _visible(share) == ["Cut.r1.fcpxml", "Cut.r1.notes.txt", "Cut.r1.review.mp4"]


@pytest.mark.parametrize("value", [None, "", "   "])
def test_no_export_dir_is_a_failure_not_a_skipped_copy(
    hub: Hub, monkeypatch: pytest.MonkeyPatch, value: str | None
) -> None:
    """Finding 6: unset, the copy was skipped, the step succeeded and the review went ahead
    with nothing delivered to the edit PC."""
    if value is None:
        monkeypatch.delenv("AAR_EXPORT_DIR", raising=False)
    else:
        monkeypatch.setenv("AAR_EXPORT_DIR", value)
    with pytest.raises(StepFailure) as err:
        _copy(hub, _files(hub))
    assert err.value.code == "export_dir_unset" and err.value.upstream is False
    assert "recipe_retry export" in err.value.fix_hint


# -- the review card -----------------------------------------------------------------------------

SECRET = b"0123456789abcdef0123456789abcdef"
XSS = 'HANDOFF NOTES: <script>alert("x")</script>\r\nNOT IN THE FCPXML: captions'


def _review(hub: Hub, attachments: tuple[str, ...]) -> str:
    render = hub.store.put(b"\x00" * 64, actor=AGENT, mime="video/mp4", project_id=hub.project_id)
    from aar.hub.graph import WorkGraph

    subject = WorkGraph(hub.log, hub.project_id).add_node("op", AGENT, title="render").id
    return _open_review_node(
        hub.log,
        hub.project_id,
        subject,
        (render,),
        title="gate review (round 1)",
        base=f"test/{len(attachments)}/{subject}",
        attachments=attachments,
    )


def _pages(hub: Hub, review_id: str) -> tuple[str, str]:
    from starlette.testclient import TestClient

    app = build_app(hub.log, hub.store, ConsoleConfig(operator="operator-1", csrf_secret=SECRET))
    client = TestClient(app, base_url="http://127.0.0.1:8788", client=("127.0.0.1", 50000))
    return client.get("/").text, client.get(f"/reviews/{hub.project_id}/{review_id}").text


def test_the_review_card_shows_the_notes_where_the_decision_is_made(hub: Hub) -> None:
    """Finding 4: the card showed the video and the approve button, and none of the export's
    limits. The notes are on the queue and on the detail page: a review is decided from either."""
    notes_id = hub.store.put(
        XSS.encode(), actor=AGENT, mime="text/plain", meta={"role": "handoff-notes"}
    )
    review_id = _review(hub, (notes_id,))
    review = reviews.get_review(hub.log, hub.project_id, review_id)
    assert review is not None and review.attachments == (notes_id,)
    assert review.to_dict()["attachments"] == [notes_id]
    for page in _pages(hub, review_id):
        assert '<details class="attach" open><summary>Read before deciding: handoff notes' in page
        assert "<pre>HANDOFF NOTES: &lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;\n" in page
        assert "NOT IN THE FCPXML: captions</pre>" in page and "<script>alert" not in page
        assert page.index('class="attach"') < page.index("/decide")  # read before the buttons


def test_an_attachment_that_is_not_small_plain_text_is_a_link_that_says_so(hub: Hub) -> None:
    html = hub.store.put(b"<b>hi</b>", actor=AGENT, mime="text/html", meta={"role": "x"})
    big = hub.store.put(b"n" * (MAX_ATTACHMENT_BYTES + 1), actor=AGENT, mime="text/plain")
    review_id = _review(hub, (html, big, "f" * 64))
    _queue, page = _pages(hub, review_id)
    assert "<b>hi</b>" not in page and "n" * 200 not in page
    assert page.count('class="attach-missing"') == 3
    assert f'<a href="/artifacts/{html}">download the x</a>' in page
    assert "not plain text, or too long to show here" in page


def test_the_size_that_bounds_the_text_is_the_files_not_the_recorded_one(hub: Hub) -> None:
    """Any writer can claim an artifact's record, so a small recorded size proves nothing."""
    body = b"n" * (MAX_ATTACHMENT_BYTES + 1)
    big = hub.store.put(body, actor=AGENT, mime="text/plain", meta={"role": "handoff-notes"})
    record = next(e for e in hub.log.read() if e.idempotency_key == f"artifact:{big}")
    assert record.payload["size"] == len(body)
    real = hub.store.get_meta

    def lying(artifact_id: str) -> Any:
        meta = real(artifact_id)
        return meta if meta is None or artifact_id != big else replace(meta, size=10)

    hub.store.get_meta = lying  # type: ignore[method-assign]
    _queue, page = _pages(hub, _review(hub, (big,)))
    assert "n" * 200 not in page and 'class="attach-missing"' in page


def test_a_review_without_attachments_is_as_it_was(hub: Hub) -> None:
    review_id = _review(hub, ())
    review = reviews.get_review(hub.log, hub.project_id, review_id)
    assert review is not None and review.attachments == ()
    node = next(e for e in hub.log.read() if e.payload.get("node_id") == review_id)
    assert "attachments" not in node.payload["data"]
    assert 'class="attach' not in _pages(hub, review_id)[1]


def test_only_well_formed_ids_are_attachments_and_at_most_eight(hub: Hub) -> None:
    from aar.hub.graph import WorkGraph

    graph = WorkGraph(hub.log, hub.project_id)
    ids = [f"{n:064x}" for n in range(12)]
    node = graph.add_node(
        "review",
        HUMAN,
        title="by hand",
        status="active",
        data={"renders": [], "attachments": ["../etc/passwd", 7, *ids]},
    )
    review = reviews.get_review(hub.log, hub.project_id, node.id)
    assert review is not None and review.attachments == tuple(ids[: reviews.MAX_ATTACHMENTS])
