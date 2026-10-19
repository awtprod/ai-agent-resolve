"""S11f: the review mirror's ``applied`` when a Resolve call raises (LOW 4), and a re-run
finishing a move an earlier run left half done (LOW 5)."""

from __future__ import annotations

import random
from typing import Any

import pytest

from aar.node.broker import OpStatus
from aar.node.gateway import GatewayError
from aar.node.ops import submit_op

from .ops_support import AGENT, Show

START = 108000
DURATION = 60  # frames 0..59 on the branch, so plans collide often


def _branch_show() -> tuple[Show, Any]:
    show = Show()
    branch = show.make_branch()
    timeline = show.project.timeline(branch.timeline.name)
    timeline.place("video", 1, show.cam_a, START, DURATION)
    return show, timeline


def _mirror(show: Show, timeline: Any, notes: list[dict], **kw: Any):
    params = {
        "timeline": {"name": timeline.GetName(), "unique_id": timeline.GetUniqueId()},
        "review_id": "rev1",
        "notes": notes,
        **kw,
    }
    op_id = submit_op(
        show.broker,
        "markers.mirror_review",
        params,
        target=show.locator("Show"),
        holder=AGENT,
        require_branch=True,
        branch_registry=show.branch_registry,
    )
    show.broker.run_until_idle()
    return show.broker.result(op_id)


def _note(note_id: str, frame: int, text: str = "t") -> dict[str, Any]:
    return {"note_id": note_id, "frame": frame, "text": text}


def _cd(note_id: str) -> str:
    return f"aar:note:rev1:{note_id}"


def _marks(timeline: Any) -> dict[int, tuple[str, str]]:
    return {f: (m["customData"], m["note"]) for f, m in sorted(timeline.markers.items())}


def _ours(timeline: Any) -> set[tuple[int, str]]:
    return {
        (f, m["customData"].rsplit(":", 1)[1])
        for f, m in timeline.markers.items()
        if m["customData"].startswith("aar:note:rev1:")
    }


def _fault(timeline: Any, method: str, n: int, how: str) -> None:
    """On the ``n``-th call of ``timeline.method``: raise before acting (``raise``), act and
    then raise (``raise-after``, a lost reply), raise a GatewayError, or return False."""
    real = getattr(timeline, method)
    count = [0]

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        count[0] += 1
        if count[0] != n:
            return real(*args, **kwargs)
        if how == "raise":
            raise RuntimeError("resolve hiccup")
        if how == "gateway":
            raise GatewayError("resolve_unavailable", "Resolve stopped answering")
        if how == "raise-after":
            real(*args, **kwargs)
            raise TimeoutError("the call landed, its reply did not")
        return False

    setattr(timeline, method, wrapped)


def _calc(before: set, applied: dict) -> set:
    """What ``applied`` says the review's markers are now: before - deleted | added."""
    deleted = {(d["frame"], d["note_id"]) for d in applied["deleted"]}
    added = {(d["frame"], d["note_id"]) for d in applied["added"]}
    return (before - deleted) | added


# -- LOW 4: a raise still reports applied, read back from Resolve -----------------------------


def test_a_raise_on_the_second_add_reports_the_first_add() -> None:
    show, tl = _branch_show()
    _fault(tl, "AddMarker", 2, "raise")
    result = _mirror(show, tl, [_note("n1", 10), _note("n2", 20)])
    assert result.status is OpStatus.FAILED and result.error.code == "resolve_call_failed"
    assert result.error.detail["applied"] == {
        "added": [{"note_id": "n1", "frame": 10}],
        "deleted": [],
    }
    assert _marks(tl) == {10: (_cd("n1"), "t")}


def test_an_add_that_lands_but_raises_is_reported_as_added() -> None:
    show, tl = _branch_show()
    _fault(tl, "AddMarker", 1, "raise-after")
    result = _mirror(show, tl, [_note("n1", 10), _note("n2", 20)])
    assert result.status is OpStatus.FAILED and result.error.code == "resolve_call_failed"
    assert result.error.detail["applied"] == {
        "added": [{"note_id": "n1", "frame": 10}],
        "deleted": [],
    }
    assert set(tl.markers) == {10}


def test_a_delete_that_lands_but_raises_is_reported_as_deleted() -> None:
    show, tl = _branch_show()
    assert _mirror(show, tl, [_note("n1", 10), _note("n2", 20)]).status is OpStatus.OK
    _fault(tl, "DeleteMarkerAtFrame", 1, "raise-after")
    result = _mirror(show, tl, [_note("n2", 20)])  # prunes n1
    assert result.status is OpStatus.FAILED
    assert result.error.detail["applied"] == {
        "added": [],
        "deleted": [{"note_id": "n1", "frame": 10}],
    }
    assert set(tl.markers) == {20}


def test_a_gateway_error_keeps_its_code_and_reports_applied() -> None:
    show, tl = _branch_show()
    _fault(tl, "AddMarker", 2, "gateway")
    result = _mirror(show, tl, [_note("n1", 10), _note("n2", 20)])
    assert result.status is OpStatus.FAILED and result.error.code == "resolve_unavailable"
    assert result.error.detail["applied"]["added"] == [{"note_id": "n1", "frame": 10}]


def test_a_write_whose_read_back_also_fails_is_listed_as_unconfirmed() -> None:
    show, tl = _branch_show()
    _fault(tl, "AddMarker", 1, "raise-after")

    def unreadable(frame: int) -> str:
        raise RuntimeError("Resolve went away")

    tl.GetMarkerCustomData = unreadable
    result = _mirror(show, tl, [_note("n1", 10)])
    assert result.status is OpStatus.FAILED
    assert result.error.detail["applied"] == {
        "added": [],
        "deleted": [],
        "unconfirmed": [{"note_id": "n1", "frame": 10, "action": "add"}],
    }


# -- LOW 5: a re-run finishes a half-done move; it never guesses ------------------------------


def test_a_rerun_finishes_a_move_whose_old_delete_raised() -> None:
    show, tl = _branch_show()
    assert _mirror(show, tl, [_note("n1", 10, "a")]).status is OpStatus.OK
    _fault(tl, "DeleteMarkerAtFrame", 1, "raise")
    failed = _mirror(show, tl, [_note("n1", 40, "a")])
    assert failed.error.detail["applied"] == {
        "added": [{"note_id": "n1", "frame": 40}],
        "deleted": [],
    }
    assert set(tl.markers) == {10, 40}  # the interrupted move left two markers for n1
    rerun = _mirror(show, tl, [_note("n1", 40, "a")])
    assert rerun.status is OpStatus.OK, rerun.error
    assert [(m.note_id, m.frame, m.status) for m in rerun.value.markers] == [("n1", 40, "present")]
    assert [(c.note_id, c.frame) for c in rerun.value.cleared] == [("n1", 10)]
    assert _marks(tl) == {40: (_cd("n1"), "a")}


def test_the_stale_frame_of_a_half_done_move_may_be_another_notes_target() -> None:
    show, tl = _branch_show()
    _mirror(show, tl, [_note("n1", 10, "a")])
    _fault(tl, "DeleteMarkerAtFrame", 1, "raise")
    _mirror(show, tl, [_note("n1", 40, "a")])
    rerun = _mirror(show, tl, [_note("n1", 40, "a"), _note("n2", 10, "b")])
    assert rerun.status is OpStatus.OK, rerun.error
    assert _marks(tl) == {10: (_cd("n2"), "b"), 40: (_cd("n1"), "a")}


def test_duplicates_that_are_not_a_half_done_move_are_refused_and_left_alone() -> None:
    show, tl = _branch_show()
    _mirror(show, tl, [_note("n1", 10, "a")])
    tl.AddMarker(30, "Cyan", "", "a", 1, _cd("n1"))  # a copy, say pasted by a human
    tl.AddMarker(5, "Blue", "", "human", 1, "")
    before = _marks(tl)
    for notes, prune in (
        ([_note("n1", 50, "a")], True),  # neither marker is the note as asked
        ([_note("n2", 20, "b")], False),  # n1 is not mirrored and kept: which one stays?
    ):
        result = _mirror(show, tl, notes, prune=prune)
        assert result.status is OpStatus.FAILED
        assert result.error.code == "duplicate_marker_custom_data"
        assert _marks(tl) == before  # refused in the pre-flight: nothing touched


def test_a_copy_beside_the_note_as_asked_is_cleared() -> None:
    show, tl = _branch_show()
    _mirror(show, tl, [_note("n1", 10, "a")])
    tl.AddMarker(11, "Cyan", "", "a", 1, _cd("n1"))
    result = _mirror(show, tl, [_note("n1", 10, "a")])
    assert result.status is OpStatus.OK  # 11 is not at the note's frame, so 10 is the one
    assert _marks(tl) == {10: (_cd("n1"), "a")}


# -- the fault fuzz: applied is exact, non-review markers are never touched, re-runs converge --


def _fuzz_case(seed: int) -> list[str]:
    rng = random.Random(seed)
    show, tl = _branch_show()
    for frame in rng.sample(range(DURATION), rng.randint(0, 10)):
        kind = rng.choice(["human", "other", "bad", "ours", "ours", "ours"])
        custom = {
            "human": "",
            "other": f"aar:note:rev2:x{frame}",
            "bad": "aar:note:rev1",
            "ours": _cd(f"n{rng.randint(0, 6)}"),
        }[kind]
        tl.AddMarker(frame, rng.choice(["Cyan", "Blue"]), "", rng.choice("ab"), 1, custom)
    ids = rng.sample([f"n{i}" for i in range(7)], rng.randint(0, 5))
    notes = [_note(n, rng.randrange(DURATION), rng.choice("abc")) for n in ids]
    prune = rng.random() < 0.7
    before_all = {f: dict(m) for f, m in tl.markers.items()}
    not_ours = {f: m for f, m in before_all.items() if not m["customData"].startswith(_cd(""))}
    before = _ours(tl)
    _fault(
        tl,
        rng.choice(["AddMarker", "DeleteMarkerAtFrame"]),
        rng.randint(1, 4),
        rng.choice(["raise", "raise-after", "gateway", "false"]),
    )
    first = _mirror(show, tl, notes, prune=prune)
    problems: list[str] = []
    if first.status is OpStatus.FAILED and first.error.code != "duplicate_marker_custom_data":
        applied = first.error.detail.get("applied")
        refused_early = first.error.code in ("marker_conflict", "note_out_of_range")
        if applied is None and not refused_early:
            problems.append(f"no applied for {first.error.code}")
        elif applied is not None and _calc(before, applied) != _ours(tl):
            problems.append(f"applied {applied} != markers {sorted(_ours(tl))}")
    for frame, marker in not_ours.items():
        if tl.markers.get(frame) != marker:
            problems.append(f"a marker that is not ours changed at {frame}")
    for method in ("AddMarker", "DeleteMarkerAtFrame"):
        setattr(tl, method, getattr(type(tl), method).__get__(tl))
    if first.status is OpStatus.FAILED and first.error.code not in (
        "marker_conflict",
        "note_out_of_range",
        "duplicate_marker_custom_data",
    ):
        rerun = _mirror(show, tl, notes, prune=prune)
        if rerun.status is not OpStatus.OK:
            problems.append(f"re-run after {first.error.code}: {rerun.error.code}")
        elif {(f, n) for f, n in _ours(tl)} != {(x["frame"], x["note_id"]) for x in notes} | (
            set() if prune else {(f, n) for f, n in before if n not in ids}
        ):
            problems.append(f"re-run did not converge: {sorted(_ours(tl))}")
    return problems


@pytest.mark.parametrize("block", range(4))
def test_mirror_fault_fuzz_reports_applied_exactly_and_reruns_converge(block: int) -> None:
    failures = {}
    for seed in range(block * 100, block * 100 + 100):
        problems = _fuzz_case(seed)
        if problems:
            failures[seed] = problems
    assert failures == {}
