"""Round-3 mutant pins for S11's node side: names, contention, mirror bounds, guard defaults.

Each test names the validator's mutant it pins (see aar-evidence/s11/r3/mut/catalogue_r3.py).
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import pytest
from pydantic import ValidationError

import aar.node.feed as feed_module
from aar.hub.events import EventLog
from aar.node.branches import (
    branch_name,
    branch_version_for,
    note_marker_custom_data,
    parse_branch,
    parse_note_marker,
    valid_agent_id,
)
from aar.node.broker import BrokerConfig, OpStatus
from aar.node.collab import contention, explain_op_contention
from aar.node.feed import EventLogTransport, NodeFeed
from aar.node.leases import Holder, LeaseManager, Scope
from aar.node.locator import ProjectLocator
from aar.node.ops import OpError, make_hot_op, submit_op

from .fake_resolve import FakeClock
from .ops_support import AGENT, Show, content_writes

HUMAN = Holder.human("operator")
START = 108000


# -- branch names (B01 B03 B04 B06 B07 MN12 MN13 MN14 MN15) and note markers (B08 B09) --------


@pytest.mark.parametrize(
    "name",
    [
        "Main [agent:x v1] copy",  # B01: suffix not at the very end
        "Main [agent:x v0]",  # MN13: v0
        "Main [agent:x v01]",  # MN13: leading zero
        " [agent:x v1]",  # MN15: empty source
        "[agent:x v1]",
        "Main [agent:a b v1]",  # MN12: a space is not in the id charset
        "Main [agent:a:b v1]",
        f"Main [agent:{'a' * 129} v1]",  # MN14: id too long
    ],
)
def test_a_string_that_is_not_exactly_a_branch_name_does_not_parse(name: str) -> None:
    assert parse_branch(name) is None


def test_a_branch_name_round_trips_and_numbers_by_source_and_agent() -> None:
    name = branch_name("Main", "thread-1", 2)
    assert name == "Main [agent:thread-1 v2]"
    assert parse_branch(name).version == 2
    assert branch_version_for(name, "Main", "thread-1") == 2
    assert branch_version_for(name, "Main", "thread-2") is None  # B06: another agent's
    assert branch_version_for(name, "Interview", "thread-1") is None  # B07: another source


@pytest.mark.parametrize(
    ("source", "agent", "version"),
    [("Main", "thread-1", 0), ("Main", "a]b", 1), ("Main", "a\nb", 1), ("", "a", 1)],
    ids=["v0", "bracket-id", "newline-id", "empty-source"],
)
def test_branch_name_refuses_what_could_not_parse_back(
    source: str, agent: str, version: int
) -> None:
    with pytest.raises(ValueError):  # B03, B04
        branch_name(source, agent, version)


def test_agent_ids_are_bounded_and_restricted() -> None:
    assert valid_agent_id("a" * 128) and not valid_agent_id("a" * 129)  # MN14
    assert not valid_agent_id("a b") and not valid_agent_id("a:b")  # MN12


def test_note_markers_parse_strictly() -> None:
    assert parse_note_marker("aar:note:rev1:n1") == ("rev1", "n1")
    assert parse_note_marker("aar:note:a:b:c") is None  # B08: no greedy review id
    assert parse_note_marker("aar:note:rev 1:n1") is None
    with pytest.raises(ValueError):  # B09: the review id is checked too
        note_marker_custom_data("rev:1", "n1")
    with pytest.raises(ValueError):
        note_marker_custom_data("rev1", "n\n1")


# -- contention matches the broker (C01 C05 C06 C07 CN02) -------------------------------------


def _quiet_guard_world() -> Show:
    show = Show()  # past the warm-up; a fresh observation of an unchanged context: idle
    show.presence.observe(show.gw.get_context(), show.clock())
    return show


def test_one_lease_blocking_two_named_timelines_is_listed_once() -> None:
    show = _quiet_guard_world()
    show.leases.acquire(HUMAN, Scope.PROJECT, (show.pid,), ttl=None)
    report = contention(
        show.leases,
        show.presence,
        holder=AGENT,
        project_id=show.pid,
        timeline_ids=("tl-a", "tl-b"),
        now=show.clock(),
    )
    assert len(report.leases) == 1  # C01


def test_held_seconds_count_up_from_the_grant() -> None:
    leases = LeaseManager(FakeClock())
    leases.acquire(HUMAN, Scope.PROJECT, ("p",), ttl=None)
    guard_show = _quiet_guard_world()
    report = contention(
        leases,
        guard_show.presence,
        holder=AGENT,
        project_id="p",
        now=1030.0,
    )
    assert report.leases[0].held_seconds == pytest.approx(30.0)  # C05


def test_explaining_an_op_uses_the_timelines_it_names() -> None:
    show = _quiet_guard_world()
    show.leases.acquire(HUMAN, Scope.TIMELINE, (show.pid, show.main.GetUniqueId()), ttl=None)

    def explain(timeline: str) -> Any:
        return explain_op_contention(
            show.leases,
            show.presence,
            "markers.add",
            {"timeline": show.ref(timeline), "markers": [{"frame": 1}]},
            target=show.locator(),
            holder=AGENT,
            now=show.clock(),
        )

    assert explain("Main").blocked is True
    assert explain("Main [agent]").blocked is False  # C06: another timeline is free


def test_explaining_an_op_needs_a_project_id() -> None:
    show = _quiet_guard_world()
    with pytest.raises(ValueError, match="project_id"):  # C07
        explain_op_contention(
            show.leases,
            show.presence,
            "timeline.list",
            {},
            target=ProjectLocator(("Shows",), "Show", None),
            holder=AGENT,
            now=show.clock(),
        )


def test_the_brokers_own_instance_hold_is_not_reported_as_contention() -> None:
    show = _quiet_guard_world()
    broker_id = BrokerConfig().broker_id
    show.leases.acquire(
        Holder.agent(broker_id), Scope.INSTANCE, (BrokerConfig().instance_id,), ttl=60
    )
    report = contention(
        show.leases, show.presence, holder=AGENT, project_id=show.pid, now=show.clock()
    )
    assert report.blocked is False and report.leases == ()  # CN02


# -- mirror input bounds, refused at submit (O05 O06 O07 O08 MN07 MN09 MN11) -------------------


def _branch(show: Show) -> dict[str, str]:
    branch = show.make_branch()
    show.project.timeline(branch.timeline.name).place("video", 1, show.cam_a, START, 1000)
    return branch.timeline.model_dump()


def _note(note_id: str = "n1", frame: int = 10, **kw: Any) -> dict[str, Any]:
    return {"note_id": note_id, "frame": frame, "text": "t", **kw}


@pytest.mark.parametrize(
    "params",
    [
        {"notes": [_note(frame=-1)]},  # O05
        {"notes": [_note(f"n{i}", i) for i in range(1001)]},  # O06
        {"notes": [_note("n1", 10), _note("n1", 20)]},  # O07
        {"review_id": "rev 1", "notes": [_note()]},  # O08
        {"review_id": "rev:1", "notes": [_note()]},
        {"notes": [_note(f"n{i}", i, text="x" * 4000) for i in range(300)]},  # MN07
        {"notes": [_note(timecode="1:00:00")]},  # MN09
        {"notes": [_note(timecode="01:00:00:10\n")]},
    ],
    ids=[
        "negative-frame",
        "too-many-notes",
        "duplicate-note-ids",
        "space-review-id",
        "colon-review-id",
        "total-text-over-cap",
        "bad-timecode",
        "timecode-newline",
    ],
)
def test_mirror_refuses_bad_input_at_submit(params: dict[str, Any]) -> None:
    show = Show()
    ref = _branch(show)
    with pytest.raises(ValidationError):
        submit_op(
            show.broker,
            "markers.mirror_review",
            {"timeline": ref, "review_id": "rev1", **params},
            target=show.locator(),
            holder=AGENT,
            branch_registry=show.branch_registry,
        )
    assert show.broker.pending() == []


@pytest.mark.parametrize("review_id", ["rev 1", "rev:1", "r\n1", "x" * 129])
def test_read_review_refuses_a_bad_review_id(review_id: str) -> None:
    show = Show()
    ref = _branch(show)
    with pytest.raises(ValidationError):  # MN11
        submit_op(
            show.broker,
            "markers.read_review",
            {"timeline": ref, "review_id": review_id},
            target=show.locator(),
            holder=AGENT,
        )


# -- mirror pre-flight and read-back (O12 O16 MN20) --------------------------------------------


def _mirror(show: Show, ref: dict[str, str], notes: list[dict], review_id: str = "rev1") -> Any:
    return show.run(
        "markers.mirror_review",
        {"timeline": ref, "review_id": review_id, "notes": notes},
        require_branch=True,
    )


def test_two_notes_on_one_frame_are_refused_before_any_write() -> None:
    show = Show()
    ref = _branch(show)
    before = len(content_writes(show.fake))
    result = _mirror(show, ref, [_note("n1", 10), _note("n2", 10)])
    assert result.status is OpStatus.FAILED and result.error.code == "marker_conflict"  # O12
    assert len(content_writes(show.fake)) == before


def test_a_note_on_another_reviews_marker_is_refused_before_any_write() -> None:
    show = Show()
    ref = _branch(show)
    assert _mirror(show, ref, [_note("n1", 20)], review_id="rev2").status is OpStatus.OK
    before = len(content_writes(show.fake))
    result = _mirror(show, ref, [_note("n1", 20)])
    assert result.status is OpStatus.FAILED and result.error.code == "marker_conflict"  # MN20
    assert len(content_writes(show.fake)) == before


def test_a_marker_that_reads_back_with_other_custom_data_is_not_ours() -> None:
    show = Show()
    ref = _branch(show)
    timeline = show.project.timeline(ref["name"])
    real = timeline.AddMarker
    timeline.AddMarker = lambda f, c, n, note, d, cd=None: real(f, c, n, note, d, "")
    result = _mirror(show, ref, [_note("n1", 10)])
    assert result.status is OpStatus.FAILED and result.error.code == "add_marker_failed"  # O16
    assert result.error.detail["applied"] == {"added": [], "deleted": []}


# -- the guard is on by default (GN01 GN06 GN07 GN14) ------------------------------------------


def test_make_hot_op_refuses_a_human_timeline_by_default() -> None:
    show = Show()
    with pytest.raises(OpError) as err:  # GN06: no require_branch passed
        make_hot_op(
            "markers.add",
            {"timeline": show.ref("Main"), "markers": [{"frame": 1}]},
            target=show.locator(),
            holder=AGENT,
            branch_registry=show.branch_registry,
        )
    assert err.value.code == "not_an_agent_branch"


def test_submit_op_refuses_a_human_timeline_by_default() -> None:
    show = Show()
    with pytest.raises(OpError) as err:  # GN07: no require_branch passed
        submit_op(
            show.broker,
            "markers.add",
            {"timeline": show.ref("Main"), "markers": [{"frame": 1}]},
            target=show.locator(),
            holder=AGENT,
            branch_registry=show.branch_registry,
        )
    assert err.value.code == "not_an_agent_branch" and show.main.GetMarkers() == {}


def test_a_human_with_the_agents_id_does_not_inherit_its_branch() -> None:
    show = Show()
    branch = show.make_branch()
    with pytest.raises(OpError) as err:  # GN01
        submit_op(
            show.broker,
            "markers.add",
            {"timeline": branch.timeline.model_dump(), "markers": [{"frame": 1}]},
            target=show.locator(),
            holder=Holder.human(AGENT.id),
            branch_registry=show.branch_registry,
        )
    assert err.value.code == "not_an_agent_branch"


def test_a_human_holder_cannot_branch() -> None:
    show = Show()
    result = show.run("timeline.branch", {"source": show.ref("Main")}, holder=HUMAN)
    assert result.status is OpStatus.FAILED and result.error.code == "no_agent_holder"  # GN14
    assert len(show.project.timelines) == 2


# -- feed construction and the gap time (N16 N17 L02) -----------------------------------------


def test_the_outbox_needs_room_for_one_spec(tmp_path) -> None:
    with pytest.raises(ValueError, match="outbox_max"):  # N17
        NodeFeed("node-1", EventLogTransport(EventLog(tmp_path / "hub.db")), outbox_max=0)


def test_the_gap_is_dated_by_its_first_drop(tmp_path, monkeypatch) -> None:
    log = EventLog(tmp_path / "hub.db")
    feed = NodeFeed("node-1", EventLogTransport(log), wall_offset=lambda: 1.8e9, outbox_max=1)
    leases = LeaseManager(FakeClock(), sink=feed.lease_sink)
    times = iter(dt.datetime(2027, 1, 1, h, tzinfo=dt.UTC) for h in range(1, 10))

    class _Clock(dt.datetime):
        @classmethod
        def now(cls, tz: Any = None) -> dt.datetime:
            return next(times)

    monkeypatch.setattr(feed_module, "datetime", _Clock)
    for i in range(3):  # two drops, at 01:00 and 02:00
        leases.acquire(AGENT, Scope.PROJECT, (f"p{i}",), ttl=60)
    feed.flush()
    gap = log.read(types=["feed.gap"])[0].payload
    assert gap["dropped"] == 2 and gap["at"] == "2027-01-01T01:00:00+00:00"  # N16


def test_acquire_refuses_a_too_long_key_part_before_granting() -> None:
    leases = LeaseManager(FakeClock())
    with pytest.raises(ValueError, match="key"):  # L02
        leases.acquire(AGENT, Scope.PROJECT, ("p" * 257,), ttl=60)
    assert leases.active() == []
