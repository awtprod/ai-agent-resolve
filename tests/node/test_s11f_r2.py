"""S11f round 2: the branch clean-up deletes only the copy it made, by id (HIGH-1), and the
refused-gap give-up count is exact (LOW-1)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from aar.hub.events import EventLog
from aar.node.feed import BACKOFF_MAX, MAX_SPEC_ATTEMPTS, EventLogTransport, wire_feed
from aar.node.leases import Holder, Scope

from .fake_resolve import FakeClock, FakeTimeline
from .ops_support import AGENT, Show

VERTICAL = {"width": 1080, "height": 1920}
BRANCH_V1 = f"Main [agent:{AGENT.id} v1]"
RES_KEYS = ("timelineResolutionWidth", "timelineResolutionHeight")


def _ids(show: Show) -> set[str]:
    return {t.GetUniqueId() for t in show.project.timelines}


def _deletes(show: Show) -> list[tuple[Any, ...]]:
    return [call for call in show.fake.calls if call[0] == "DeleteTimelines"]


def _foreign(show: Show, name: str, uid: str) -> FakeTimeline:
    """Another actor (a human, or another node branching the same agent) makes ``name``."""
    timeline = FakeTimeline(show.fake, name, uid, show.project, start_frame=show.main.start_frame)
    show.project.timelines.append(timeline)
    return timeline


def _patch_duplicate(show: Show, monkeypatch: pytest.MonkeyPatch, wrapper: Any) -> None:
    real = type(show.main).DuplicateTimeline
    monkeypatch.setattr(
        type(show.main), "DuplicateTimeline", lambda self, name: wrapper(real, self, name)
    )


def _patch_settings(show: Show, monkeypatch: pytest.MonkeyPatch, wrapper: Any) -> None:
    """Route SetSettings on every timeline but Main (the copy) through ``wrapper``."""
    timeline_type = type(show.main)
    real = timeline_type.SetSettings

    def patched(self: Any, settings: dict[str, Any]) -> bool:
        if self.GetUniqueId() == show.main.GetUniqueId():
            return real(self, settings)
        return wrapper(real, self, settings)

    monkeypatch.setattr(timeline_type, "SetSettings", patched)


# -- (i) a same-named timeline made by someone else in the window is never deleted ------------


def test_a_foreign_same_named_timeline_made_during_the_op_is_never_deleted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The validator's val_low7: the other actor's timeline takes the name first, so
    DuplicateTimeline returns None (a name collision). Nothing was made: nothing is undone."""
    show = Show()

    def racing(real: Any, source: Any, name: str) -> Any:
        if name == BRANCH_V1 and "tl-FOREIGN" not in _ids(show):
            _foreign(show, name, "tl-FOREIGN")
        return real(source, name)

    _patch_duplicate(show, monkeypatch, racing)
    error = show.fails("timeline.branch", {"source": show.ref("Main")})
    assert error.code == "duplicate_failed"
    assert "tl-FOREIGN" in _ids(show)
    assert _deletes(show) == []
    assert "removed" not in error.detail and "left_behind" not in error.detail
    assert "created_timeline_id" not in error.detail


# -- (ii) with two same-named new timelines, only the carried id is touched -------------------


def test_with_a_same_named_twin_at_verify_only_the_carried_copy_is_deleted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Our copy is made, then a twin with the same name appears before the op verifies it."""
    show = Show()

    def twin(real: Any, source: Any, name: str) -> Any:
        made = real(source, name)
        _foreign(show, name, "tl-TWIN")
        return made

    _patch_duplicate(show, monkeypatch, twin)
    before = _ids(show)
    error = show.fails("timeline.branch", {"source": show.ref("Main")})
    assert error.code == "duplicate_unverified"
    ours = error.detail["created_timeline_id"]
    assert ours not in before and ours != "tl-TWIN"
    assert _deletes(show) == [("DeleteTimelines", [ours])]
    assert error.detail["removed"] == {"name": BRANCH_V1, "unique_id": ours}
    assert _ids(show) == before | {"tl-TWIN"}  # the twin, with the very same name, is kept


def test_with_a_same_named_twin_during_the_resolution_step_only_ours_is_deleted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    show = Show()

    def refuse_and_race(real: Any, timeline: Any, settings: dict[str, Any]) -> bool:
        if "timelineResolutionWidth" in settings:
            _foreign(show, BRANCH_V1, "tl-TWIN")
            return False
        return real(timeline, settings)

    _patch_settings(show, monkeypatch, refuse_and_race)
    before = _ids(show)
    error = show.fails("timeline.branch", {"source": show.ref("Main"), "resolution": VERTICAL})
    assert error.code == "resolution_set_failed"
    ours = error.detail["created_timeline_id"]
    assert _deletes(show) == [("DeleteTimelines", [ours])]
    assert error.detail["removed"]["unique_id"] == ours
    assert _ids(show) == before | {"tl-TWIN"}


def test_our_copy_renamed_by_someone_in_the_window_is_still_removed_by_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    show = Show()

    def refuse_and_rename(real: Any, timeline: Any, settings: dict[str, Any]) -> bool:
        if "timelineResolutionWidth" in settings:
            timeline._name = "renamed in the window"
            return False
        return real(timeline, settings)

    _patch_settings(show, monkeypatch, refuse_and_rename)
    before = _ids(show)
    error = show.fails("timeline.branch", {"source": show.ref("Main"), "resolution": VERTICAL})
    assert error.detail["removed"] == {
        "name": "renamed in the window",
        "unique_id": error.detail["created_timeline_id"],
    }
    assert _ids(show) == before


def test_a_copy_already_gone_before_the_clean_up_deletes_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The re-read before the delete: if the carried id is no longer there, nothing is touched.
    show = Show()

    def refuse_and_vanish(real: Any, timeline: Any, settings: dict[str, Any]) -> bool:
        if "timelineResolutionWidth" in settings:
            show.project.timelines.remove(timeline)  # someone deleted our copy meanwhile
            return False
        return real(timeline, settings)

    _patch_settings(show, monkeypatch, refuse_and_vanish)
    before = _ids(show)
    error = show.fails("timeline.branch", {"source": show.ref("Main"), "resolution": VERTICAL})
    assert error.code == "resolution_set_failed" and "created_timeline_id" in error.detail
    assert _deletes(show) == [] and "removed" not in error.detail
    assert _ids(show) == before


# -- (iii) a failure with no created id deletes nothing ---------------------------------------


def test_a_copy_resolve_made_but_did_not_return_is_never_deleted_by_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DuplicateTimeline makes the copy but returns None (a lost reply). The op cannot prove
    the new timeline is its own, so it deletes nothing (the error says why it failed)."""
    show = Show()

    def lost_reply(real: Any, source: Any, name: str) -> Any:
        real(source, name)
        return None

    _patch_duplicate(show, monkeypatch, lost_reply)
    before = _ids(show)
    error = show.fails("timeline.branch", {"source": show.ref("Main")})
    assert error.code == "duplicate_failed" and "created_timeline_id" not in error.detail
    assert _deletes(show) == []
    assert len(_ids(show) - before) == 1  # left for a human: it is not provably ours


def test_a_duplicate_that_returns_the_source_deletes_nothing() -> None:
    show = Show()
    show.fake.faults.add("DuplicateTimeline:returns-source")  # the copy is made, then refused
    before = _ids(show)
    error = show.fails("timeline.branch", {"source": show.ref("Main")})
    assert error.code == "duplicate_unverified" and "created_timeline_id" not in error.detail
    assert _deletes(show) == []
    assert before <= _ids(show)  # the source above all, and every timeline that was there


# -- (iv) every failure after the copy was made carries its id --------------------------------


def _duplicate_fails(show: Show, **params: Any) -> Any:
    return show.fails(
        "timeline.duplicate", {"source": show.ref("Main"), "name": "Main copy", **params}
    )


def _created(show: Show, error: Any, before: set[str]) -> str:
    created = error.detail["created_timeline_id"]
    assert created not in before and created in _ids(show)
    assert show.project.timeline("Main copy").GetUniqueId() == created
    return created


def test_resolution_set_failed_carries_the_created_id(monkeypatch: pytest.MonkeyPatch) -> None:
    show = Show()
    _patch_settings(
        show,
        monkeypatch,
        lambda real, tl, s: False if "timelineResolutionWidth" in s else real(tl, s),
    )
    before = _ids(show)
    error = _duplicate_fails(show, resolution=VERTICAL)
    assert error.code == "resolution_set_failed"
    _created(show, error, before)


def test_colour_settings_not_restored_carries_the_created_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    show = Show()
    # The custom resolution resets the colour science (the trap); putting it back fails.
    _patch_settings(
        show,
        monkeypatch,
        lambda real, tl, s: real(tl, s) if "timelineResolutionWidth" in s else False,
    )
    before = _ids(show)
    error = _duplicate_fails(show, resolution=VERTICAL)
    assert error.code == "colour_settings_not_restored"
    _created(show, error, before)
    assert error.detail["mismatches"]  # the original detail is kept beside the id


def test_resolution_unverified_carries_the_created_id(monkeypatch: pytest.MonkeyPatch) -> None:
    show = Show()

    def ignore_resolution(real: Any, timeline: Any, settings: dict[str, Any]) -> bool:
        real(timeline, {k: v for k, v in settings.items() if k not in RES_KEYS})
        return True

    _patch_settings(show, monkeypatch, ignore_resolution)
    before = _ids(show)
    error = _duplicate_fails(show, resolution=VERTICAL)
    assert error.code == "resolution_unverified"
    _created(show, error, before)


def test_duplicate_unverified_carries_the_created_id(monkeypatch: pytest.MonkeyPatch) -> None:
    show = Show()

    def twin(real: Any, source: Any, name: str) -> Any:
        made = real(source, name)
        _foreign(show, "someone else's timeline", "tl-OTHER")  # the count is off by one
        return made

    _patch_duplicate(show, monkeypatch, twin)
    before = _ids(show)
    error = _duplicate_fails(show)
    assert error.code == "duplicate_unverified"
    _created(show, error, before)


def test_a_resolve_call_that_raises_after_the_copy_carries_the_created_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    show = Show()

    def hiccup(real: Any, timeline: Any, settings: dict[str, Any]) -> bool:
        raise RuntimeError("Resolve stopped answering")

    _patch_settings(show, monkeypatch, hiccup)
    before = _ids(show)
    error = _duplicate_fails(show, resolution=VERTICAL)
    assert error.code == "resolve_call_failed"
    _created(show, error, before)


def test_failures_before_or_without_a_provable_copy_carry_no_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for fault in ("DuplicateTimeline:detached", "DuplicateTimeline:returns-source"):
        show = Show()
        show.fake.faults.add(fault)
        error = _duplicate_fails(show)
        assert error.code == "duplicate_unverified" and "created_timeline_id" not in error.detail
    show = Show()
    _foreign(show, "Main copy", "tl-TAKEN")  # the name is taken: refused before any copy
    error = _duplicate_fails(show, if_exists="error")
    assert error.code == "timeline_exists" and "created_timeline_id" not in error.detail


# -- LOW-1: a refused gap is tried exactly MAX_SPEC_ATTEMPTS times ---------------------------


def test_a_refused_gap_is_tried_exactly_max_spec_attempts_times(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "hub.db")
    real = EventLogTransport(log)
    gap_tries: list[str | None] = []

    class NoGaps:
        def deliver(self, spec: Any, *, node_id: str) -> None:
            if spec.type == "feed.gap":
                gap_tries.append(spec.idempotency_key)
                raise ValueError("the hub refuses gaps")
            real.deliver(spec, node_id=node_id)

    clock = FakeClock()
    feed, leases = wire_feed(
        "node-1", NoGaps(), clock=clock, wall_offset=lambda: 1.8e9, outbox_max=1
    )
    leases.acquire(Holder.agent("a"), Scope.PROJECT, ("pa",), ttl=600)
    leases.acquire(Holder.agent("b"), Scope.PROJECT, ("pb",), ttl=600)  # one drop: a gap
    for flush in range(1, MAX_SPEC_ATTEMPTS + 3):
        feed.flush()
        clock.advance(BACKOFF_MAX)
        given_up = [q.type for q in feed.quarantined] == ["feed.gap"]
        assert given_up is (flush >= MAX_SPEC_ATTEMPTS), flush
    assert len(gap_tries) == MAX_SPEC_ATTEMPTS  # not one more
    assert len(set(gap_tries)) == 1  # the same gap, resent unchanged
