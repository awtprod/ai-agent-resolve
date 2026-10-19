"""S11f: timeline.branch follow-ups (LOW 6, LOW 7, LOW 8, V03)."""

from __future__ import annotations

from typing import Any

import pytest

from aar.node.leases import Holder
from aar.node.ops import OpError, submit_op

from .fake_resolve import FakeResolve
from .ops_support import AGENT, Show

VERTICAL = {"width": 1080, "height": 1920}


def _resolution(show: Show, name: str) -> tuple[str, str]:
    settings = show.project.timeline(name).GetSettings()
    return settings["timelineResolutionWidth"], settings["timelineResolutionHeight"]


def _names(show: Show) -> list[str]:
    return [t.GetName() for t in show.project.timelines]


# -- LOW 8: refusals that need no Resolve call are made at submit ---------------------------


@pytest.mark.parametrize(
    ("holder", "registry", "code"),
    [
        (Holder.human("op"), True, "no_agent_holder"),
        (Holder.agent("job-runner:node-1"), True, "invalid_agent_id"),
        (AGENT, False, "no_branch_registry"),
    ],
    ids=["human-holder", "runner-style-holder", "no-registry"],
)
@pytest.mark.parametrize("require_branch", [True, False], ids=["guard-on", "guard-off"])
def test_a_branch_refused_for_its_holder_or_registry_never_touches_resolve(
    holder: Holder, registry: bool, code: str, require_branch: bool
) -> None:
    show = Show()
    other = show.fake.project("Other")
    source = {"name": "Other main", "unique_id": other.timelines[0].GetUniqueId()}
    before = len(show.fake.calls)
    with pytest.raises(OpError) as err:
        submit_op(
            show.broker,
            "timeline.branch",
            {"source": source},
            target=show.locator("Other"),
            holder=holder,
            require_branch=require_branch,
            branch_registry=show.branch_registry if registry else None,
        )
    assert err.value.code == code
    show.broker.run_until_idle()
    # Never queued: no LoadProject, no SaveProject, no render-preset round trip, nothing.
    assert show.fake.calls[before:] == []
    assert show.fake.GetProjectManager().GetCurrentProject().GetName() == "Show"


# -- LOW 6: no resolution means the source's, as for timeline.duplicate(reuse) --------------


def test_a_plain_branch_never_reuses_a_branch_at_another_resolution() -> None:
    show = Show()
    landscape = show.make_branch()  # v1, the source's 1920x1080
    vertical = show.ok("timeline.branch", {"source": show.ref("Main"), "resolution": VERTICAL})
    assert vertical.version == 2 and _resolution(show, vertical.timeline.name) == ("1080", "1920")
    plain = show.ok("timeline.branch", {"source": show.ref("Main")})
    assert not plain.created and plain.timeline.unique_id == landscape.timeline.unique_id


def test_a_plain_branch_with_only_a_vertical_branch_cuts_a_new_one() -> None:
    show = Show()
    vertical = show.ok("timeline.branch", {"source": show.ref("Main"), "resolution": VERTICAL})
    plain = show.ok("timeline.branch", {"source": show.ref("Main")})
    assert plain.created and plain.version == vertical.version + 1
    assert _resolution(show, plain.timeline.name) == _resolution(show, "Main")


# -- V03: a renamed branch still holds its version --------------------------------------------


def test_a_branch_a_human_renamed_still_holds_its_version() -> None:
    show = Show()
    first = show.make_branch()
    show.project.timeline(first.timeline.name)._name = "Main FINAL (human)"
    again = show.ok("timeline.branch", {"source": show.ref("Main"), "if_exists": "new"})
    assert again.created and again.version == 2  # v1 is still owned, under another name
    assert again.timeline.name == f"Main [agent:{AGENT.id} v2]"


# -- LOW 7: a copy made by a branch that then fails is deleted, never left unowned ----------


def _failing_resolution(show: Show, monkeypatch: pytest.MonkeyPatch) -> None:
    """SetSettings refuses a resolution on any timeline but Main (the copy's step fails)."""
    timeline_type = type(show.main)
    real = timeline_type.SetSettings

    def refuse(self: Any, settings: dict[str, Any]) -> bool:
        if self.GetUniqueId() != show.main.GetUniqueId() and "timelineResolutionWidth" in settings:
            return False
        return real(self, settings)

    monkeypatch.setattr(timeline_type, "SetSettings", refuse)


def test_a_branch_whose_resolution_fails_deletes_its_copy_and_a_retry_cuts_the_same_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    show = Show()
    before = _names(show)
    _failing_resolution(show, monkeypatch)
    error = show.fails("timeline.branch", {"source": show.ref("Main"), "resolution": VERTICAL})
    assert error.code == "resolution_set_failed"
    name = f"Main [agent:{AGENT.id} v1]"
    assert error.detail["removed"]["name"] == name
    assert _names(show) == before  # no unowned half-made branch is left behind
    assert show.branch_registry.owner(show.pid, error.detail["removed"]["unique_id"]) is None
    deletes = [call for call in show.fake.calls if call[0] == "DeleteTimelines"]
    assert deletes == [("DeleteTimelines", [error.detail["removed"]["unique_id"]])]

    monkeypatch.undo()
    retry = show.ok("timeline.branch", {"source": show.ref("Main"), "resolution": VERTICAL})
    assert retry.created and retry.version == 1 and retry.timeline.name == name
    assert _names(show) == [*before, name]


def test_a_copy_that_cannot_be_deleted_is_reported_as_left_behind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    show = Show()
    show.fake.faults.add("DeleteTimelines:fail")
    _failing_resolution(show, monkeypatch)
    error = show.fails("timeline.branch", {"source": show.ref("Main"), "resolution": VERTICAL})
    assert error.code == "resolution_set_failed" and "removed" not in error.detail
    assert error.detail["left_behind"]["name"] == f"Main [agent:{AGENT.id} v1]"
    assert error.detail["left_behind"]["name"] in _names(show)


def test_a_branch_refused_before_any_copy_deletes_nothing() -> None:
    show = Show()
    show.fake.faults.add("DuplicateTimeline:detached")  # returns a timeline not in the project
    error = show.fails("timeline.branch", {"source": show.ref("Main")})
    assert error.code == "duplicate_unverified" and "removed" not in error.detail
    assert not [call for call in show.fake.calls if call[0] == "DeleteTimelines"]


def test_the_clean_up_never_deletes_a_same_named_timeline_that_was_there_before(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A human makes a timeline with the very name the branch is about to take (a race the
    # version scan cannot see: here the scan is blinded). The duplicate is refused, and the
    # clean-up must not take the human's timeline for its own half-made copy.
    fake = FakeResolve(
        {"Show": ["Main", "Main [agent]", f"Main [agent:{AGENT.id} v1]"]},
        current="Show",
        folder=("Shows",),
    )
    show = Show(fake)
    monkeypatch.setattr("aar.node.ops.collab.branch_version_for", lambda *args: None)
    before = {t.GetUniqueId() for t in show.project.timelines}
    error = show.fails("timeline.branch", {"source": show.ref("Main")})
    assert error.code == "timeline_exists" and "removed" not in error.detail
    assert {t.GetUniqueId() for t in show.project.timelines} == before
    assert not [call for call in show.fake.calls if call[0] == "DeleteTimelines"]


def test_reuse_returns_the_latest_of_several_owned_branches_at_the_asked_resolution() -> None:
    # V05: with v1 and v2 both at the source's resolution, a plain re-run reuses v2.
    show = Show()
    first = show.make_branch()
    second = show.ok("timeline.branch", {"source": show.ref("Main"), "if_exists": "new"})
    assert (first.version, second.version) == (1, 2)
    again = show.ok("timeline.branch", {"source": show.ref("Main")})
    assert not again.created and again.version == 2
    assert again.timeline.unique_id == second.timeline.unique_id
