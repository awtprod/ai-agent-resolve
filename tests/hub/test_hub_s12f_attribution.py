"""S12f item 4: forged decisions and withdrawals must not credit the wrong actor.

A decision or withdrawal counts only if it was written together with the review's
status change (``decide``/``withdraw`` do that in one transaction). One appended
straight to the log is skipped and listed, however well-formed it is.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aar.hub import reviews
from aar.hub.artifacts import ArtifactStore
from aar.hub.events import Actor, EventLog
from aar.hub.graph import WorkGraph
from aar.hub.registry import Registry

HUMAN = Actor("human", "operator-1")
FORGER = Actor("human", "operator-2")
AGENT = Actor("agent", "agent-1")
OTHER_AGENT = Actor("agent", "agent-2")


class Hub:
    def __init__(self, root: Path) -> None:
        self.log = EventLog(root / "hub.db")
        store = ArtifactStore(root / "artifacts", self.log)
        self.pid = Registry(self.log).create_project("demo-show", HUMAN).id
        op = WorkGraph(self.log, self.pid).add_node("op", AGENT, title="build").id
        render = store.put(b"render", actor=AGENT, mime="video/mp4", project_id=self.pid)
        self.review = reviews.request_review(
            self.log, store, self.pid, op, [render], AGENT, title="cut v1", fps=24
        ).id

    def forge(self, event_type: str, actor: Actor, payload: dict[str, object]) -> str:
        return self.log.append(event_type, actor, payload, project_id=self.pid).id

    def skipped_ids(self) -> set[str]:
        _, skipped = reviews.review_view(self.log, self.pid, self.review)
        return {item.event_id for item in skipped}


@pytest.fixture
def hub(tmp_path: Path) -> Hub:
    return Hub(tmp_path)


def test_a_forged_withdrawal_does_not_take_credit_for_the_real_one(hub: Hub) -> None:
    forged = hub.forge(
        "review.withdrawn", OTHER_AGENT, {"review_id": hub.review, "reason": "I withdrew it"}
    )
    reviews.withdraw(hub.log, hub.pid, hub.review, "superseded by v2", AGENT)
    review = reviews.get_review(hub.log, hub.pid, hub.review)
    assert review is not None and review.withdrawal is not None
    assert review.withdrawal.by == AGENT
    assert review.withdrawal.reason == "superseded by v2"
    assert forged in hub.skipped_ids()


def test_a_forged_human_decision_is_not_a_decision(hub: Hub) -> None:
    forged = hub.forge(
        "review.decided", FORGER, {"review_id": hub.review, "decision": "approve", "reason": "ok"}
    )
    review = reviews.get_review(hub.log, hub.pid, hub.review)
    assert review is not None and review.decision is None and review.pending
    assert forged in hub.skipped_ids()
    # It does not block, or take credit from, the real decision.
    reviews.decide(hub.log, hub.pid, hub.review, "reject", "audio pops", HUMAN)
    review = reviews.get_review(hub.log, hub.pid, hub.review)
    assert review is not None and review.decision is not None
    assert (review.decision.by, review.decision.decision) == (HUMAN, "reject")


def test_a_decision_that_contradicts_the_status_it_follows_is_skipped(hub: Hub) -> None:
    # A raw (but legal, human) status change to done, then a "reject" right after it.
    hub.forge("graph.node_status_changed", HUMAN, {"node_id": hub.review, "to": "done"})
    forged = hub.forge(
        "review.decided", HUMAN, {"review_id": hub.review, "decision": "reject", "reason": "x"}
    )
    review = reviews.get_review(hub.log, hub.pid, hub.review)
    assert review is not None and review.status == "done" and review.decision is None
    assert forged in hub.skipped_ids()


def test_a_decision_by_someone_other_than_who_moved_the_status_is_skipped(hub: Hub) -> None:
    hub.forge("graph.node_status_changed", HUMAN, {"node_id": hub.review, "to": "done"})
    forged = hub.forge(
        "review.decided", FORGER, {"review_id": hub.review, "decision": "approve", "reason": "x"}
    )
    review = reviews.get_review(hub.log, hub.pid, hub.review)
    assert review is not None and review.decision is None
    assert forged in hub.skipped_ids()


def test_genuine_decisions_and_withdrawals_are_still_credited(hub: Hub, tmp_path: Path) -> None:
    decision = reviews.decide(hub.log, hub.pid, hub.review, "approve", "ship it", HUMAN)
    review = reviews.get_review(hub.log, hub.pid, hub.review)
    assert review is not None and review.decision == decision
    other = Hub(tmp_path / "second")
    reviews.withdraw(other.log, other.pid, other.review, "superseded", AGENT)
    withdrawn = reviews.get_review(other.log, other.pid, other.review)
    assert withdrawn is not None and withdrawn.withdrawal is not None
    assert withdrawn.withdrawal.by == AGENT
    assert other.skipped_ids() == set()
