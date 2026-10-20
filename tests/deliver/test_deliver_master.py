"""``deliver_master``: pull, store (role master), QA, record the hash and the measured QA facts.

No transcode and no review request. The master is stored whether QA passes or fails; the hash
fact is always written; a QA fact is written only for a check that measured a value.
"""

from __future__ import annotations

from pathlib import Path

from deliver_support import (
    AGENT,
    FakeEditHost,
    Hub,
    QaStub,
    make_hub,
    make_profile,
    qa_report,
    render_output,
)

from aar.deliver.service import deliver_master
from aar.hub.facts import facts_view
from aar.qa.report import QACheck

MB = 1_000_000


def _master(hub: Hub, sources: Path, workdir: Path, qa: QaStub, *, source: str = "render_720p.mp4"):
    return deliver_master(
        project_id=hub.project_id,
        deliverable="longform-1",
        actor=AGENT,
        profile=make_profile(),
        kind="longform",
        render_output=render_output(sources / source, render_job_id="rj-9"),
        workdir=str(workdir),
        edit_host=FakeEditHost(sources),
        log=hub.log,
        store=hub.store,
        qa_check=qa,
    )


def _qa_measured(status: str = "pass", *, truepeak: float | None = -1.5, duration=True) -> QaStub:
    checks = [
        QACheck(id="decode", status=status, message="x"),
        QACheck(
            id="loudness",
            status="pass",
            message="x",
            measured={"integrated_lufs": -22.3, "true_peak_dbtp": truepeak},
        ),
    ]
    if duration:
        checks.append(
            QACheck(id="duration", status="pass", message="x", measured={"render_s": 6.25})
        )
    return QaStub(qa_report(checks, kind="longform"))


def test_master_is_stored_and_hash_and_qa_facts_recorded(tmp_path: Path, sources: Path) -> None:
    hub = make_hub(tmp_path)
    result = _master(hub, sources, tmp_path / "work", _qa_measured())
    assert result.role == "master"
    assert result.qa_status == "pass"
    assert result.review_id is None
    assert result.transcoded is False
    assert hub.store.has(result.artifact_id)
    meta = hub.store.get_meta(result.artifact_id)
    assert meta.meta["role"] == "master"
    assert meta.producing_op == "deliver.master"

    facts = facts_view(hub.log, hub.project_id)
    assert facts.get("render.hash.longform-1").current.value == {
        "sha256": result.artifact_id,
        "bytes": result.size,
    }
    assert facts.get("qa.lufs.longform-1").current.value == {"value": -22.3, "unit": "LUFS"}
    assert facts.get("qa.truepeak.longform-1").current.value == {"value": -1.5, "unit": "dBTP"}
    assert facts.get("qa.duration.longform-1").current.value == {"value": 6.25, "unit": "s"}
    # No review was requested for a master.
    assert list(
        hub.log.iter_events(types=["graph.node_added"], project_id=hub.project_id)
    ) and not [
        e
        for e in hub.log.iter_events(types=["graph.node_added"], project_id=hub.project_id)
        if e.payload.get("kind") == "review"
    ]


def test_master_writes_only_measured_qa_facts(tmp_path: Path, sources: Path) -> None:
    hub = make_hub(tmp_path)
    # true peak not measured (None) and no duration check: those facts must be skipped.
    result = _master(hub, sources, tmp_path / "work", _qa_measured(truepeak=None, duration=False))
    facts = facts_view(hub.log, hub.project_id)
    assert facts.get("qa.lufs.longform-1") is not None
    assert facts.get("qa.truepeak.longform-1") is None
    assert facts.get("qa.duration.longform-1") is None
    assert set(result.fact_keys) == {"render.hash.longform-1", "qa.lufs.longform-1"}


def test_master_qa_fail_still_writes_the_hash_and_stores(tmp_path: Path, sources: Path) -> None:
    hub = make_hub(tmp_path)
    # A decode failure with no measured numbers: the master is still stored, the hash still written.
    qa = QaStub(qa_report([QACheck(id="decode", status="fail", message="bad")], kind="longform"))
    result = _master(hub, sources, tmp_path / "work", qa)
    assert result.qa_status == "fail"
    assert hub.store.has(result.artifact_id)
    facts = facts_view(hub.log, hub.project_id)
    assert facts.get("render.hash.longform-1") is not None
    assert result.fact_keys == ["render.hash.longform-1"]  # no QA facts (nothing measured)
    assert result.review_id is None
    review_nodes = [
        e
        for e in hub.log.iter_events(types=["graph.node_added"], project_id=hub.project_id)
        if e.payload.get("kind") == "review"
    ]
    assert review_nodes == []
