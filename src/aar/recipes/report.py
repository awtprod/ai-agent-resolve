"""The built-in final step ``recipe.report``: write a deterministic ``REPORT.md`` artifact.

``recipe.report`` is a cold, idempotent step (recipes end with it). It renders the run as Markdown
from the log alone -- every step's attempts, rounds, durations, outcomes and error codes; the gate
decisions with the human who decided; the QA summaries a QA step recorded; the facts the run
recorded; and every produced artifact with its sha256 -- then stores it as an artifact. The text is
a pure function of the log, so :func:`build_report_text` has a golden test.

For now "archiving" (step 16) is this report plus the recipe node going ``done``; files move in
S16b.
"""

from __future__ import annotations

from aar.hub.common import parse_ts
from aar.hub.events import EventLog
from aar.hub.facts import facts_view
from aar.recipes.context import StepContext, StepOutput
from aar.recipes.state import OpAttempt, RunState, load_state

REPORT_MIME = "text/markdown"


def _duration(op: OpAttempt) -> str:
    if op.created is None or op.result_at is None:
        return "-"
    try:
        secs = (parse_ts(op.result_at) - parse_ts(op.created)).total_seconds()
    except (ValueError, OverflowError):
        return "-"
    return f"{secs:.3f}s"


def _step_order(state: RunState) -> list[str]:
    if state.recipe_def is not None:
        return [step.id for step in state.recipe_def.steps]
    return sorted({op.step_id for op in state.ops})


def build_report_text(log: EventLog, recipe_node: str) -> str:
    """Render ``REPORT.md`` for the run of ``recipe_node`` as Markdown (pure over the log)."""
    state = load_state(log, recipe_node)
    lines: list[str] = []
    recipe_name = state.recipe_data.recipe if state.recipe_data is not None else "?"
    version = state.recipe_def.version if state.recipe_def is not None else 0
    lines.append(f"# Recipe report: {recipe_name} v{version}")
    lines.append("")
    lines.append(f"- recipe_node: {recipe_node}")
    lines.append(f"- status: {state.recipe_status}")
    lines.append(f"- review rounds: used {state.current_round()} of {state.max_review_rounds}")
    lines.append("")

    lines.append("## Steps")
    for step_id in _step_order(state):
        attempts = sorted(state.attempts_for(step_id), key=lambda op: (op.round, op.attempt))
        kind = attempts[0].kind if attempts else "?"
        lines.append(f"### {step_id} ({kind})")
        if not attempts:
            lines.append("- (no attempts)")
        for op in attempts:
            outcome = op.result_outcome or ("cancelled" if op.is_cancelled else op.status)
            code = ""
            if op.result_error and isinstance(op.result_error, dict):
                code = f" [{op.result_error.get('code')}]"
            flag = " (superseded)" if op.superseded else ""
            lines.append(
                f"- round {op.round} attempt {op.attempt}: {outcome}{code} {_duration(op)}{flag}"
            )
        lines.append("")

    lines.append("## Gates")
    gate_lines = []
    for step_id in _step_order(state):
        for op in sorted(state.attempts_for(step_id), key=lambda op: (op.round, op.attempt)):
            if op.kind != "gate" or op.review_id is None:
                continue
            review = state.reviews.get(op.review_id)
            if review is None or review.decision is None:
                continue
            d = review.decision
            gate_lines.append(
                f"- {step_id} round {op.round}: {d.decision} by {d.by.kind}:{d.by.id} -- {d.reason}"
            )
    lines.extend(gate_lines or ["- (no decided gates)"])
    lines.append("")

    lines.append("## QA")
    qa_lines = []
    for step_id in _step_order(state):
        outputs = state.done_outputs(step_id)
        qa = outputs.get("qa") if isinstance(outputs, dict) else None
        if isinstance(qa, dict):
            passed = qa.get("passed")
            qa_lines.append(f"- {step_id}: {'pass' if passed else 'fail'}")
            for reason in qa.get("reasons", []):
                if isinstance(reason, dict):
                    mark = "ok" if reason.get("passed") else "no"
                    qa_lines.append(f"  - [{mark}] {reason.get('name')}: {reason.get('detail')}")
    lines.extend(qa_lines or ["- (no QA checks)"])
    lines.append("")

    lines.append("## Facts")
    view = facts_view(log, state.project_id)
    fact_lines = []
    for fact in view.current():
        source = fact.current.source or {}
        if source.get("recipe_node") != recipe_node:
            continue
        value = _canonical_value(fact.current.value)
        fact_lines.append(
            f"- {fact.key} = {value} (by {fact.current.by.kind}:{fact.current.by.id})"
        )
    lines.extend(fact_lines or ["- (no facts)"])
    lines.append("")

    lines.append("## Artifacts")
    seen: dict[str, str] = {}
    for step_id in _step_order(state):
        for op in sorted(state.attempts_for(step_id), key=lambda op: (op.round, op.attempt)):
            for aid in op.result_artifacts:
                seen.setdefault(aid, f"{step_id} attempt {op.attempt}")
    art_lines = [f"- {aid} (produced by {by})" for aid, by in sorted(seen.items())]
    lines.extend(art_lines or ["- (no artifacts)"])
    lines.append("")
    return "\n".join(lines)


def _canonical_value(value: object) -> str:
    import json

    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def report_step(ctx: StepContext) -> StepOutput:
    """The ``recipe.report`` cold callable: render and store ``REPORT.md``."""
    text = build_report_text(ctx.log, ctx.recipe_node)
    artifact_id = ctx.put(text.encode("utf-8"), mime=REPORT_MIME, meta={"schema": "aar.report/1"})
    return StepOutput(outputs={"report_sha256": artifact_id}, artifact_ids=(artifact_id,))
