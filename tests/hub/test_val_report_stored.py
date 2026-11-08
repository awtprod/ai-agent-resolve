"""The stored REPORT.md vs the final state."""

from typing import Any

from val_support import approve, complete_plan, start_run, worker_idle

from aar.recipes.report import build_report_text
from aar.recipes.state import load_state


def test_stored_report_header(hub: Any, registered_recipe: Any) -> None:
    rn, fake = start_run(hub)
    worker_idle(hub, fake)
    complete_plan(hub, rn, fake, "r1")
    approve(hub, rn, fake)
    worker_idle(hub, fake)
    st = load_state(hub.log, rn)
    stored = hub.store.read_bytes(st.done_op("report").result_artifacts[0]).decode()
    print("\nOBS stored header:", stored.splitlines()[:5])
    print("OBS final header:", build_report_text(hub.log, rn).splitlines()[:5])
    report_lines = [ln for ln in stored.splitlines() if "report" in ln][:4]
    print("OBS stored lists report step as:", report_lines)
