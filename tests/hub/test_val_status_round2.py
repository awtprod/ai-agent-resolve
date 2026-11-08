"""Status view in round 2 for steps done in round 1 (carried forward)."""

import json
from typing import Any

from val_support import complete_plan, reject, start_run, step_states, worker_idle

from aar.recipes.state import load_state
from aar.recipes.status import status_view


def test_round2_status_of_carried_steps(hub: Any, registered_recipe: Any) -> None:
    rn, fake = start_run(hub)
    worker_idle(hub, fake)
    complete_plan(hub, rn, fake, "r1")
    reject(hub, rn, fake)
    st = load_state(hub.log, rn)
    v = status_view(hub.log, rn, dispatcher=fake)
    c = next(s for s in v["steps"] if s["step_id"] == "c")
    print(
        "\nOBS round2:",
        json.dumps(
            {
                "round": st.current_round(),
                "c_done": st.is_step_done("c"),
                "h_done": st.is_step_done("h"),
                "states": step_states(v),
                "c_entry": c,
            }
        ),
    )
    assert step_states(v)["c"] == "done", "a step done in round 1 is reported 'ready' in round 2"
