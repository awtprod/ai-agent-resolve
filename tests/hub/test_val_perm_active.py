"""start/cancel/retry by another agent on an ACTIVE recipe (the creator check is the only guard)."""

import json
from typing import Any

from val_support import to_gate
from write_support import OTHER

from aar.hub.writes import WriteError
from aar.recipes.run import recipe_cancel, recipe_start
from aar.recipes.state import load_state


def test_other_agent_cannot_cancel_or_restart_an_active_recipe(
    hub: Any, registered_recipe: Any
) -> None:
    rn, _fake = to_gate(hub)
    before = [(o.step_id, o.status, o.superseded) for o in load_state(hub.log, rn).ops]
    codes = []
    for fn in (recipe_cancel, recipe_start):
        try:
            fn(hub.log, rn, OTHER, idempotency_key="o")
            codes.append("accepted")
        except WriteError as exc:
            codes.append(exc.code)
        except Exception as exc:
            codes.append(type(exc).__name__)
    after = [(o.step_id, o.status, o.superseded) for o in load_state(hub.log, rn).ops]
    print("\nOBS other agent on active recipe:", json.dumps([codes, before == after]))
    assert codes == ["not_creator", "not_creator"] and before == after
