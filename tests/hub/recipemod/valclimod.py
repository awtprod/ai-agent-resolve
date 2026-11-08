"""A recipe module for the AAR_RECIPES CLI probe (registered on import)."""

from pydantic import BaseModel, ConfigDict

from aar.recipes import registry
from aar.recipes.context import StepOutput
from aar.recipes.models import RecipeDef, StepDef


class ValCliParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    n: int = 1


def _c(ctx):
    return StepOutput(outputs={"ok": True})


registry.register_callable("valcli.c", _c)
registry.register(
    RecipeDef(
        name="valcli",
        version=1,
        params_model=ValCliParams,
        steps=(StepDef(id="c", kind="cold", callable_id="valcli.c"),),
    )
)
