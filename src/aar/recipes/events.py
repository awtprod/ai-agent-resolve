"""The ``recipe.*`` event family: the runner's own events, with typed payloads.

Every piece of recipe-run state lives in these events plus the graph (op nodes and their status)
and the reviews, so a killed runner resumes from the log alone. All payloads carry the op node id
so the fold can group them. They are written with S13g1's atomicity pattern (one transaction, a
``hub.write`` record last, idempotency keyed per actor); see :mod:`aar.recipes.engine`.
"""

from __future__ import annotations

# Op data schema and node/edge conventions.
STEP_SCHEMA = "aar.step/1"

# recipe.* event types (the runner family).
EVENT_CLAIMED = "recipe.claimed"
EVENT_HEARTBEAT = "recipe.heartbeat"
EVENT_HOT_INTENT = "recipe.hot_intent"
EVENT_HOT_SUBMITTED = "recipe.hot_submitted"
EVENT_RESULT = "recipe.result"
EVENT_REVIEW_LINKED = "recipe.review_linked"
EVENT_CANCELLED = "recipe.cancelled"

RECIPE_EVENT_TYPES = (
    EVENT_CLAIMED,
    EVENT_HEARTBEAT,
    EVENT_HOT_INTENT,
    EVENT_HOT_SUBMITTED,
    EVENT_RESULT,
    EVENT_REVIEW_LINKED,
    EVENT_CANCELLED,
)

# hub.write ops (the record appended last by each runner write).
OP_START = "recipe.start"
OP_CLAIM = "recipe.claim"
OP_HEARTBEAT = "recipe.heartbeat"
OP_HOT_INTENT = "recipe.hot_intent"
OP_HOT_SUBMITTED = "recipe.hot_submitted"
OP_RESULT = "recipe.result"
OP_REVIEW_OPEN = "recipe.review_open"
OP_STEP_COMPLETE = "recipe.step_complete"
OP_RETRY = "recipe.retry"
OP_CANCEL = "recipe.cancel"
OP_ADVANCE = "recipe.advance"

# Outcomes carried by recipe.result.
OUTCOME_DONE = "done"
OUTCOME_FAILED = "failed"
