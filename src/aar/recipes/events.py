"""The ``recipe.*`` event family: the runner's own events, with typed payloads.

Every piece of recipe-run state lives in these events plus the graph (op nodes and their status)
and the reviews, so a killed runner resumes from the log alone. All payloads carry the op node id
so the fold can group them. They are written with S13g1's atomicity pattern (one transaction, a
``hub.write`` record last, idempotency keyed per actor); see :mod:`aar.recipes.engine`.
"""

from __future__ import annotations

from aar.hub.events import Actor

# The single, constant runner identity. Every structural write the engine makes (op nodes and
# edges, claims, heartbeats, hot intent/submitted, results, review requests and links, round
# rollbacks, cancels and the recipe status mirror) is attributed to RUNNER, with the real caller
# recorded alongside (see aar.recipes.engine). No tool surface can produce this actor, so the
# scheduler can trust that a recipe.* event or an aar.step/1 op node authored by RUNNER was
# written by the engine and not forged by an agent through the generic event API.
RUNNER = Actor("runner", "recipes")
RUNNER_KIND = "runner"

# Op data schema and node/edge conventions.
STEP_SCHEMA = "aar.step/1"

# recipe.* event types (the runner family). All are written with actor RUNNER; the real caller
# or worker is recorded in the payload (caller/worker/by) and in the hub.write record.
EVENT_STARTED = "recipe.started"
EVENT_CLAIMED = "recipe.claimed"
EVENT_HEARTBEAT = "recipe.heartbeat"
EVENT_HOT_INTENT = "recipe.hot_intent"
EVENT_HOT_SUBMITTED = "recipe.hot_submitted"
EVENT_RESULT = "recipe.result"
EVENT_REVIEW_LINKED = "recipe.review_linked"
EVENT_CANCELLED = "recipe.cancelled"
EVENT_RUN_CANCELLED = "recipe.run_cancelled"
EVENT_HOT_BAD_POLL = "recipe.hot_bad_poll"

RECIPE_EVENT_TYPES = (
    EVENT_STARTED,
    EVENT_CLAIMED,
    EVENT_HEARTBEAT,
    EVENT_HOT_INTENT,
    EVENT_HOT_SUBMITTED,
    EVENT_RESULT,
    EVENT_REVIEW_LINKED,
    EVENT_CANCELLED,
    EVENT_RUN_CANCELLED,
    EVENT_HOT_BAD_POLL,
)

# hub.write ops (the record appended last by each runner write).
OP_START = "recipe.start"
OP_CLAIM = "recipe.claim"
OP_HEARTBEAT = "recipe.heartbeat"
OP_HOT_INTENT = "recipe.hot_intent"
OP_HOT_SUBMITTED = "recipe.hot_submitted"
OP_HOT_BAD_POLL = "recipe.hot_bad_poll"
OP_RESULT = "recipe.result"
OP_REVIEW_OPEN = "recipe.review_open"
OP_STEP_COMPLETE = "recipe.step_complete"
OP_RETRY = "recipe.retry"
OP_CANCEL = "recipe.cancel"
OP_ADVANCE = "recipe.advance"

# Outcomes carried by recipe.result.
OUTCOME_DONE = "done"
OUTCOME_FAILED = "failed"
