"""The node op library: verified hot ops with the known Resolve traps built in.

Importing this package registers every op. Run them only through the broker::

    op_id = submit_op(broker, "timeline.describe", {}, target=locator, holder=holder)
    ...  # broker.tick() runs it
    result = broker.result(op_id).value  # a DescribeResult

See ``framework`` for the contract every op follows.
"""

from __future__ import annotations

from aar.node.ops import read, write
from aar.node.ops.framework import (
    ItemRef,
    MediaRef,
    Note,
    OpEnv,
    OpError,
    OpModel,
    OpSpec,
    TimelineIdentity,
    TimelineRef,
    execute,
    get_op,
    list_ops,
    make_hot_op,
    submit_op,
)

__all__ = [
    "ItemRef",
    "MediaRef",
    "Note",
    "OpEnv",
    "OpError",
    "OpModel",
    "OpSpec",
    "TimelineIdentity",
    "TimelineRef",
    "execute",
    "get_op",
    "list_ops",
    "make_hot_op",
    "read",
    "submit_op",
    "write",
]
