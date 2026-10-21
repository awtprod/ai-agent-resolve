"""A crash-safety journal an op can write to while it runs.

Some ops have side effects in Resolve that must survive a node crash: ``render.build`` adds a
render job whose Resolve ``JobId`` is the only way to find it again. If the node dies after
``AddRenderJob`` but before the job runner checkpointed the op's result, a resumed job must find
that render job, not add a second one. So the op records each side effect *as it happens*, on
the node's durable job store, through a journal the job runner installs around the op body.

``journal(event, data)`` persists ``data`` for the op's job and returns whether the job still
wants the op's work: ``False`` once the job was cancelled or failed (or the runner closed), so
an op whose result would land on a dead job undoes its own side effect instead of leaving an
orphan behind. The journal is per op run, set in a :class:`~contextvars.ContextVar` by the
runner's op wrapper on the thread that runs the op; an op run outside a job sees ``None``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

Journal = Callable[[str, dict[str, Any]], bool]
"""``journal(event, data) -> still_wanted``. ``data`` must be JSON-serialisable."""

_JOURNAL: ContextVar[Journal | None] = ContextVar("aar_op_journal", default=None)


def current_journal() -> Journal | None:
    """The journal of the op running on this thread, or ``None`` outside a job."""
    return _JOURNAL.get()


@contextmanager
def op_journal(journal: Journal) -> Iterator[None]:
    """Install ``journal`` for the op body run inside this block."""
    token = _JOURNAL.set(journal)
    try:
        yield
    finally:
        _JOURNAL.reset(token)
