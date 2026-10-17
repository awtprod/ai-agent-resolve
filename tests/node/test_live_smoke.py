"""Read-only smoke test against a running Resolve Studio. Runs only with ``AAR_LIVE=1``.

It calls ``ScriptApiGateway.get_context()`` and nothing else: no switch, no save, no preset.
``DaVinciResolveScript`` must be importable (Resolve's ``Developer/Scripting/Modules`` on
``PYTHONPATH``).
"""

from __future__ import annotations

import pytest

from aar.node.gateway import VALID_PAGES, ResolveContext, ScriptApiGateway

pytestmark = pytest.mark.live


def test_get_context_reads_the_live_instance() -> None:
    import DaVinciResolveScript as dvr  # type: ignore[import-not-found]

    resolve = dvr.scriptapp("Resolve")
    assert resolve is not None, "scriptapp('Resolve') returned None: is Resolve Studio running?"

    ctx = ScriptApiGateway(resolve).get_context()
    print(ctx)

    assert isinstance(ctx, ResolveContext)
    assert ctx.page in VALID_PAGES
    assert ctx.project, "no project is open"
    assert ctx.project_id
    assert ctx.render_mode in (0, 1)
    assert ctx.render_format is None or isinstance(ctx.render_format, str)
    assert ctx.render_codec is None or isinstance(ctx.render_codec, str)
    assert ctx.folder is not None and ctx.folder.unique_id
    if ctx.timeline is not None:
        assert ctx.timeline_id
        assert ctx.playhead
