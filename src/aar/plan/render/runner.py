"""One ffmpeg call of a render: in the work dir, with a timeout, failures typed (S18b)."""

from __future__ import annotations

from pathlib import Path

from aar.media import MediaError, MediaTimeout, MediaToolMissing, run_tool
from aar.plan.render.errors import RenderError

BASE_ARGS = ("-hide_banner", "-nostdin", "-nostats")
#: Enough stderr for loudnorm's JSON and an error's context; run_tool keeps only this tail.
STDERR_LIMIT = 1024 * 1024


def run_ffmpeg(args: list[str], stage: str, timeout_s: float, cwd: Path) -> str:
    """Run ffmpeg with ``args`` in ``cwd`` (the work dir); its stderr, or a ``render_failed`` /
    ``render_timeout`` / ``tool_missing`` refusal naming the stage and the stderr tail."""
    try:
        return run_tool(
            "ffmpeg", [*BASE_ARGS, *args], timeout_s=timeout_s, cwd=cwd, stderr_limit=STDERR_LIMIT
        ).stderr
    except MediaToolMissing as exc:
        raise RenderError(
            "tool_missing", str(exc), "install ffmpeg or set AAR_FFMPEG", stage=stage
        ) from None
    except MediaTimeout as exc:
        raise RenderError(
            "render_timeout",
            f"ffmpeg ran past {exc.timeout_s:g} s in the {stage} pass and was stopped",
            "raise [render] timeout_per_media_s or timeout_base_s for this machine",
            stage=stage,
            timeout_s=exc.timeout_s,
        ) from None
    except MediaError as exc:
        raise RenderError(
            "render_failed",
            f"ffmpeg failed in the {stage} pass: {exc.summary()}",
            "check the hub media and the plan; the details carry ffmpeg's last lines",
            stage=stage,
            exit_code=exc.exit_code,
            stderr_tail=exc.stderr_tail[-800:],
        ) from None
