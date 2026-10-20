"""``aar deliver review`` and ``aar deliver master``: carry a finished render to the hub.

The edit host comes from ``AAR_EDIT_*`` (``aar.host.env_config``), as ``aar host pull`` uses it.
Exit status: 0 done, 1 QA failed, 2 bad input or a missing tool, 3 a transport error.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from aar import cli_io

_PROG = "aar deliver"
EXIT_OK = 0
EXIT_QA_FAILED = 1  # QA failed (review refused, or a master whose QA failed)
EXIT_INPUT = 2  # bad arguments, a missing tool, or unusable configuration
EXIT_TRANSPORT = 3  # the verified transfer to or from the edit PC failed

# DeliverError codes that mean "QA failed" (exit 1); every other code is bad input (exit 2).
_QA_CODES = frozenset({"review_qa_failed"})


def register(subparsers: Any) -> None:
    """Add the ``deliver`` command group to the top-level parser."""
    deliver = subparsers.add_parser(
        "deliver",
        help="Deliver a finished render: review (under the cap, QA'd, sent) or master (hashed).",
        description=(
            "Pull a finished render from the edit PC and carry it to a stored, hashed, QA'd "
            "artifact. 'review' fits it under the profile cap and requests a human review; "
            "'master' stores and QA's it. The edit host is named by AAR_EDIT_* configuration. "
            "Exit: 0 done, 1 QA failed, 2 bad input or a missing tool, 3 transport error."
        ),
    )
    sub = deliver.add_subparsers(dest="deliver_command", metavar="COMMAND", required=True)

    review = sub.add_parser("review", help="Make the under-cap review MP4, QA it, request review.")
    _add_render_output_args(review)
    _add_common_args(review)
    review.add_argument("--subject", required=True, help="The work-graph node under review.")
    review.add_argument("--title", required=True, help="The review's title.")
    review.add_argument("--timeline", help="The timeline name shown in the review.")
    review.add_argument("--fps", type=float, help="The review's frame rate (for note timecodes).")
    review.set_defaults(func=_cmd_review)

    master = sub.add_parser("master", help="Store and QA the master render, recording its hash.")
    _add_render_output_args(master)
    _add_common_args(master)
    master.set_defaults(func=_cmd_master)


def _add_render_output_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--remote", help="Edit-PC path of the finished render.")
    parser.add_argument("--frame-count", type=int, help="The render's frame count.")
    parser.add_argument("--frame-rate", type=float, help="The render's frame rate.")
    parser.add_argument("--duration", type=float, help="The render's duration in seconds.")
    parser.add_argument("--render-job-id", help="The Resolve render job id of the render.")
    parser.add_argument(
        "--render-json",
        type=Path,
        help="A JSON file with the render job's measured dict, instead of the fields above.",
    )


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--deliverable", required=True, help="The deliverable name (a fact key).")
    parser.add_argument("--project", required=True, help="The hub project id.")
    parser.add_argument(
        "--kind", required=True, choices=("short", "longform"), help="The source render kind."
    )
    parser.add_argument("--db", required=True, help="The hub event-log database path.")
    parser.add_argument("--artifacts", required=True, help="The hub artifact store directory.")
    parser.add_argument(
        "--workdir", required=True, help="Absolute work directory (never a temp dir)."
    )
    parser.add_argument(
        "--profile", default="ccn", help="Profile name or .toml path (default: ccn)."
    )
    parser.add_argument(
        "--actor", required=True, metavar="KIND:ID", help="Who is delivering, e.g. agent:recipe-1."
    )
    parser.add_argument("--json", action="store_true", help="Print the result as JSON.")


# -- building the inputs ----------------------------------------------------------------------


def _parse_actor(value: str) -> Any:
    from aar.hub.events import Actor

    kind, sep, ident = value.partition(":")
    if not sep:
        raise ValueError(f"--actor must look like KIND:ID, got {value!r}")
    return Actor(kind, ident)


def _render_output(args: argparse.Namespace) -> Any:
    from aar.deliver.model import RenderOutput

    if args.render_json is not None:
        try:
            data = json.loads(Path(args.render_json).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"--render-json could not be read: {exc}") from exc
        return RenderOutput.from_measured(data)
    fields = {
        "--remote": args.remote,
        "--frame-count": args.frame_count,
        "--frame-rate": args.frame_rate,
        "--duration": args.duration,
        "--render-job-id": args.render_job_id,
    }
    missing = [name for name, value in fields.items() if value is None]
    if missing:
        raise ValueError(f"give --render-json FILE, or all of: {', '.join(missing)} (and the rest)")
    return RenderOutput.from_measured(
        {
            "output_path": args.remote,
            "frame_count": args.frame_count,
            "frame_rate": args.frame_rate,
            "duration_seconds": args.duration,
            "render_job_id": args.render_job_id,
        }
    )


def _edit_host() -> Any:
    from aar.host import env_config
    from aar.host.runner import SubprocessRunner
    from aar.host.transport import EditHost

    return EditHost(env_config(), SubprocessRunner())


def _hub(args: argparse.Namespace) -> tuple[Any, Any]:
    from aar.hub.artifacts import ArtifactStore
    from aar.hub.events import EventLog

    log = EventLog(args.db)
    store = ArtifactStore(args.artifacts, log)
    return log, store


# -- commands ---------------------------------------------------------------------------------


def _cmd_review(args: argparse.Namespace) -> int:
    def run() -> int:
        from aar.deliver.service import deliver_review
        from aar.plan.profile import load_profile

        actor = _parse_actor(args.actor)
        render_output = _render_output(args)
        profile = load_profile(args.profile)
        log, store = _hub(args)
        result = deliver_review(
            project_id=args.project,
            subject_id=args.subject,
            deliverable=args.deliverable,
            actor=actor,
            profile=profile,
            title=args.title,
            timeline=args.timeline,
            fps=args.fps,
            kind=args.kind,
            render_output=render_output,
            workdir=args.workdir,
            edit_host=_edit_host(),
            log=log,
            store=store,
        )
        return _emit(result, args.json)

    return _dispatch(args, run)


def _cmd_master(args: argparse.Namespace) -> int:
    def run() -> int:
        from aar.deliver.service import deliver_master
        from aar.plan.profile import load_profile

        actor = _parse_actor(args.actor)
        render_output = _render_output(args)
        profile = load_profile(args.profile)
        log, store = _hub(args)
        result = deliver_master(
            project_id=args.project,
            deliverable=args.deliverable,
            actor=actor,
            profile=profile,
            kind=args.kind,
            render_output=render_output,
            workdir=args.workdir,
            edit_host=_edit_host(),
            log=log,
            store=store,
        )
        # The master is stored whether or not QA passed; a failed QA is exit 1.
        return _emit(result, args.json, qa_gate=True)

    return _dispatch(args, run)


# -- output and error mapping -----------------------------------------------------------------


def _emit(result: Any, as_json: bool, *, qa_gate: bool = False) -> int:
    code = EXIT_QA_FAILED if qa_gate and result.qa_status == "fail" else EXIT_OK
    if as_json:
        text = json.dumps(result.to_dict(), indent=2) + "\n"
    else:
        extra = f" review {result.review_id}" if result.review_id else ""
        text = (
            f"{result.role} {result.deliverable}: {result.artifact_id} "
            f"({result.size} bytes, QA {result.qa_status}){extra}\n"
        )
    return cli_io.emit(_PROG, text) or code


def _error(
    code: str, message: str, fix_hint: str, detail: dict[str, Any], exit_code: int, as_json: bool
) -> int:
    if as_json:
        body = {
            "ok": False,
            "error": {"code": code, "message": message, "fix_hint": fix_hint, "detail": detail},
        }
        return cli_io.emit(_PROG, json.dumps(body, indent=2) + "\n") or exit_code
    cli_io.warn(f"error: {message}" + (f" ({fix_hint})" if fix_hint else ""))
    return exit_code


def _dispatch(args: argparse.Namespace, run: Any) -> int:
    from aar.deliver.errors import DeliverError
    from aar.host.errors import ConfigError, HostError, TransportError
    from aar.media import MediaToolMissing
    from aar.plan.profile import ProfileError

    as_json: bool = args.json
    try:
        return run()
    except DeliverError as exc:
        code = EXIT_QA_FAILED if exc.code in _QA_CODES else EXIT_INPUT
        return _error(exc.code, exc.message, exc.fix_hint, exc.detail, code, as_json)
    except MediaToolMissing as exc:
        return _error(
            "media_tool_missing",
            str(exc),
            "install ffmpeg/ffprobe or set AAR_FFMPEG",
            {},
            EXIT_INPUT,
            as_json,
        )
    except (ConfigError, ProfileError, ValueError) as exc:
        return _error(type(exc).__name__, str(exc), "", {}, EXIT_INPUT, as_json)
    except TransportError as exc:
        return _error(
            getattr(exc, "kind", "transport_error"), str(exc), "", {}, EXIT_TRANSPORT, as_json
        )
    except HostError as exc:  # any other typed edit-PC error
        return _error(getattr(exc, "kind", "host_error"), str(exc), "", {}, EXIT_TRANSPORT, as_json)
    except OSError as exc:
        return _error(type(exc).__name__, str(exc), "", {}, EXIT_INPUT, as_json)
