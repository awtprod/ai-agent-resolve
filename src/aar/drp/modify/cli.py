"""``aar drp edit`` and ``aar drp edit-schema``.

``edit`` always prints one JSON object (the change list, the refusals and the ``unverified``
label) and exits with the same codes as the other ``aar drp`` commands.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from aar.drp import cli as drp_cli
from aar.drp.errors import DrpError
from aar.drp.modify.engine import Refusal
from aar.drp.modify.plan import MAX_PLAN_BYTES, DrpEditPlan, PlanProblem, parse_plan
from aar.drp.modify.write import EditOutcome, edit_file

__all__ = [
    "EXIT_ERROR",
    "EXIT_OK",
    "EXIT_OUTPUT_ERROR",
    "EXIT_PIPE_CLOSED",
    "EXIT_REFUSED",
    "register",
]

EXIT_OK = 0
EXIT_REFUSED = 1  # like `diff --gate` failing: the plan or the written file did not pass
EXIT_ERROR = 2  # like `diff`: a file cannot be read, usage error, or an internal error
EXIT_OUTPUT_ERROR = drp_cli.EXIT_OUTPUT_ERROR
EXIT_PIPE_CLOSED = drp_cli.EXIT_PIPE_CLOSED

EXIT_STATUS_HELP = """\
exit status:
  0    edited and verified (written, or with --dry-run checked and not written)
  1    refused: the plan, the input or the written file failed a check; nothing was written
  2    error: the input or the plan file cannot be read, usage error, or an internal error
  74   the output could not be written: the --out file (then nothing was placed), or stdout
  141  the output pipe was closed early (for example by `head`)
With 74 or 141 from stdout, the edit may still have completed: the file and its label
<out>.aar.json are in place if the label exists; read it for the result.
Every written file is labelled "unverified" (see <out>.aar.json) until the Resolve sandbox
round trip exists.
"""


def register(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add ``edit`` and ``edit-schema`` to the ``aar drp`` subcommands."""
    fmt = argparse.RawDescriptionHelpFormatter
    p = sub.add_parser(
        "edit",
        help="apply a whitelisted edit plan to a copy of a .drp/.drt (prints JSON)",
        description=(
            "Apply a plan of whitelisted edits (rename, path_map, marker_add/update/delete) and "
            "write the result to a new file, atomically, after the S2 gate and an exact check of "
            "the change list. The input is never modified. Plan schema: aar drp edit-schema."
        ),
        epilog=EXIT_STATUS_HELP,
        formatter_class=fmt,
    )
    p.add_argument("file", help="the .drp or .drt to edit (read only)")
    p.add_argument("--plan", required=True, help="plan JSON file, or - for stdin")
    p.add_argument("--out", help="new file to write (required unless --dry-run)")
    p.add_argument("--dry-run", action="store_true", help="check everything, write nothing")
    p.add_argument("--force", action="store_true", help="replace --out if it exists")
    p.set_defaults(func=_edit)
    schema = sub.add_parser(
        "edit-schema", help="print the JSON Schema of the edit plan", formatter_class=fmt
    )
    schema.set_defaults(func=_schema)


def _schema(args: argparse.Namespace) -> int:
    text = json.dumps(DrpEditPlan.model_json_schema(), indent=2, ensure_ascii=True) + "\n"
    return drp_cli._emit(text)


def _read_plan(source: str) -> bytes:
    if source == "-":
        stream = sys.stdin.buffer if sys.stdin is not None else None
        if stream is None:
            raise OSError("stdin is closed")
        return stream.read(MAX_PLAN_BYTES + 1)
    with Path(source).open("rb") as fh:
        return fh.read(MAX_PLAN_BYTES + 1)


def _print(outcome: dict[str, object], code: int) -> int:
    text = json.dumps(outcome, indent=1, ensure_ascii=True, allow_nan=False) + "\n"
    written = drp_cli._emit(text)
    return written if written != EXIT_OK else code


def _bare(status: str, file: str, refusal: Refusal) -> dict[str, object]:
    return EditOutcome(status, {"path": file}, {}, refusals=[refusal]).to_dict()


def _edit(args: argparse.Namespace) -> int:
    try:
        raw = _read_plan(args.plan)
    except OSError as exc:
        drp_cli._warn(f"aar drp edit: cannot read the plan: {exc}")
        return _print(_bare("error", args.file, Refusal("plan-unreadable", str(exc))), EXIT_ERROR)
    try:
        plan = parse_plan(raw)
    except PlanProblem as exc:
        refusal = Refusal(exc.code, str(exc), details={"problems": exc.details})
        return _print(_bare("refused", args.file, refusal), EXIT_REFUSED)
    try:
        outcome = edit_file(args.file, plan, args.out, dry_run=args.dry_run, force=args.force)
    except DrpError as exc:
        drp_cli._warn(f"aar drp edit: {exc}")
        return _print(_bare("error", args.file, Refusal("input-unreadable", str(exc))), EXIT_ERROR)
    except OSError as exc:
        drp_cli._warn(f"aar drp edit: cannot write the output: {exc}")
        refusal = Refusal("output-error", f"cannot write the output: {exc}")
        return _print(_bare("error", args.file, refusal), EXIT_OUTPUT_ERROR)
    except Exception as exc:  # an internal error must never look like a refusal (1) or success
        drp_cli._warn(f"aar drp edit: internal error: {type(exc).__name__}: {exc}")
        drp_cli._warn(traceback.format_exc())
        refusal = Refusal("internal-error", f"{type(exc).__name__}: {exc}")
        return _print(_bare("error", args.file, refusal), EXIT_ERROR)
    return _print(outcome.to_dict(), EXIT_REFUSED if outcome.refused else EXIT_OK)
