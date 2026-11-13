"""``aar hub lessons list``, ``show`` and ``propose``: the lessons from reviews (S23, DESIGN R6).

``list`` and ``show`` only read: the database is opened read-only and never created.
Accepting, dropping, retiring and adding a lesson are human-only, like deciding a review, and
stay in the console: this CLI's actors are self-declared, so a CLI write path for them would let
any caller claim to be a human and write instructions into future cut guides.

``propose`` (S23b) is one bounded pass of :mod:`aar.hub.lessons_proposer`: a model drafts lessons
from decided reviews and they are recorded as proposals, by an agent only (``--actor``, default
``agent:lessons-proposer``). A proposal applies to nothing until a person accepts it. The
gateway is ``AAR_LLM_BASE_URL`` (default ``http://127.0.0.1:8317``) with the key in
``AAR_LLM_API_KEY`` (there is no flag for the key); the model is ``--model``, else
``AAR_LESSONS_MODEL``, else ``gpt-6-luna``. The database must exist.

Output is JSON on stdout (``list``: ``{"lessons": [...], "skipped": {count, events}}``; ``show``:
the lesson; ``propose``: the pass's report). A refusal prints ``{"error": {code, message,
fix_hint, details}}`` on stderr and exits 2; ``show`` of a well-formed id that names no lesson
exits 1 (``unknown_lesson``). ``propose`` exits 0 when the pass ran (with nothing to do, with
another pass holding the lock (``busy``), or stopped by a full backlog, too); 1 when it failed:
a setup failure stopped it (the gateway, its key, a model it does not serve, the lock file),
ambiguous failures in a row stopped it, or it recorded a strike (the report on stdout, the
error on stderr); and 2 when it could not start: no key (``not_configured``), a refused gateway
URL (``origin``), a bad argument or a missing database.
"""

from __future__ import annotations

import argparse

from aar.hub.write_cli import _error, _print

STATE_CHOICES = ("proposed", "accepted", "dropped", "retired", "all")


def register(hub_sub: argparse._SubParsersAction) -> None:
    """Add ``lessons`` to the ``aar hub`` group."""
    lessons = hub_sub.add_parser(
        "lessons",
        help="Read the lessons people accepted from reviews (read-only).",
        description=(
            "Lessons are one-line editing rules that the ccn cut guide prints. Agents propose "
            "them; a person accepts, drops, retires or adds them in the console. This command "
            "only reads them."
        ),
    )
    lessons_sub = lessons.add_subparsers(dest="lessons_command", required=True)

    listing = lessons_sub.add_parser("list", help="List lessons as JSON (oldest first).")
    listing.add_argument("--db", required=True, help="Path to the SQLite event-log database.")
    listing.add_argument(
        "--state",
        choices=STATE_CHOICES,
        default="all",
        help="Only lessons in this state (default: all).",
    )
    listing.add_argument(
        "--scope",
        help="Only lessons that apply to this scope: global, client:<name> or project:<id>.",
    )
    listing.set_defaults(func=_list)

    show = lessons_sub.add_parser("show", help="Show one lesson as JSON.")
    show.add_argument("--db", required=True, help="Path to the SQLite event-log database.")
    show.add_argument("lesson_id", help="Lesson id (les_...).")
    show.set_defaults(func=_show)

    propose = lessons_sub.add_parser(
        "propose",
        help="Draft lessons from decided reviews with a model (one bounded pass).",
        description=(
            "Ask a model, through the local model gateway, for at most 3 lessons from each "
            "decided review not drafted yet, oldest decision first, and record them as "
            "proposals a person accepts or drops in the console. The gateway is "
            "AAR_LLM_BASE_URL (default http://127.0.0.1:8317) and its key AAR_LLM_API_KEY. "
            "Exit 0 when the pass ran (also with nothing to do, another pass running, or a "
            "full backlog), 1 when it failed: a setup failure or failures in a row stopped "
            "it, or a review's failure could not be charged to it (report on stdout, error on "
            "stderr), 2 when it could not start (JSON error on stderr)."
        ),
    )
    propose.add_argument("--db", required=True, help="Path to the SQLite event-log database.")
    propose.add_argument(
        "--limit",
        type=int,
        help="At most this many model calls in the pass (1-50, default 5).",
    )
    propose.add_argument(
        "--model",
        help="The model id (default: AAR_LESSONS_MODEL, else gpt-6-luna).",
    )
    propose.add_argument(
        "--actor",
        metavar="agent:ID",
        help="The agent the proposals are recorded as (default agent:lessons-proposer).",
    )
    propose.set_defaults(func=_propose)


def _list(args: argparse.Namespace) -> int:
    from aar.hub.cli import _DatabaseUnavailable, _open_existing
    from aar.hub.common import skipped_summary
    from aar.hub.lessons import parse_scope, scan_lessons
    from aar.hub.writes import WriteError

    try:
        scope = None if args.scope is None else str(parse_scope(args.scope))
    except WriteError as exc:
        return _error(exc.code, exc.message, exc.fix_hint, details=exc.details)
    try:
        log = _open_existing(args.db, read_only=True)
    except _DatabaseUnavailable as exc:
        return _error("database_unavailable", str(exc), "Pass --db an existing hub database.")
    try:
        found, skipped = scan_lessons(log)
    finally:
        log.close()
    rows = [
        lesson.to_dict()
        for lesson in found
        if args.state in ("all", lesson.state) and scope in (None, lesson.scope)
    ]
    return _print({"lessons": rows, "skipped": skipped_summary(skipped)})


def _show(args: argparse.Namespace) -> int:
    from aar.hub.cli import _DatabaseUnavailable, _open_existing
    from aar.hub.lessons import check_lesson_id, get_lesson
    from aar.hub.writes import WriteError

    try:
        lesson_id = check_lesson_id(args.lesson_id)
    except WriteError as exc:
        return _error(exc.code, exc.message, exc.fix_hint, details=exc.details)
    try:
        log = _open_existing(args.db, read_only=True)
    except _DatabaseUnavailable as exc:
        return _error("database_unavailable", str(exc), "Pass --db an existing hub database.")
    try:
        lesson = get_lesson(log, lesson_id)
    finally:
        log.close()
    if lesson is None:
        return _error(
            "unknown_lesson",
            f"no lesson {lesson_id!r} in the hub",
            "List the lessons with `aar hub lessons list --db DB`.",
            exit_status=1,
            details={"lesson_id": lesson_id},
        )
    return _print(lesson.to_dict())


def _propose(args: argparse.Namespace) -> int:
    import os

    from aar.hub import lessons_proposer as proposer
    from aar.hub.cli import _DatabaseUnavailable, _open_existing
    from aar.hub.events import parse_actor_text
    from aar.hub.writes import WriteError
    from aar.llm.gateway import GatewayError, gateway_from_env

    try:
        limit = proposer.check_limit(proposer.DEFAULT_LIMIT if args.limit is None else args.limit)
        model = proposer.lessons_model(args.model, os.environ)
    except ValueError as exc:
        return _error("invalid_input", str(exc), "See `aar hub lessons propose --help`.")
    try:
        actor = proposer.DEFAULT_ACTOR if args.actor is None else parse_actor_text(args.actor)
    except ValueError as exc:  # ActorNotAllowed too: a malformed or reserved actor
        return _error("invalid_input", str(exc), "Use --actor agent:ID (agent:lessons-proposer).")
    try:
        actor = proposer.check_proposer(actor)  # an agent with a valid id, nothing else
    except WriteError as exc:
        return _error(exc.code, exc.message, exc.fix_hint, details=exc.details)
    try:
        client = gateway_from_env(os.environ)
    except GatewayError as exc:
        return _error(exc.code, exc.message, exc.hint)
    try:
        log = _open_existing(args.db)
    except _DatabaseUnavailable as exc:
        return _error("database_unavailable", str(exc), "Pass --db an existing hub database.")
    try:
        report = proposer.propose_from_reviews(log, client, model=model, actor=actor, limit=limit)
    finally:
        log.close()
    written = _print(report.to_dict())
    failure = report.error()
    if written != 0 or failure is None:
        return written
    return _error(
        failure["code"],
        failure["message"],
        failure["hint"],
        exit_status=1,
        details={"reason": failure["reason"]},
    )
