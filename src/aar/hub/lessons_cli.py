"""``aar hub lessons list`` and ``show``: read the lessons from reviews (S23a, DESIGN R6).

Read-only on purpose. Accepting, dropping, retiring and adding a lesson are human-only, like
deciding a review, and stay in the console: this CLI's actors are self-declared, so a CLI write
path would let any caller claim to be a human and write instructions into future cut guides.
The database is opened read-only and never created.

Output is JSON on stdout (``list``: ``{"lessons": [...], "skipped": {count, events}}``; ``show``:
the lesson). A refusal prints ``{"error": {code, message, fix_hint, details}}`` on stderr and
exits 2; ``show`` of a well-formed id that names no lesson exits 1 (``unknown_lesson``).
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
