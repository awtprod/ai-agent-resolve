"""The ``cut`` step's full agent guide, written by the ``guide`` cold step as an artifact.

The engine caps an agent step's instruction at 2 KiB, so the instruction only says how to fetch
this guide. The guide has no size pressure: it gives the exact commands with the hub database and
artifact store the worker itself was started with, the artifact ids of this run's inputs, the
environment ``aar recipe`` needs, the brief, how to find the review that rejected the last cut,
the house rules, and worked examples of ``keep.json`` and ``schedule.json``.
"""

from __future__ import annotations

import json
import shlex
from collections.abc import Sequence
from typing import Any

from aar.hub.writes import Brief

RECIPES_ENV = "AAR_RECIPES=aar.recipes.ccn"
#: The Python that writes submit.json from plan.json; its two arguments are the project id and
#: ``derives_from``.
MAKE_SUBMIT = (
    'import json,sys; json.dump({"project_id": sys.argv[1], "derives_from": sys.argv[2], '
    '"plan": json.load(open("plan.json"))}, open("submit.json", "w"))'
)


def _q(value: str) -> str:
    return shlex.quote(value)


def command(*words: str) -> str:
    """One shell line from ``words``, each quoted: it can be pasted into ``sh`` as is."""
    return shlex.join(words)


def artifact_get(
    aar: Sequence[str], artifact_id: str, out: str, *, db: str, artifacts: str, project_id: str
) -> str:
    """``aar hub artifact get`` of one artifact into ``out`` (replacing an earlier copy, so the
    same line works again in a later round)."""
    return command(
        *aar, "hub", "artifact", "get", artifact_id, "--db", db, "--artifacts", artifacts,
        "--project", project_id, "--out", out, "--force",
    )  # fmt: skip


def build_guide(
    *,
    db: str,
    artifacts: str,
    project_id: str,
    recipe_node: str,
    brief_id: str | None,
    brief: Brief | None,
    stub: dict[str, Any],
    sync: dict[str, Any],
    transcript: dict[str, Any],
    laughs: dict[str, Any],
    aar: Sequence[str],
    python: str,
) -> str:
    """The guide's Markdown text for one run (module docstring). ``aar`` is the argv that runs
    the worker's own ``aar`` and ``python`` its interpreter: every command the guide prints
    names them and quotes every word, so each line runs in ``sh`` exactly as printed."""
    where = {"db": db, "artifacts": artifacts, "project_id": project_id}
    fetch = [
        ("stub.json", stub["stub"], "the plan stub: sources, frame rate, timecodes"),
        ("transcript.json", transcript["transcript"], "every word with start_s and end_s"),
        ("transcript.txt", transcript["text"], "the same words as readable lines"),
    ]
    if sync.get("report"):
        fetch.append(("sync.json", sync["report"], "the camera offsets for --sync"))
    if laughs.get("map"):
        fetch.append(("laughs.json", laughs["map"], "the laugh map for --laughs"))
    cameras = int(stub.get("cameras", 1))
    produce_argv = [
        *aar, "plan", "produce", "stub.json", "--transcript", "transcript.json",
        "--keep", "keep.json", "--out", "plan.json", str(stub["pause_trim_flag"]),
    ]  # fmt: skip
    if cameras > 1:
        produce_argv += ["--schedule", "schedule.json"]
    if sync.get("report"):
        produce_argv += ["--sync", "sync.json"]
    if laughs.get("map"):
        produce_argv += ["--laughs", "laughs.json"]
    produce = command(*produce_argv)
    lint_argv = [*aar, "plan", "lint", "plan.json", "--transcript", "transcript.json"]
    if laughs.get("map"):
        lint_argv += ["--laughs", "laughs.json"]
    lint = command(*lint_argv)
    recipes = f"{RECIPES_ENV} {command(*aar)}"
    wordless = transcript.get("wordless") or {}
    stretches = wordless.get("stretches") or []
    labels = [s.get("camera") for s in stub.get("sources", [])]
    lines: list[str] = [
        "# Cut guide: ccn-longform",
        "",
        f"Recipe node `{recipe_node}`, project `{project_id}`, brief `{brief_id}`.",
        "Your job in step `cut`: choose what to keep of the show, make a plan that lints",
        "clean, submit it and complete the step. The hub then renders, QAs and exports it, and",
        "a human reviews the render.",
        "",
        "## The hub",
        "",
        f"- Database: `{db}`",
        f"- Artifact store: `{artifacts}`",
        f"- `aar recipe ...` commands need the recipe module: prefix them with `{RECIPES_ENV}`",
        "  (or pass `--recipes aar.recipes.ccn`).",
        f"- Status: `{recipes} recipe status {_q(recipe_node)} --db {_q(db)}`",
        "  (MCP: `recipe_status`).",
        "",
        "## The brief",
        "",
    ]
    if brief is not None:
        lines.append("```json")
        lines.append(json.dumps(brief.model_dump(mode="json", by_alias=True), indent=1))
        lines.append("```")
        lines.append("Follow its title, hook and notes when choosing what to keep.")
    else:
        lines.append("(no typed brief)")
    lines += [
        "",
        "## 1. Fetch the inputs",
        "",
        "Work in a folder of your own. Run:",
        "",
        "```sh",
    ]
    lines += [artifact_get(aar, aid, name, **where) for name, aid, _ in fetch]
    lines += ["```", ""]
    lines += [f"- `{name}`: {what}" for name, _, what in fetch]
    lines += [
        "",
        f"Recording `{stub['recording_id']}`, {stub['fps']} fps, cameras: "
        + ", ".join(f"{s.get('camera')} = `{s.get('id')}`" for s in stub.get("sources", [])),
        "",
        "## 2. Read the show",
        "",
        "`transcript.txt` has one `[start-end] words` line per spoken run, in recording",
        "seconds. A line can hold several sentences: to put a range edge between two",
        "sentences, look up the exact `start_s`/`end_s` of the words around it in",
        "`transcript.json` and put the edge in the gap between them, never inside a word.",
        "",
        "A `(no words transcribed)` line, and every wordless stretch below, is **unknown",
        "content, not silence**: quiet or far-field speech the transcriber missed, or the",
        "audience laughing. Do not assume it is empty; do not end a range inside one unless",
        "you mean to drop it.",
        "",
    ]
    if stretches:
        lines.append("Wordless stretches (recording seconds):")
        lines += [
            f"- {s.get('start_s')}-{s.get('end_s')} s ({s.get('duration_s')} s, "
            f"{s.get('level_dbfs')} dBFS)"
            for s in stretches
        ]
        lines.append("")
    lines += [
        "## 3. House rules (the ccn profile)",
        "",
        "- Keep ranges chronological and not overlapping.",
        "- Never splice dialogue: start and end every range between sentences, keeping whole",
        "  thoughts (a setup with its punchline).",
        "- End the piece on the laugh: the last range ends just after the laugh line.",
        "- With one camera every cut between ranges is a jump cut: prefer few, long ranges.",
        "",
        "## 4. Write keep.json",
        "",
        "```json",
        json.dumps(
            {
                "recording": stub["recording_id"],
                "ranges": [
                    {"start_s": 612.4, "end_s": 731.9, "label": "the parking story"},
                    {"start_s": 802.0, "end_s": 845.6, "label": "callback, ends on the laugh"},
                ],
            },
            indent=1,
        ),
        "```",
        "",
    ]
    if cameras > 1:
        lines += [
            "With more than one camera also write schedule.json: which camera is on screen,",
            f"in recording seconds (cameras: {', '.join(str(c) for c in labels)}):",
            "",
            "```json",
            json.dumps(
                {
                    "spans": [
                        {"start_s": 612.4, "end_s": 650.0, "camera": labels[0]},
                        {"start_s": 650.0, "end_s": 845.6, "camera": labels[1]},
                    ]
                },
                indent=1,
            ),
            "```",
            "",
        ]
    lines += [
        "## 5. Make and check the plan",
        "",
        "```sh",
        produce,
        lint,
        "```",
        "",
        "Both must exit 0. If produce or lint reports an error, change keep.json (or",
        "schedule.json) and run both again. Warnings are allowed; read them.",
        "",
        "## 6. Submit the plan and complete the step",
        "",
        "Make submit.json from plan.json; `D` is this recipe node in round 1 (in a later",
        "round, the review that rejected the last cut: see 7):",
        "",
        "```sh",
        command(python, "-c", MAKE_SUBMIT, project_id, recipe_node),
        "```",
        "",
        "Then submit it, and complete the step with the node id the write prints",
        "(`node.id`). Replace each `<...>` placeholder below with its value (keep the quotes):",
        "",
        "```sh",
        f"{command(*aar)} hub write plan --db {_q(db)} --artifacts {_q(artifacts)} "
        "--actor 'agent:<you>' --idempotency-key '<new key>' submit.json",
        'echo \'{"plan_node": "<node id the write printed>"}\' | '
        f"{recipes} recipe complete --db {_q(db)} --artifacts {_q(artifacts)} "
        "--actor 'agent:<you>' --idempotency-key '<new key>' "
        f"--recipe {_q(recipe_node)} --step cut",
        "```",
        "",
        "MCP: `plan_submit` with the same fields, then `step_complete` with",
        '`{"plan_node": ...}`. A plan with lint errors is refused: fix it and submit again.',
        "",
        "## 7. Later rounds (the human rejected the last cut)",
        "",
        "`recipe_status` then shows round 2 or later and step `cut` open again. Find the",
        "review that rejected the last cut and read why:",
        "",
        "```sh",
        command(*aar, "hub", "review", "list", "--db", db, "--project", project_id, "--all"),
        f"{command(*aar, 'hub', 'review', 'show', '--db', db, '--project', project_id)} "
        "'<review id>'",
        "```",
        "",
        "It is the review titled `gate review (round N)` for the previous round N whose",
        "subject is the plan you submitted then (`plan_get` shows that plan derives from",
        f"`{recipe_node}` or from an earlier review of it) and whose decision is `reject`; its",
        "`reason` and `notes` (at timeline frames) say what to change. MCP: `review_get`.",
        "Change the cut accordingly (a plan identical to a rejected one is refused), and",
        "make submit.json with that review's id as `derives_from` (the last argument of the",
        "python line in 6), not the recipe node, then submit and complete as in 6.",
        "",
        "## After the cut",
        "",
        "The hub renders the plan, gates its QA and exports the FCPXML and the review MP4.",
        "A render or QA failure cannot come back to this step: the run must be cancelled",
        "and a new one started (its transcript is reused, so that is cheap).",
        "",
    ]
    return "\n".join(lines)


__all__ = ["RECIPES_ENV", "artifact_get", "build_guide", "command"]
