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
from typing import Any

from aar.hub.writes import Brief

RECIPES_ENV = "AAR_RECIPES=aar.recipes.ccn"


def _q(value: str) -> str:
    return shlex.quote(value)


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
) -> str:
    """The guide's Markdown text for one run (module docstring)."""
    get = (
        f"aar hub artifact get {{id}} --db {_q(db)} --artifacts {_q(artifacts)} "
        f"--project {_q(project_id)} --out {{file}}"
    )
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
    produce = (
        "aar plan produce stub.json --transcript transcript.json --keep keep.json "
        f"--out plan.json {stub['pause_trim_flag']}"
    )
    if cameras > 1:
        produce += " --schedule schedule.json"
    if sync.get("report"):
        produce += " --sync sync.json"
    if laughs.get("map"):
        produce += " --laughs laughs.json"
    lint = "aar plan lint plan.json --transcript transcript.json"
    if laughs.get("map"):
        lint += " --laughs laughs.json"
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
        f"- Status: `{RECIPES_ENV} aar recipe status {recipe_node} --db {_q(db)}`",
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
    lines += [get.format(id=aid, file=name) for name, aid, _ in fetch]
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
        "Write submit.json, with `D` = this recipe node in round 1:",
        "",
        "```json",
        json.dumps(
            {"project_id": project_id, "derives_from": recipe_node, "plan": "<plan.json here>"},
            indent=1,
        ),
        "```",
        "",
        "```sh",
        f"aar hub write plan --db {_q(db)} --artifacts {_q(artifacts)} --actor agent:<you> "
        "--idempotency-key <new key> submit.json",
        'echo \'{"plan_node": "<node id the write printed>"}\' | '
        f"{RECIPES_ENV} aar recipe complete --db {_q(db)} --artifacts {_q(artifacts)} "
        f"--actor agent:<you> --idempotency-key <new key> --recipe {recipe_node} --step cut",
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
        f"aar hub review list --db {_q(db)} --project {project_id} --all",
        f"aar hub review show --db {_q(db)} --project {project_id} <review id>",
        "```",
        "",
        "It is the review titled `gate review (round N)` for the previous round N whose",
        "subject is the plan you submitted then (`plan_get` shows that plan derives from",
        f"`{recipe_node}` or from an earlier review of it) and whose decision is `reject`; its",
        "`reason` and `notes` (at timeline frames) say what to change. MCP: `review_get`.",
        "Change the cut accordingly (a plan identical to a rejected one is refused), and",
        "submit it with `derives_from` = that review's id, not the recipe node.",
        "",
        "## After the cut",
        "",
        "The hub renders the plan, gates its QA and exports the FCPXML and the review MP4.",
        "A render or QA failure cannot come back to this step: the run must be cancelled",
        "and a new one started (its transcript is reused, so that is cheap).",
        "",
    ]
    return "\n".join(lines)


__all__ = ["RECIPES_ENV", "build_guide"]
