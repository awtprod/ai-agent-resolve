"""The ``cut`` step's instruction: a pointer to the full guide (the engine caps it at 2 KiB)."""

from __future__ import annotations

CUT_INSTRUCTION = (
    "Choose the cut of this show, following the full guide the guide step wrote. Get it: "
    "recipe_status (MCP) shows the guide step's outputs; run its read_guide command, "
    "`aar hub artifact get <guide> --db <db> --artifacts <artifacts> --project <project> "
    "--out guide.md` (the outputs give every value), and follow guide.md step by step. In "
    "short: fetch stub.json, transcript.json and transcript.txt; keep ranges chronological; "
    "never splice dialogue (edges between sentences, from transcript.json word times; whole "
    "thoughts); end on the laugh; a '(no words transcribed)' line is unknown content, not "
    "silence. Write keep.json, run `aar plan produce` with stub.pause_trim_flag and `aar plan "
    "lint` until both exit 0, submit with plan_submit (derives_from = this recipe node in "
    "round 1, else the review that rejected the last cut: see the guide), then step_complete "
    'with {"plan_node": <id>}. `aar recipe ...` commands need AAR_RECIPES=aar.recipes.ccn.'
)
