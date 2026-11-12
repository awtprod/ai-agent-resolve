"""The ``cut`` step's instruction: a pointer to the full guide (the engine caps it at 2 KiB)."""

from __future__ import annotations

CUT_INSTRUCTION = (
    "Choose the cut of this show, following the full guide the guide step wrote. Get it: "
    "recipe_status (MCP) shows the guide step's outputs; run its read_guide command in a shell "
    "exactly as printed (a full `hub artifact get ... --out guide.md` line naming the worker's "
    "own aar, which is not on PATH, with every path quoted), then follow guide.md step by step, "
    "running its commands as printed too. In short: fetch stub.json, transcript.json and "
    "transcript.txt; keep ranges chronological; never splice dialogue (edges between "
    "sentences, from transcript.json word times; whole thoughts); end on the laugh; a '(no "
    "words transcribed)' line is unknown content, not silence. Write keep.json, run the "
    "guide's `plan produce` (with stub.pause_trim_flag) and `plan lint` lines until both exit "
    "0, submit with plan_submit (derives_from = this recipe node in round 1, else the review "
    "that rejected the last cut: see the guide), then step_complete with "
    '{"plan_node": <id>}. `recipe ...` commands need AAR_RECIPES=aar.recipes.ccn.'
)
