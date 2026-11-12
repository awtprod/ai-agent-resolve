"""The ``cut`` step's instruction: the agent's whole briefing (the engine caps it at 2 KiB)."""

from __future__ import annotations

CUT_INSTRUCTION = (
    "Choose the cut. recipe_status shows each step's outputs. Fetch an artifact with "
    "`aar hub artifact get ID --db DB --artifacts DIR --out FILE`: stub.stub>stub.json, "
    "transcript.transcript>transcript.json, transcript.text>transcript.txt, "
    "sync.report>sync.json and laughs.map>laughs.json when set. transcript.txt is "
    "[start-end] words, in recording seconds. A '(no words transcribed)' line or a "
    "transcript.wordless stretch is unknown content (quiet speech, laughter), not silence. "
    "Rules: ranges chronological; never splice dialogue: start and end each range between "
    "sentences, whole thoughts only; end on the laugh. Write keep.json "
    '{"recording":R,"ranges":[{"start_s":S,"end_s":E,"label":L}]} with R = '
    "stub.recording_id; with more than one camera also schedule.json "
    '{"spans":[{"start_s":S,"end_s":E,"camera":"A"}]}. Run `aar plan produce stub.json '
    "--transcript transcript.json --keep keep.json --out plan.json F` with F = "
    "stub.pause_trim_flag (add --schedule, --sync sync.json, --laughs laughs.json when you "
    "have them), then `aar plan lint plan.json --transcript transcript.json`; change "
    "keep.json until both exit 0. Submit with plan_submit (or `aar hub write plan`): "
    '{"project_id":P,"derives_from":D,"plan":<plan.json>}, D = this recipe node in round 1, '
    "else the review that rejected the last cut (read its notes). Then step_complete (or "
    "`aar recipe complete --step cut`) with outputs "
    '{"plan_node":<id plan_submit returned>}. A plan with lint errors is refused: fix it and '
    "submit again. After this step the hub renders, QAs and exports the cut; a failed render "
    "or QA cannot come back here: cancel the run and start a new one (the transcript is "
    "reused)."
)
