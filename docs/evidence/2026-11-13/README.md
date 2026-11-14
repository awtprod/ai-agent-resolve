# Review card with handoff notes (S21)

A gate review in the console, opened by the recipe engine with the export step's handoff notes
attached. The notes sit under the video and beside (wide) or above (narrow) the Approve and
Reject buttons, on the queue and on the detail page.

- `review-card-handoff-notes.png`: the review queue at 1280 px.
- `review-card-handoff-notes-narrow.png`: the detail page at 420 px.

Made from a throwaway hub: a test-pattern video, and notes built by
`aar.recipes.ccn.handoff.handoff_notes` from the export report of the `ccn_2997` golden plan
(`tests/plan/fixtures/export`). No real show, media path or person appears in them.

# Lessons page (S23a)

The console's `/lessons` page, from a throwaway hub with synthetic names only (`Synthetic Show`,
`Synthetic Special`, `operator-1`, `agent-1`), built through `aar.hub.lessons` and
`aar.hub.reviews`. Two proposals wait, so the nav reads `Lessons (2)`; each card shows the why,
the source review (linked), who proposed it and when, and the form: the text (editable),
"Applies to" preselected to the suggestion, an optional reason, Accept and Drop. Below them, the
accepted lessons grouped Global, `Client: ccn` and `Project: Synthetic Show`, each with a retire
form, and the form to add one. The hub also holds a dropped proposal and a retired lesson; the
page does not show them.

- `lessons-page.png`: 1280 px wide.
- `lessons-page-narrow.png`: 390 px wide.
- `lessons-page-after-accept.png`: the same page after the first proposal was accepted with
  edited text and a wider scope (`Client: ccn`), submitted twice over HTTP: one acceptance, the
  flash, `Lessons (1)`, and the lesson marked "edited from the proposal".

Served by `aar hub console --operator operator-1` on a loopback port and captured with headless
Chrome.

# Lessons drafted by the proposer (S23b)

- `lessons-drafted.png`: `/lessons` at 1280 px after one live `aar hub lessons propose` pass
  (`gpt-6-luna` through the local model gateway) over a throwaway hub with synthetic names only
  (`Synthetic Show`, `Synthetic Special`, `operator-1`): two proposals by the agent
  `lessons-proposer`, each with the model's why, its source review and the suggested scope,
  waiting for a person to accept or drop them. The rejected review's instructions about that one
  cut were not drafted, and the accepted global lesson was not repeated. Retaken in S23b fix
  round 1 from the same hub (a copy of it), so that a project reads with its id, as S23a's fix
  made the console show it: "Project: Synthetic Special (prj_...)" in the suggestion and in
  "Applies to".
