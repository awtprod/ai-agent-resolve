"""The handoff notes: what the person importing the timeline on the edit PC has to know.

The review MP4 is a preview rendered on the hub; the FCPXML is the timeline. They differ, and
the export's own report (which says how) stays in the hub's store. :func:`handoff_notes` puts
the part a person acts on into one plain-text file that travels with the timeline (the export
step copies it beside the FCPXML) and is shown on the review card, so nobody approves the video
without seeing what still has to be done in Resolve:

* how to import, and the edit-PC media paths the timeline links. The hub never sees those
  files, so the notes say they are unchecked and what a wrong one looks like;
* the sound: the review render brings its mix to the loudness target with one gain and a
  limiter (:mod:`aar.plan.render.loudness`); the FCPXML carries the plan's own clip levels
  only. The gain is not written into the clips: without the limiter it would clip the peaks
  the review tamed, and the mastering belongs to the edit PC;
* every plan feature the FCPXML cannot carry (captions, grades and LUTs, bus routing, the
  template, markers, ...) with where it is in the plan and what to do about it, and the
  export's notes.

The text is the same for the same plan, report and render (no clock, no host names), so a
repeated export copies nothing twice. Every value that comes from the plan is flattened to one
line: a marker name cannot forge a heading.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

#: Entries listed per section; the rest are counted (the full report is on the hub).
MAX_LISTED = 40
#: Characters kept of one value from the plan or the report.
MAX_VALUE = 400
NOTES_SUFFIX = ".notes.txt"
NOTES_MIME = "text/plain"


def _line(value: Any) -> str:
    """``value`` as one bounded line of printable text."""
    text = "".join(c if c.isprintable() else " " for c in str(value))
    text = " ".join(text.split())
    return text if len(text) <= MAX_VALUE else text[: MAX_VALUE - 3] + "..."


def _item(item: Mapping[str, Any]) -> str:
    """A report item as the export's own text gives it: ``location: detail``."""
    where = _line(item.get("location") or "")
    return f"{where}: {_line(item.get('detail', ''))}" if where else _line(item.get("detail", ""))


def _rows(value: Any) -> list[Mapping[str, Any]]:
    return [row for row in value if isinstance(row, Mapping)] if isinstance(value, list) else []


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _section(title: str, lines: Sequence[str], total: int, report_id: str) -> list[str]:
    out = ["", f"{title} ({total})" if total else title, *lines]
    if total > MAX_LISTED:
        out.append(
            f"  ... and {total - MAX_LISTED} more: see the export report on the hub "
            f"(artifact {report_id})"
        )
    return out


def _sound(loudness: Mapping[str, Any]) -> list[str]:
    """What the review video's sound has that the timeline does not."""
    target = _number(loudness.get("target_lufs"))
    gain = _number(loudness.get("gain_db"))
    reduction = _number(loudness.get("limiter_reduction_db"))
    goal = f"{target:g} LUFS" if target is not None else "the loudness target"
    todo = (
        f"  Still to do in Resolve: bring the mix to {goal} and limit its peaks (Fairlight or "
        "the Deliver page)."
    )
    if gain is None:
        return [
            "  The review video's mix was not levelled (it was silent, or the render did not "
            "report a gain); the timeline plays the clips at the plan's own levels.",
            todo,
        ]
    moved = "left at its own level" if gain == 0 else f"{'raised' if gain > 0 else 'lowered'} by"
    amount = "" if gain == 0 else f" {abs(gain):g} dB"
    limited = (
        f"the limiter took {reduction:g} dB off its peaks"
        if reduction is not None and reduction > 0
        else "the limiter did not have to act"
    )
    differs = (
        "at the same level as the review video"
        if gain == 0
        else f"about {abs(gain):g} dB {'quieter' if gain > 0 else 'louder'} than the review video"
    )
    return [
        f"  The review video's mix was {moved}{amount} to reach {goal}, and {limited}.",
        "  The timeline has neither the gain nor the limiter: it plays the clips at the plan's "
        f"own levels, {differs}, with unlimited peaks.",
        todo,
    ]


def _quoted(value: Any) -> str:
    return f'"{_line(value)}"'


def _library(record: Mapping[str, Any]) -> list[str]:
    """The library hand-off (S25d), in the owner's terms: which library and project to open,
    and that saving is all; or why the cut is not in the library."""
    label = _quoted(record.get("label") or record.get("library") or "the project library")
    if record.get("status") != "placed":
        error = record.get("error") if isinstance(record.get("error"), Mapping) else {}
        return [
            "",
            "LIBRARY PROJECT: NOT WRITTEN",
            f"  The cut was not written into the project library {label}:",
            f"  {_line(error.get('message', ''))} [{_line(error.get('code', ''))}]",
            f"  To fix: {_line(error.get('hint', ''))}",
            "  Nothing is lost: import the FCPXML beside this file as usual (IMPORT above).",
        ]
    user = record.get("user")
    lines = [
        "",
        "OR OPEN IT AS A LIBRARY PROJECT (NEW ROUTE, STILL BEING CONFIRMED)",
        f"  The cut is also in the shared project library {label} as the project "
        f"{_quoted(record.get('project_name'))}.",
        "  1. In Resolve open the Project Manager and choose the project library "
        f"{label}" + (f" (user {_quoted(user)})" if user else "") + ".",
        f"  2. Open the project {_quoted(record.get('project_name'))}; its timeline is "
        f"{_quoted(record.get('timeline'))}.",
        "  3. Edit, then save the project (File > Save Project, or Ctrl+S). Saving is all that",
        "     is needed: the hub notices each save while this review is open and records it for",
        "     review. There is nothing to export and no .drt to make for it.",
        "  Until library projects have been confirmed in Resolve, the FCPXML stays the dependable",
        "  route: if the project does not open as expected, import the FCPXML instead and tell",
        "  the agent what you saw.",
    ]
    ops = [_line(op) for op in record.get("operations_not_opened") or []]
    if ops:
        lines.append(
            "  Not yet seen working in Resolve (check these first): " + ", ".join(ops[:12]) + "."
        )
    unverified = [_line(p) for p in record.get("unverified_media") or []]
    if unverified:
        lines.append(
            "  Clips written without full evidence for their kind (check they play and link): "
            + ", ".join(unverified[:MAX_LISTED])
            + "."
        )
    if record.get("not_carried"):
        lines.append(
            f"  The library project leaves out {int(record['not_carried'])} plan detail(s) (gain, "
            "framing, labels, notes, markers, captions): redo them in Resolve as for the FCPXML."
        )
    if record.get("pickup"):
        lines.append(f"  The hub will NOT pick up saves of it: {_line(record['pickup'])}.")
    return lines


def handoff_notes(
    *,
    project: str,
    timeline: str,
    round_: int,
    report: Mapping[str, Any],
    report_id: str,
    loudness: Mapping[str, Any] | None,
    library: Mapping[str, Any] | None = None,
) -> str:
    """The notes for one exported round, from the export's ``report`` (stored as artifact
    ``report_id``) and the render step's ``loudness`` output; with ``library``, the library
    hand-off's record (S25d). CRLF lines: it is read on Windows."""
    lines = [
        f"HANDOFF NOTES: {_line(timeline)}, round {round_} (project {_line(project)})",
        "",
        "Beside this file: the .fcpxml (the timeline to import) and the .review.mp4 (the video "
        "that was reviewed).",
        "The review video is a preview rendered on the hub, not the timeline. Everything below "
        "differs between the two, or has to be done in Resolve after the import.",
    ]

    # First what the reviewed video does not show (read at the approval), then the import.
    lines += _section("SOUND", _sound(loudness or {}), 0, "")

    missing = _rows(report.get("not_expressible"))
    listed: list[str] = []
    for item in missing[:MAX_LISTED]:
        listed.append(f"  - {_item(item)}")
        if item.get("action"):
            listed.append(f"      do: {_line(item['action'])}")
    if not missing:
        listed.append("  Nothing: the FCPXML carries every feature this plan uses.")
    lines += _section("NOT IN THE FCPXML: REDO IN RESOLVE", listed, len(missing), report_id)

    fusion_raw = report.get("fusion")
    fusion_placements = (
        _rows(fusion_raw.get("placements")) if isinstance(fusion_raw, Mapping) else []
    )
    if fusion_placements:
        flines: list[str] = [
            "  Beside this file: a .fusion.json manifest lists every Fusion graphic placement.",
            "  Use 'aar fusion customize' or 'aar fusion render' on the hub, or apply the",
            "  templates in Resolve's Fusion page matching the track and times below.",
        ]
        for p in fusion_placements[:MAX_LISTED]:
            tpl = _line(p.get("setting") or p.get("template") or "Fusion graphic")
            track = p.get("track", "?")
            t_in = p.get("record_in_s", 0.0)
            t_out = p.get("record_out_s", 0.0)
            kind = _line(p.get("kind") or "graphic")
            flines.append(
                f"  - [{p.get('index', '?')}] {kind} on V{track} ({t_in}s..{t_out}s): {tpl}"
            )
            inputs = p.get("inputs")
            if isinstance(inputs, Mapping) and inputs:
                pairs = ", ".join(f"{_line(k)}={_line(v)}" for k, v in inputs.items())
                flines.append(f"      inputs: {pairs}")
            elif isinstance(p.get("lines"), list) and p["lines"]:
                flines.append(f"      lines: {_line(' | '.join(str(x) for x in p['lines']))}")
        lines += _section("FUSION GRAPHICS", flines, len(fusion_placements), report_id)

    media = _rows(report.get("media"))
    paths = sorted({_line(m.get("windows_path") or m.get("src") or "?") for m in media})
    lines += _section(
        "MEDIA ON THE EDIT PC",
        [
            (
                "  PC source and hub copy verified at ingestion (receipts checked against the "
                "hub bytes and mapped PC paths). This is a snapshot: if you moved or changed "
                "the PC files afterwards, regenerate their receipts. After import check for "
                "Media Offline."
                if report.get("media_evidence", {}).get("status") == "verified_snapshot"
                else "  The timeline links these files; they are NOT checked. "
                "Verify receipts before import."
            ),
            *[f"  - {path}" for path in paths[:MAX_LISTED]],
        ],
        len(paths),
        report_id,
    )

    steps = [s for s in report.get("import") or [] if isinstance(s, str)]
    lines += _section("IMPORT", [f"  {n}. {_line(s)}" for n, s in enumerate(steps, start=1)], 0, "")

    if library is not None:
        lines += _library(library)

    if report.get("return_directory"):
        lines += [
            "",
            "SEND YOUR EDITS BACK",
            "  In Resolve use File > Export > Timeline and choose DaVinci Resolve Timeline (.drt).",
            f"  Save a new file in the export share's {_line(report['return_directory'])} folder.",
            "  The hub worker picks it up automatically (normally within a minute while idle).",
            "  Recipe status will show the returned plan and diff. Grades, Fusion, Fairlight",
            "  and enabled/disabled clip state are not fully reconstructed; review the limits",
            "  before asking the agent to continue. The active plan is never replaced silently.",
        ]

    notes = _rows(report.get("notes"))
    if notes:
        lines += _section(
            "ALSO CHECK",
            [f"  - {_item(n)}" for n in notes[:MAX_LISTED]],
            len(notes),
            report_id,
        )
    return "\r\n".join(lines) + "\r\n"


__all__ = ["MAX_LISTED", "NOTES_MIME", "NOTES_SUFFIX", "handoff_notes"]
