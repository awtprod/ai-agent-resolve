"""``aar.plan.export``: an EditPlan v2 as an FCPXML 1.10 timeline Resolve imports, with no Resolve.

A pure, cold, deterministic library on the hub, a sibling of the ``plan.apply`` compiler (which
stays as it is). :func:`export_fcpxml` lints the plan (and refuses one with lint errors), builds
the :class:`~aar.plan.export.layout.Layout`, writes the document with the stdlib, reads it back
with our own reader, and refuses to hand out a file that does not reproduce the plan exactly.
Every plan feature FCPXML cannot carry is listed in the export report, with its location.

* :mod:`~aar.plan.export.timebase` -- integer frames <-> exact rational seconds.
* :mod:`~aar.plan.export.media` -- plan media -> ``file:`` URLs Resolve on the edit PC can open.
* :mod:`~aar.plan.export.layout` -- every mapping decision, in plan terms, plus the report items.
* :mod:`~aar.plan.export.fcpxml` -- layout -> FCPXML text.
* :mod:`~aar.plan.export.reader` / :mod:`~aar.plan.export.verify` -- the self-check.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from aar.plan.apply.inputs import plan_hash
from aar.plan.export.errors import EXPORT_CODES, ExportError
from aar.plan.export.fcpxml import FCPXML_VERSION, write_fcpxml
from aar.plan.export.layout import SPLIT_OVERLAP, Layout, build_layout
from aar.plan.export.media import MediaMap, load_media_map, media_map_from_mapping
from aar.plan.export.timebase import frame_duration_text
from aar.plan.export.verify import VerifyResult, verify_fcpxml
from aar.plan.inputs import LaughMap, Transcript
from aar.plan.lint import Finding, lint
from aar.plan.model import EditPlan
from aar.plan.profile import Profile

__all__ = [
    "EXPORT_CODES",
    "ExportError",
    "ExportResult",
    "VerifyResult",
    "build_layout",
    "export_fcpxml",
    "layout_for",
    "load_media_map",
    "media_map_from_mapping",
    "verify_plan_fcpxml",
]


@dataclass
class ExportResult:
    """The written document, its report, and the lint warnings that did not block it."""

    fcpxml: str
    report: dict[str, Any]
    warnings: list[Finding]
    layout: Layout


def layout_for(
    plan: EditPlan,
    profile: Profile,
    *,
    media_map: MediaMap | None = None,
    allow_unknown_timecode: bool = False,
) -> Layout:
    """The export layout for ``plan`` with the profile's timeline start."""
    return build_layout(
        plan,
        media_map=media_map,
        timeline_start=profile.export.timeline_start,
        allow_unknown_timecode=allow_unknown_timecode,
    )


#: How to bring the file into Resolve (import test, 2026-11-08).
IMPORT_STEPS = [
    "In Resolve use File > Import > Timeline and pick the .fcpxml.",
    "Leave 'Import source clips' on: it links each clip to its media file. A scripted import "
    "needs importSourceClips: True with sourceClipsPath (the folder holding the media); with it "
    "off the clips come in unlinked (Media Offline).",
    "A clip links only when its asset starts at its file's start timecode "
    "(Source.start_timecode); the notes name every source without one.",
    "A 9:16 short (any size other than the project's): importing it changes the project's "
    "resolution, which flips every timeline without custom settings. Import a short into a "
    "project of its own, or a timeline with custom settings, and tick 'Use sizing information' "
    "and 'Automatically set project settings' (the import otherwise ignores the FCPXML size: a "
    "9:16 short came in at 1920x1080).",
    "Audio crossfades and fades arrive as volume keyframes on overlapping clips; replace them "
    "with Fairlight crossfades by hand if preferred.",
]


def _audio_tracks(layout: Layout) -> list[dict[str, Any]]:
    """The plan audio tracks clips' sound is planned on, by name, with where it lands after
    import (dense, F9 and round 3)."""
    rows: list[dict[str, Any]] = []
    for number, track in enumerate(layout.plan.audio_tracks, start=1):
        clips = [c for c in layout.clips if c.audio_track == number]
        if not clips:
            continue
        imported = sorted({layout.imported_audio_track(c.lane) for c in clips})
        rows.append(
            {
                "track": f"A{number}",
                "name": track.name,
                "imported": [f"A{n}" for n in imported],
                "clips": len(clips),
                "gain_db": track.gain_db,
                "mute": track.mute,
                "bus": track.bus,
            }
        )
    return rows


def _split_crossfade_audio(layout: Layout) -> list[dict[str, Any]]:
    """Each split crossfade and where its overlapping sound goes: the two clips' sound overlap,
    and Resolve puts the overlap on another audio track (the rerun found A1 holding the later
    clip's sound, round 4), so no track is asserted."""
    rows: list[dict[str, Any]] = []
    for spec in layout.transitions:
        if spec.form != "split":
            continue
        pair = [
            c
            for c in layout.clips
            if c.lane == 0 and spec.cut in (c.record_in, c.record_out) and c.audio_track
        ]
        rows.append(
            {
                "location": spec.location,
                "cut": spec.cut,
                "timecode": layout.plan.fps.timecode(layout.tc_start + spec.cut),
                "frames": spec.frames,
                "plan_tracks": sorted({f"A{c.audio_track}" for c in pair}),
                "imported": sorted({f"A{layout.imported_audio_track(c.lane)}" for c in pair}),
                "overlap": SPLIT_OVERLAP,
            }
        )
    return rows


def _lanes(layout: Layout) -> list[dict[str, Any]]:
    """Each lane in the file, the plan track it carries and the track it is expected on after
    import (lanes are dense, so plan V3 with no V2 imports as V2)."""
    rows: dict[int, dict[str, Any]] = {}
    for track, lane in sorted(layout.lanes.items()):
        if lane == 0:
            what = "V1, the primary storyline, with its clips' sound"
        else:
            what = f"plan V{track} (connected clips) with their sound"
        rows[lane] = {
            "lane": lane,
            "carries": what,
            "imported_video": f"V{lane + 1}",
            "imported_audio": f"A{layout.imported_audio_track(lane)}",
        }
    for lane in sorted({t.lane for t in layout.titles}):
        rows[lane] = {
            "lane": lane,
            "carries": "titles: " + ", ".join(t.location for t in layout.titles if t.lane == lane),
            "imported_video": f"V{lane + 1}",
            "imported_audio": None,
        }
    return [rows[lane] for lane in sorted(rows, reverse=True)]


def _titles(layout: Layout) -> list[dict[str, Any]]:
    """Each title and the sting with the lane it sits on: one lane while none overlap in time,
    the first free lane above for one that overlaps another (Resolve trims overlaps on a
    track)."""
    fps = layout.plan.fps
    return [
        {
            "location": t.location,
            "kind": t.kind,
            "name": t.name,
            "record_in": t.record_in,
            "record_out": t.record_out,
            "timecode": fps.timecode(layout.tc_start + t.record_in),
            "lane": t.lane,
            "imported_video": f"V{t.lane + 1}",
        }
        for t in layout.titles
    ]


def build_report(
    layout: Layout, text: str, warnings: list[Finding], check: VerifyResult
) -> dict[str, Any]:
    """The export report: what was exported, how to import it, and every plan feature FCPXML
    (or Resolve's import) cannot carry."""
    plan = layout.plan
    res = plan.resolution
    return {
        "format": "fcpxml",
        "fcpxml_version": FCPXML_VERSION,
        "fcpxml_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "plan": {"project": plan.project, "timeline": plan.timeline, "sha256": plan_hash(plan)},
        "timeline": {
            "fps": layout.timebase.rate_text,
            "frame_duration": frame_duration_text(layout.timebase),
            "start": layout.timeline_start,
            "tc_start_frames": layout.tc_start,
            "total_frames": layout.total_frames,
            "resolution": f"{res.width}x{res.height}",
        },
        "import": IMPORT_STEPS,
        "exported": {
            "segments": sum(c.location.startswith("segments[") for c in layout.clips),
            "outro": any(c.location == "outro" for c in layout.clips),
            "transitions": len(layout.transitions),
            "titles": len(layout.titles),
            "assets": len(layout.assets),
        },
        "media": [ref.as_dict() for ref in layout.media.values()],
        "lanes": _lanes(layout),
        "titles": _titles(layout),
        "audio_tracks": _audio_tracks(layout),
        "split_crossfade_audio": _split_crossfade_audio(layout),
        "transitions": [
            {
                "location": t.location,
                "cut": t.cut,
                "frames": t.frames,
                "alignment": t.alignment,
                "video": t.video,
                "audio": t.audio,
                "form": t.form,
            }
            for t in layout.transitions
        ],
        "not_imported": {
            "heading": "not imported: add by hand",
            "markers": [
                {
                    "location": m.location,
                    "timecode": plan.fps.timecode(layout.tc_start + m.record),
                    "record_frame": m.record,
                    "duration": m.duration,
                    "name": m.value,
                    "color": plan.markers[i].color,
                    "note": m.note,
                }
                for i, m in enumerate(layout.markers)
            ],
        },
        "not_expressible": [item.as_dict() for item in layout.not_expressible],
        "notes": [item.as_dict() for item in layout.notes],
        "lint_warnings": [{"rule": f.rule, "message": f.message} for f in warnings],
        "self_check": check.as_dict(),
    }


def export_fcpxml(
    plan: EditPlan,
    profile: Profile,
    *,
    media_map: MediaMap | None = None,
    transcript: Transcript | None = None,
    laughs: LaughMap | None = None,
    allow_unknown_timecode: bool = False,
) -> ExportResult:
    """Lint, build, write and self-check one plan; a typed :class:`ExportError` on refusal.

    A used source with no ``start_timecode`` is refused (``start_timecode_unknown``) unless
    ``allow_unknown_timecode``: its asset then starts at 0s, with a report note."""
    report = lint(plan, profile, transcript=transcript, laughs=laughs)
    if not report.ok:
        errors = [f for f in report.findings if f.severity == "error"]
        raise ExportError(
            "lint_errors",
            f"the plan has {len(errors)} lint error(s)",
            "fix the lint errors (aar plan lint) before exporting",
            errors=[{"rule": f.rule, "message": f.message} for f in errors],
        )
    warnings = [f for f in report.findings if f.severity == "warn"]
    layout = layout_for(
        plan, profile, media_map=media_map, allow_unknown_timecode=allow_unknown_timecode
    )
    text = write_fcpxml(layout)
    # The self-check compares the file with a layout built afresh from the plan.
    check = verify_fcpxml(
        text,
        layout_for(
            plan, profile, media_map=media_map, allow_unknown_timecode=allow_unknown_timecode
        ),
    )
    if not check.ok:
        raise ExportError(
            "self_check_failed",
            f"the written FCPXML does not reproduce the plan ({len(check.problems)} problem(s))",
            "this is an exporter bug; report the plan and the problems",
            problems=check.problems[:50],
        )
    return ExportResult(text, build_report(layout, text, warnings, check), warnings, layout)


def verify_plan_fcpxml(
    text: str,
    plan: EditPlan,
    profile: Profile,
    *,
    media_map: MediaMap | None = None,
) -> VerifyResult:
    """Read an FCPXML file back and compare it with ``plan`` (``aar plan export --verify``)."""
    return (
        verify_fcpxml(  # any file: an asset at 0s for an unknown timecode is compared, not refused
            text, layout_for(plan, profile, media_map=media_map, allow_unknown_timecode=True)
        )
    )
