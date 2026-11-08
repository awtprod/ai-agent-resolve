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
from aar.plan.export.layout import AUDIO_CROSSFADE_FORMS, AudioCrossfade, Layout, build_layout
from aar.plan.export.media import MediaMap, load_media_map, media_map_from_mapping
from aar.plan.export.timebase import frame_duration_text
from aar.plan.export.verify import VerifyResult, verify_fcpxml
from aar.plan.inputs import LaughMap, Transcript
from aar.plan.lint import Finding, lint
from aar.plan.model import EditPlan
from aar.plan.profile import Profile

__all__ = [
    "AUDIO_CROSSFADE_FORMS",
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
    audio_crossfade: AudioCrossfade = "split",
) -> Layout:
    """The export layout for ``plan`` with the profile's timeline start."""
    return build_layout(
        plan,
        media_map=media_map,
        timeline_start=profile.export.timeline_start,
        audio_crossfade=audio_crossfade,
    )


def _audio_tracks(layout: Layout) -> list[dict[str, Any]]:
    """The plan audio tracks clips sit on, by name, with their lanes (F9)."""
    rows: list[dict[str, Any]] = []
    for number, track in enumerate(layout.plan.audio_tracks, start=1):
        clips = [c for c in layout.clips if c.audio_track == number and c.enable != "video"]
        if not clips:
            continue
        rows.append(
            {
                "track": f"A{number}",
                "name": track.name,
                "lanes": sorted({c.lane for c in clips}, reverse=True),
                "clips": len(clips),
                "gain_db": track.gain_db,
                "mute": track.mute,
                "bus": track.bus,
            }
        )
    return rows


def _lanes(layout: Layout) -> list[dict[str, Any]]:
    rows: dict[int, str] = {}
    for clip in layout.clips:
        if clip.enable == "audio":
            rows[clip.lane] = f"A{clip.audio_track} (audio-only clips)"
        elif clip.lane == 0:
            rows[0] = "V1, the primary storyline, with its clips' sound (A1)"
        else:
            rows.setdefault(
                clip.lane,
                f"V{clip.lane + 1} (connected clips) with their sound "
                f"(A{clip.lane + 1} by position)",
            )
    if layout.titles:
        rows[layout.title_lane] = f"titles and sting (V{layout.title_lane + 1})"
    return [{"lane": lane, "carries": rows[lane]} for lane in sorted(rows, reverse=True)]


def build_report(
    layout: Layout, text: str, warnings: list[Finding], check: VerifyResult
) -> dict[str, Any]:
    """The export report: what was exported and every plan feature FCPXML cannot carry."""
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
        "exported": {
            "segments": sum(c.location.startswith("segments[") for c in layout.clips),
            "outro": any(c.location == "outro" for c in layout.clips),
            "audio_only_clips": sum(c.enable == "audio" for c in layout.clips),
            "transitions": len(layout.transitions),
            "titles": len(layout.titles),
            "markers": len(layout.markers),
            "assets": len(layout.assets),
        },
        "media": [ref.as_dict() for ref in layout.media.values()],
        "lanes": _lanes(layout),
        "audio_tracks": _audio_tracks(layout),
        "audio_crossfade": layout.audio_crossfade,
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
    audio_crossfade: AudioCrossfade = "split",
) -> ExportResult:
    """Lint, build, write and self-check one plan; a typed :class:`ExportError` on refusal."""
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
    layout = layout_for(plan, profile, media_map=media_map, audio_crossfade=audio_crossfade)
    text = write_fcpxml(layout)
    # The self-check compares the file with a layout built afresh from the plan.
    fresh = layout_for(plan, profile, media_map=media_map, audio_crossfade=audio_crossfade)
    check = verify_fcpxml(text, fresh)
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
    audio_crossfade: AudioCrossfade = "split",
) -> VerifyResult:
    """Read an FCPXML file back and compare it with ``plan`` (``aar plan export --verify``)."""
    layout = layout_for(plan, profile, media_map=media_map, audio_crossfade=audio_crossfade)
    return verify_fcpxml(text, layout)
