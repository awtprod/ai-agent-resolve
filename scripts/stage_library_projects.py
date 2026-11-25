"""Build small staging projects for a person to open in Resolve: one per unproven operation.

    python scripts/stage_library_projects.py --out NEW_FOLDER --seed SEED.db --facts FACTS.jsonl \\
        --templates PROJECT.db [PROJECT.db ...]

Not part of the product path: it calls the product code (``aar.drp.libwrite``,
``aar.plan.export.library``) to build, self-check and place each project in a library of its
own at ``NEW_FOLDER`` (a folder that must not exist yet, so it is never a real library). Each
project is as small as possible and named for what it tests; ``NEW_FOLDER/STAGING.md`` says, for
each one, what a person should see in Resolve if it worked, and ``NEW_FOLDER/staging.json``
holds every report. The seed must hold a media clip of its own with sound (its timeline's items
are removed by ``remove_items``); the facts must cover two files of a verified picture kind at
the seed timeline's rate (for the authored clips).

Projects (``aar-stage-...``):

a. two clips authored from facts, of a verified kind, cut together (the plan export);
b. the seed's own clip in linked cuts, the timeline renamed (only ``rename_timeline`` new);
c1. picture-only items; c2. sound-only items (the seed's own clip);
d1. linked cuts with gaps; d2. a split edit, the sound leading the picture by 10 frames.
"""

from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

from aar.drp.items.media import media_view
from aar.drp.items.rows import read_items
from aar.drp.library import LibraryOptions
from aar.drp.libwrite import Authored, ProjectWriteError, make_project
from aar.drp.libwrite.cut import Cut, author_cut, pick_item_templates
from aar.drp.libwrite.seed import read_seed
from aar.drp.mediapool import formulas as fm
from aar.drp.mediapool.errors import MediaPoolError
from aar.drp.mediapool.gate import load_facts
from aar.drp.mediapool.kinds import KINDS, kind_of_facts
from aar.plan.export.errors import ExportError
from aar.plan.export.library import export_library
from aar.plan.model import EditPlan
from aar.plan.profile import parse_profile
from aar.plan.timeline_import import READER_WARNING_CODES

_OPTS = LibraryOptions(settle_seconds=0.0)
_PROFILE = {
    "name": "staging",
    "qa": {
        "longform_lufs": -22.0,
        "short_lufs": -14.5,
        "lufs_tolerance": 0.5,
        "review_mp4_max_mb": 20,
    },
    "rules": {"ranges-valid": {}},
    "export": {"timeline_start": "01:00:00:00"},
}


def _tc(frames: int, rate: Fraction) -> str:
    nominal = round(rate)
    ff = frames % nominal
    s = frames // nominal
    return f"{s // 3600:02d}:{s // 60 % 60:02d}:{s % 60:02d}:{ff:02d}"


def _writer_project(
    library: Path,
    seed: Path,
    name: str,
    templates: list[Path],
    cuts_of: Any,
    timeline_name: str | None = None,
) -> dict[str, Any]:
    """A writer-level project (the seed's own clip): integrity, reader and cut-list checks."""
    picks = pick_item_templates([read_items(p, options=_OPTS) for p in [*templates, seed]])
    facts_out: dict[str, Any] = {}

    def author(s: Any) -> Authored:
        clip = next(c for c in s.clips if c.audio and c.video is not None)
        cuts = cuts_of(clip)
        items, specs = author_cut(s, cuts, picks)
        facts_out["clip"] = media_view(clip).name
        facts_out["start"] = s.start_frame
        facts_out["rate"] = s.rate
        facts_out["cuts"] = cuts
        return Authored([], items, specs)

    result = make_project(
        library,
        seed,
        project_name=name,
        timeline_name=timeline_name,
        author=author,
        allowed_issues=READER_WARNING_CODES,
        allow_unopened=True,
        options=_OPTS,
    )
    return {
        "project": name,
        "path": str(result.placed.path),
        "operations": result.operations(),
        "self_check": "integrity ok; read back with no refusing issue; cut list as written",
        "timeline": timeline_name or result.seed.timeline_name,
        "clip": facts_out["clip"],
        "start": facts_out["start"],
        "rate": facts_out["rate"],
        "cuts": facts_out["cuts"],
    }


def _describe(entry: dict[str, Any], what: str) -> str:
    rate = entry["rate"]
    parts = []
    for cut in entry["cuts"]:
        kinds = "+".join(k for k, on in (("picture", cut.picture), ("sound", cut.sound)) if on)
        at = _tc(entry["start"] + cut.record_in, rate)
        line = f"{kinds} at {at}, {cut.duration} frames from source frame {cut.src_in}"
        if cut.sound_record_in is not None:
            line += (
                f" (sound at {_tc(entry['start'] + cut.sound_record_in, rate)}, "
                f"{cut.sound_duration} frames from source frame {cut.sound_src_in})"
            )
        parts.append(line)
    return (
        f"{what} Timeline {entry['timeline']!r} holds, of clip {entry['clip']!r}: "
        + "; ".join(parts)
        + ". The project opens with no relink or offline prompt; the cuts play the clip's "
        "picture and sound at those source frames."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True, help="a new folder (must not exist)")
    parser.add_argument("--seed", type=Path, required=True)
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--templates", type=Path, nargs="+", required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        print(f"error: {args.out} exists; give a new folder", file=sys.stderr)
        return 2
    (args.out / "Resolve Projects" / "Users" / "guest" / "Projects").mkdir(parents=True)
    library = args.out
    entries: list[dict[str, Any]] = []
    statements: list[str] = []
    try:
        facts = load_facts(args.facts)
        seed = read_seed(args.seed, options=_OPTS)
        seed_rate = seed.rate
        # (a) two authored clips of a verified picture kind at the seed's rate.
        chosen = []
        for f in sorted(facts.values(), key=lambda x: x.path):
            try:
                kind = kind_of_facts(f)
                status = KINDS.get(kind)
                if kind.media != "video" or status is None or status.status != "verified":
                    continue
                if fm.video_rate(f) != seed_rate:
                    continue
                frames = fm.video_frames(f, seed_rate)
            except MediaPoolError:
                continue
            if frames >= 60 and fm.start_timecode(f) is not None:
                chosen.append((f, frames))
            if len(chosen) == 2:
                break
        if len(chosen) == 2:
            sources = [
                {"id": f"s{i}", "media": f.path, "duration": n} for i, (f, n) in enumerate(chosen)
            ]
            plan = EditPlan.model_validate(
                {
                    "version": 2,
                    "project": "staging",
                    "timeline": seed.timeline_name,
                    "kind": "longform",
                    "fps": f"{seed_rate.numerator}/{seed_rate.denominator}",
                    "resolution": {"width": 1920, "height": 1080},
                    "profile": "staging",
                    "sources": sources,
                    "segments": [
                        {"track": 1, "source": "s0", "src_in": 0, "src_out": 30},
                        {"track": 1, "source": "s1", "src_in": 10, "src_out": 40},
                    ],
                }
            )
            result = export_library(
                plan,
                parse_profile(_PROFILE),
                library=library,
                seed=args.seed,
                facts=facts,
                templates=args.templates,
                project_name="aar-stage-a-authored-clips",
                seed_resolution="1920x1080",
                allow_unopened=True,
                options=_OPTS,
            )
            entries.append({"project": "aar-stage-a-authored-clips", "report": result.report})
            names = [Path(f.path.replace("\\", "/")).name for f, _ in chosen]
            tc1 = fm.start_timecode(chosen[1][0])
            statements.append(
                "aar-stage-a-authored-clips: the project opens with no relink or offline prompt; "
                f"the media pool holds {names[0]!r} and {names[1]!r} online, with thumbnails "
                "and their own start timecodes; the timeline holds two picture-only cuts from "
                f"01:00:00:00: 30 frames of {names[0]!r} from its first frame, then 30 frames of "
                f"{names[1]!r} from its frame 10 (its source timecode at the cut is {tc1} plus "
                "10 frames); nothing is re-analysed or relinked. (Uses add_media, the main open "
                "risk, and add_picture_only_items: the verified kinds at this rate have no sound.)"
            )
        else:
            statements.append(
                "aar-stage-a-authored-clips: NOT BUILT: the facts hold no two files of a verified "
                "picture kind at the seed timeline's rate"
            )
        # (b)..(d) writer-level projects on the seed's own clip.
        specs: list[tuple[str, str | None, Any, str]] = [
            (
                "aar-stage-b-renamed-timeline",
                "aar renamed timeline",
                lambda clip: [Cut(clip, 0, 48, 0), Cut(clip, 48, 48, 240), Cut(clip, 96, 48, 120)],
                "Only the timeline's name is new: it reads 'aar renamed timeline' in the timeline "
                "tab and in the media pool.",
            ),
            (
                "aar-stage-c1-picture-only",
                None,
                lambda clip: [
                    Cut(clip, 0, 48, 0, sound=False),
                    Cut(clip, 48, 48, 240, sound=False),
                ],
                "Picture items with no sound: V1 holds the cuts, A1 is empty.",
            ),
            (
                "aar-stage-c2-sound-only",
                None,
                lambda clip: [
                    Cut(clip, 0, 48, 0, picture=False),
                    Cut(clip, 48, 48, 240, picture=False),
                ],
                "Sound items with no picture: A1 holds the cuts, V1 is empty.",
            ),
            (
                "aar-stage-d1-gaps",
                None,
                lambda clip: [Cut(clip, 15, 48, 0), Cut(clip, 93, 48, 240)],
                "Gaps: 15 empty frames at the start and 30 between the two linked cuts.",
            ),
            (
                "aar-stage-d2-split-edit",
                None,
                lambda clip: [
                    Cut(
                        clip,
                        10,
                        48,
                        250,
                        sound_record_in=0,
                        sound_src_in=240,
                        sound_duration=58,
                    )
                ],
                "A split edit: the sound starts 10 frames before the picture (J-cut), the two "
                "still linked (moving one moves the other).",
            ),
        ]
        for name, timeline_name, cuts_of, what in specs:
            entry = _writer_project(
                library, args.seed, name, args.templates, cuts_of, timeline_name
            )
            statements.append(f"{name}: " + _describe(entry, what))
            entry["cuts"] = [
                {k: getattr(c, k) for k in c.__dataclass_fields__ if k != "clip"}
                for c in entry["cuts"]
            ]
            entry["rate"] = str(entry["rate"])
            entries.append(entry)
    except (ExportError, ProjectWriteError, MediaPoolError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    (args.out / "staging.json").write_text(json.dumps(entries, indent=2, default=str) + "\n")
    (args.out / "STAGING.md").write_text(
        "# Staging projects: what a person should see in Resolve if each worked\n\n"
        + "\n".join(f"- {line}" for line in statements)
        + "\n"
    )
    print("\n".join(statements))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
