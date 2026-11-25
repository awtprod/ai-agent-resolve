"""An EditPlan as a project in a Resolve disk library: the second output of the export's layout.

:func:`export_library` lints the plan (refusing lint errors, as the FCPXML export does), builds
the same :class:`~aar.plan.export.layout.Layout`, and writes it through the library writer
(:func:`aar.drp.libwrite.make_project`): a copy of the owner's seed project with a fresh id and
name, one media-pool clip per source authored from the file's measured facts
(:func:`aar.drp.mediapool.author`), and the seed's timeline filled with one picture item per
clip and its linked sound (:func:`aar.drp.libwrite.cut.author_cut`). Before the project is
placed in the library it is read back with the timeline import (S20) against the plan it was
written from, and refused (``self_check_failed``) unless that import reports no change: the
rule the FCPXML export applies to its own file.

What the library project cannot carry yet is either refused (it would change the cut: a
transition, the head's dissolve, a title or the sting, a clip off V1, a fade, a sound silenced
by a muted track) or listed in the report under ``not_carried`` in the FCPXML report's own
vocabulary (gain, transform and reframe, a label or note, markers, captions; and everything
the layout already lists as not expressible).
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from functools import wraps
from pathlib import Path
from typing import Any, ParamSpec, TypeVar

from aar.drp.items.errors import ItemError
from aar.drp.items.media import media_view
from aar.drp.items.rows import ItemSet, read_items
from aar.drp.library import SQL_ERRORS, LibraryError, LibraryOptions
from aar.drp.libwrite import (
    Authored,
    ProjectResult,
    ProjectWriteError,
    check_project_name,
    make_project,
)
from aar.drp.libwrite.cut import Cut, ItemTemplates, author_cut, pick_item_templates
from aar.drp.libwrite.ops import OPERATIONS, unopened
from aar.drp.libwrite.place import read_placed
from aar.drp.libwrite.seed import Seed
from aar.drp.mediapool import formulas as fm
from aar.drp.mediapool.author import AuthoredClip, Placement, Template, author
from aar.drp.mediapool.errors import DAMAGE_ERRORS, MediaPoolError
from aar.drp.mediapool.facts import MediaFacts
from aar.drp.mediapool.kinds import KINDS, kind_of_facts
from aar.drp.mediapool.templates import ClipRows, read_templates
from aar.plan.apply.inputs import plan_hash
from aar.plan.export import layout_for
from aar.plan.export.errors import ExportError
from aar.plan.export.layout import Layout, ReportItem, muted_without_bus
from aar.plan.inputs import LaughMap, Transcript
from aar.plan.lint import Finding, lint
from aar.plan.model import EditPlan
from aar.plan.profile import Profile

__all__ = ["LibraryExport", "export_library"]

FORMAT = "library"


@dataclass
class LibraryExport:
    report: dict[str, Any]
    warnings: list[Finding]
    result: ProjectResult[dict[str, Any]]


def _refuse(code: str, message: str, hint: str, **details: Any) -> ExportError:
    return ExportError(code, message, hint, **details)


def _from_writer(exc: ProjectWriteError) -> ExportError:
    return ExportError(exc.code, exc.message, exc.hint, **exc.details)


_P = ParamSpec("_P")
_R = TypeVar("_R")


def _refusing(what: str) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """The export's one typed boundary (as the writers' :func:`aar.drp.libwrite.errors.refusing`):
    an ``ExportError`` passes; the library writer's refusal becomes the export's of the same
    code (its details, ``residue`` among them, kept); a media-pool or item refusal becomes
    ``media_refused`` or ``item_refused``; a library reader's refusal and whatever damage the
    code below lets escape become ``bad_input`` with the original type and message, and so does
    an ``OSError`` (a file that cannot be read: the placement turns the library's own into
    ``library_unwritable`` first). Interrupts pass unchanged."""

    def wrap(func: Callable[_P, _R]) -> Callable[_P, _R]:
        @wraps(func)
        def guarded(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            try:
                return func(*args, **kwargs)
            except ExportError:
                raise
            except ProjectWriteError as exc:
                raise _from_writer(exc) from exc
            except ItemError as exc:
                raise _refuse(
                    "item_refused", f"{exc.message} [{exc.code}]", exc.hint, item_code=exc.code
                ) from exc
            except MediaPoolError as exc:
                raise _refuse(
                    "media_refused", f"{exc.message} [{exc.code}]", exc.hint, media_code=exc.code
                ) from exc
            except LibraryError as exc:
                raise _refuse(
                    "bad_input",
                    f"{what}: {exc.message} [{exc.code}]",
                    exc.hint,
                    library_code=exc.code,
                ) from exc
            except OSError as exc:  # reading an input (the library's own are refused below)
                raise _refuse(
                    "bad_input",
                    f"{what}: a file could not be read ({type(exc).__name__}: {exc})",
                    "check the plan's seed, facts and template files are there and readable",
                ) from exc
            except (*DAMAGE_ERRORS, *SQL_ERRORS) as exc:
                detail = f"{type(exc).__name__}: {exc}"[:400]
                raise _refuse(
                    "bad_input",
                    f"{what} failed on a damaged value ({detail})",
                    "report the plan, the seed and the templates",
                ) from exc

        return guarded

    return wrap


def _unsupported(layout: Layout) -> list[dict[str, str]]:
    """Plan features a library project cannot carry yet and that would change the cut."""
    plan = layout.plan
    out: list[dict[str, str]] = []
    for t in layout.transitions:
        out.append({"location": t.location, "feature": f"{t.kind} transition"})
    for t in layout.titles:
        out.append({"location": t.location, "feature": f"{t.kind} title"})
    for c in layout.clips:
        if c.lane != 0:
            out.append({"location": c.location, "feature": "a clip off V1 (another track)"})
        if c.fade_in or c.fade_out:
            out.append({"location": c.location, "feature": "an audio fade"})
        if muted_without_bus(plan, c.audio_track):
            out.append(
                {"location": c.location, "feature": "sound silenced by a muted track (no gain)"}
            )
    return out


def _not_carried(layout: Layout) -> list[dict[str, str]]:
    """What the plan holds and the project does not, in the FCPXML report's vocabulary; none of
    it changes the cut list (the timeline import keeps the plan's own values for all of it)."""
    plan = layout.plan
    items: list[ReportItem] = []
    for c in layout.clips:
        if c.volume_db:
            items.append(
                ReportItem(
                    "gain", c.location, f"{c.volume_db:+.2f} dB", "set the clip's volume by hand"
                )
            )
        if c.transform is not None or c.conform:
            items.append(
                ReportItem(
                    "transform",
                    c.location,
                    "the plan's framing (reframe, transform or fill)",
                    "reframe the clip in the Inspector by hand",
                )
            )
        if c.name and c.name != Path(c.src).name:
            items.append(ReportItem("label", c.location, c.name, "rename the item by hand"))
        if c.note:
            items.append(ReportItem("notes", c.location, c.note, "add it as a clip note by hand"))
    for i, m in enumerate(layout.markers):
        items.append(
            ReportItem(
                "marker",
                m.location,
                f"{m.value!r} at {plan.fps.timecode(layout.tc_start + m.record)} "
                f"({plan.markers[i].color})",
                "add by hand",
            )
        )
    for i, _ in enumerate(plan.captions):
        items.append(ReportItem("caption", f"captions[{i}]", "captions are burned in", "none"))
    return [item.as_dict() for item in items] + [x.as_dict() for x in layout.not_expressible]


def _facts_for(layout: Layout, facts: dict[str, MediaFacts]) -> dict[str, MediaFacts]:
    out: dict[str, MediaFacts] = {}
    missing = []
    for src, ref in layout.media.items():
        found = facts.get(ref.windows_path.lower())
        if found is None:
            missing.append({"source": src, "path": ref.windows_path})
        else:
            out[src] = found
    if missing:
        raise _refuse(
            "facts_missing",
            f"{len(missing)} source(s) have no measured facts: "
            + ", ".join(m["source"] for m in missing[:8]),
            "measure the files on the edit PC with the facts script and pass the file with "
            "--facts (each source's edit-PC path must be in it)",
            sources=missing,
        )
    return out


def _check_facts(layout: Layout, facts: dict[str, MediaFacts]) -> None:
    """Each source's facts must be of the file the plan cuts: a picture at the plan's rate
    holding every frame the plan reads, and its start timecode when the plan states one."""
    plan = layout.plan
    rate = Fraction(plan.fps.num, plan.fps.den)
    used: dict[str, int] = {}  # by file (the clip's src URL)
    for c in layout.clips:
        used[c.src] = max(used.get(c.src, 0), c.src_out)
    for src, f in facts.items():
        url = layout.media[src].src
        source = next(s for s in plan.sources if s.id == src)
        try:
            kind = kind_of_facts(f)
            if kind.media != "video":
                raise _refuse(
                    "facts_mismatch",
                    f"source {src} is a {kind.media} file; a plan segment of it would not read "
                    "back (the timeline import carries only pictures with their own sound)",
                    "cut pictures only, or use the FCPXML export",
                    source=src,
                )
            file_rate = fm.video_rate(f)
            frames = fm.video_frames(f, file_rate)
            timecode = fm.start_timecode(f)
        except MediaPoolError as exc:
            raise _refuse(
                "facts_mismatch",
                f"source {src}: its facts do not give a rate and length ({exc.message})",
                exc.hint,
                source=src,
            ) from None
        if file_rate != rate:
            raise _refuse(
                "facts_mismatch",
                f"source {src} runs at {file_rate} fps, the plan at {rate}",
                "check the facts belong to the plan's file; mixed rates are not supported",
                source=src,
            )
        if used.get(url, 0) > frames:
            raise _refuse(
                "facts_mismatch",
                f"source {src} has {frames} frames by its facts; the plan reads to {used[url]}",
                "check the facts belong to the plan's file",
                source=src,
            )
        if source.start_timecode is not None and source.start_timecode != timecode:
            raise _refuse(
                "facts_mismatch",
                f"source {src}: the plan says it starts at {source.start_timecode}, its facts "
                f"at {timecode}",
                "check the facts belong to the plan's file",
                source=src,
            )


def _check_seed(layout: Layout, seed: Seed, stated: tuple[int, int] | None) -> str:
    plan = layout.plan
    rate = Fraction(plan.fps.num, plan.fps.den)
    hint = (
        "use a seed whose timeline has the plan's frame rate, resolution and start (changing "
        "them in a project is not proven)"
    )
    if seed.rate != rate:
        raise _refuse(
            "seed_mismatch", f"the seed timeline runs at {seed.rate} fps, the plan at {rate}", hint
        )
    if seed.start_frame != layout.tc_start:
        raise _refuse(
            "seed_mismatch",
            f"the seed timeline starts at frame {seed.start_frame}, the plan at "
            f"{layout.tc_start} ({layout.timeline_start})",
            hint,
        )
    want = (plan.resolution.width, plan.resolution.height)
    if seed.resolution is not None:
        if seed.resolution != want:
            raise _refuse(
                "seed_mismatch",
                f"the seed timeline is {seed.resolution[0]}x{seed.resolution[1]} (from the "
                f"{seed.resolution_from}), the plan {want[0]}x{want[1]}",
                hint,
            )
        return str(seed.resolution_from)
    if stated is None:
        raise _refuse(
            "seed_mismatch",
            "the seed timeline follows the project's resolution, which the project does not "
            "store, so it cannot be checked against the plan's",
            "give the seed timeline its own resolution in Resolve (Timeline Settings), or state "
            "the project's with --seed-resolution WxH",
        )
    if stated != want:
        raise _refuse(
            "seed_mismatch",
            f"the stated seed resolution {stated[0]}x{stated[1]} is not the plan's "
            f"{want[0]}x{want[1]}",
            hint,
        )
    return "stated by the caller (the project does not store it)"


@dataclass
class _Candidate:
    clip: ClipRows
    facts: MediaFacts | None
    source: str


def _candidates(
    sets: Sequence[tuple[str, Sequence[ClipRows]]], facts: dict[str, MediaFacts]
) -> list[_Candidate]:
    out = []
    for source, clips in sets:
        for clip in clips:
            try:
                path = media_view(clip).path
            except MediaPoolError:
                continue
            out.append(_Candidate(clip, facts.get(path.lower()), source))
    return out


def _author_media(
    layout: Layout,
    seed: Seed,
    facts: dict[str, MediaFacts],
    candidates: list[_Candidate],
    allow_unverified: bool,
    added_at: int,
) -> tuple[dict[str, AuthoredClip], list[dict[str, Any]]]:
    taken = set(seed.taken)
    clips: dict[str, AuthoredClip] = {}
    rows: list[dict[str, Any]] = []
    project_rate = seed.project_rate or seed.rate
    files: dict[str, str] = {}  # src URL -> the first source (in id order) naming that file
    for src in sorted(facts):
        files.setdefault(layout.media[src].src, src)
    for index, (url, src) in enumerate(sorted(files.items(), key=lambda kv: kv[1])):
        f = facts[src]
        kind = kind_of_facts(f)
        status = KINDS.get(kind)
        ranked = sorted(candidates, key=lambda c: not _same_kind(c.facts, kind))
        placement = Placement(
            folder_id=seed.folder_id,
            folder_index=seed.folder_count + index,
            project_rate=project_rate,
            added_at=added_at,
            lockable_map_id=seed.lockable_map_id,
        )
        last: MediaPoolError | None = None
        authored: AuthoredClip | None = None
        used: _Candidate | None = None
        for cand in ranked:
            try:
                authored = author(
                    f,
                    Template(cand.clip, cand.facts, cand.source),
                    placement,
                    taken=frozenset(taken),
                    allow_unverified=allow_unverified,
                )
                used = cand
                break
            except MediaPoolError as exc:
                if last is None or exc.code != "kind_mismatch":
                    last = exc
        if authored is None or used is None:
            raise _refuse(
                "media_refused",
                f"source {src} ({f.name}): no template clip gives its rows"
                + (f": {last.code}: {last.message}" if last else " (no template clips)"),
                last.hint
                if last
                else "give --templates a project Resolve saved with a clip of the same kind",
                source=src,
                media_code=last.code if last else "no_template",
            )
        taken |= {authored.media_id}
        taken |= set(ClipRows.from_tables(authored.rows).ids())
        clips[url] = authored
        rows.append(
            {
                "source": src,
                "sources": sorted(k for k in facts if layout.media[k].src == url),
                "path": f.path,
                "kind": kind.label(),
                "kind_status": status.status if status else "unsupported",
                "status": authored.status,
                "unverified": list(authored.unverified),
                "template": f"{Path(used.source).name}: {used.clip.media.get('Name')}",
                "notes": list(authored.notes),
            }
        )
    return clips, rows


def _same_kind(facts: MediaFacts | None, kind: object) -> bool:
    """A template whose own facts are of the file's kind (tried first: it can verify)."""
    if facts is None:
        return False
    try:
        return kind_of_facts(facts) == kind
    except MediaPoolError:
        return False


def _parse_resolution(text: str | None) -> tuple[int, int] | None:
    if text is None:
        return None
    w, _, h = text.lower().partition("x")
    if not (w.isascii() and w.isdigit() and h.isascii() and h.isdigit()):
        raise _refuse("seed_mismatch", f"--seed-resolution {text!r} is not WxH", "e.g. 1920x1080")
    return int(w), int(h)


@_refusing("exporting the plan as a library project")
def export_library(
    plan: EditPlan,
    profile: Profile,
    *,
    library: Path,
    seed: Path,
    facts: dict[str, MediaFacts],
    templates: Sequence[Path] = (),
    project_name: str,
    media_map: Any = None,
    transcript: Transcript | None = None,
    laughs: LaughMap | None = None,
    allow_unverified_media: bool = False,
    allow_unopened: bool = False,
    seed_resolution: str | None = None,
    user: str | None = None,
    options: LibraryOptions | None = None,
) -> LibraryExport:
    """Lint, lay out, write, self-check and place one plan as a library project; a typed
    :class:`ExportError` on refusal (nothing is placed then). See the module docstring."""
    from aar.plan.timeline_import import (
        READER_WARNING_CODES,
        TimelineImportError,
        import_timeline,
        read_timeline,
    )

    check_project_name(project_name)  # before any work (a bad_project_name refusal)
    opts = options or LibraryOptions()
    report = lint(plan, profile, transcript=transcript, laughs=laughs)
    if not report.ok:
        errors = [f for f in report.findings if f.severity == "error"]
        raise _refuse(
            "lint_errors",
            f"the plan has {len(errors)} lint error(s)",
            "fix the lint errors (aar plan lint) before exporting",
            errors=[{"rule": f.rule, "message": f.message} for f in errors],
        )
    warnings = [f for f in report.findings if f.severity == "warn"]
    layout = layout_for(plan, profile, media_map=media_map, allow_unknown_timecode=True)
    blocked = _unsupported(layout)
    if blocked:
        raise _refuse(
            "unsupported_plan_feature",
            f"the plan has {len(blocked)} feature(s) a library project cannot carry yet: "
            + ", ".join(sorted({b["feature"] for b in blocked})),
            "remove them from the plan, or export it as FCPXML (--format fcpxml)",
            features=blocked,
        )
    stated = _parse_resolution(seed_resolution)
    used_facts = _facts_for(layout, facts)
    _check_facts(layout, used_facts)
    try:
        template_sets = [(str(p), read_templates(p, options=opts).clips) for p in templates]
        item_sets: list[ItemSet] = [read_items(p, options=opts) for p in templates]
    except LibraryError as exc:
        raise _refuse(
            "media_refused",
            f"a template project cannot be read: {exc.message} [{exc.code}]",
            exc.hint,
        ) from None
    except MediaPoolError as exc:
        raise _refuse(
            "media_refused", f"a template project: {exc.message} [{exc.code}]", exc.hint
        ) from None
    added_at = int(time.time())
    state: dict[str, Any] = {}

    def author_all(s: Seed) -> Authored:
        state["resolution_from"] = _check_seed(layout, s, stated)
        candidates = _candidates([*template_sets, (str(seed), s.clips)], facts)
        clips, media_rows = _author_media(
            layout, s, used_facts, candidates, allow_unverified_media, added_at
        )
        # The seed's own items are templates too (read_seed has read the same file; a failure
        # here is the seed's and is refused, never skipped).
        seed_items = read_items(s.path, options=LibraryOptions(settle_seconds=0.0))
        picks: ItemTemplates = pick_item_templates([*item_sets, seed_items])
        cuts = []
        for c in layout.clips:
            rows = ClipRows.from_tables(clips[c.src].rows)
            cuts.append(
                Cut(
                    rows,
                    c.record_in,
                    c.record_out - c.record_in,
                    c.src_in,
                    picture=True,
                    sound=bool(rows.audio),
                )
            )
        try:
            items, specs = author_cut(s, cuts, picks)
        except ItemError as exc:
            raise _refuse("item_refused", f"{exc.message} [{exc.code}]", exc.hint) from None
        all_clips = [clips[src] for src in sorted(clips)]
        return Authored(all_clips, items, specs, extra=media_rows, cuts=cuts)

    def import_check(path: Path, s: Seed, authored: Authored) -> dict[str, Any]:
        try:
            timeline = read_timeline(path, library_options=LibraryOptions(settle_seconds=0.0))
            result = import_timeline(
                plan, timeline.timeline, timeline.media, media_map=media_map, profile=None
            )
        except (TimelineImportError, ExportError) as exc:
            raise _refuse(
                "self_check_failed",
                f"the written project does not import back: {exc.code}: {exc.message}",
                "this is an exporter bug; report the plan",
            ) from None
        diff = result.diff
        problems = []
        if diff.get("unchanged") is not True:
            problems.append("the import of the written project reports changes")
        for key in ("not_carried", "foreign", "added", "dropped", "track_mismatch"):
            if diff.get(key):
                problems.append(f"{key}: {len(diff[key])}")
        if diff.get("export_errors"):
            problems.append("the edited plan does not lay out")
        if problems:
            raise _refuse(
                "self_check_failed",
                "the written project does not reproduce the plan: " + "; ".join(problems),
                "this is an exporter bug; report the plan",
                problems=problems,
            )
        return {
            "unchanged": True,
            "segments": len(diff.get("segments") or []),
            "reader_warnings": diff.get("reader_warnings", {}).get("count", 0),
        }

    result = make_project(
        library,
        seed,
        project_name=project_name,
        timeline_name=plan.timeline,
        author=author_all,
        allowed_issues=READER_WARNING_CODES,
        check=import_check,
        allow_unopened=allow_unopened,
        user=user,
        options=opts,
    )
    out = _report(layout, result, warnings, state.get("resolution_from"), seed)
    return LibraryExport(out, warnings, result)


def _report(
    layout: Layout,
    result: ProjectResult[dict[str, Any]],
    warnings: list[Finding],
    resolution_from: str | None,
    seed_path: Path,
) -> dict[str, Any]:
    plan = layout.plan
    res = plan.resolution
    built = result.built
    try:  # the project is placed: a failure to hash it is reported, not a refusal
        # Read only when it is the file the placement wrote (else a typed refusal).
        digest: str | None = hashlib.sha256(read_placed(result.placed)).hexdigest()
    except OSError as exc:
        digest = None
        result.placed.cleanup.append(f"the placed Project.db could not be read to hash it: {exc}")
    ops = list(built.operations)
    items = result.authored.items
    return {
        "format": FORMAT,
        "project": {
            "name": result.drp.project.name if result.drp.project else None,
            "folder": result.placed.folder.name,
            "path": str(result.placed.path),
            "user": result.placed.user,
            "project_id": built.project_id,
            "sha256": digest,
        },
        "plan": {"project": plan.project, "timeline": plan.timeline, "sha256": plan_hash(plan)},
        "timeline": {
            "name": plan.timeline,
            "fps": f"{plan.fps.num}/{plan.fps.den}",
            "start": layout.timeline_start,
            "tc_start_frames": layout.tc_start,
            "total_frames": layout.total_frames,
            "resolution": f"{res.width}x{res.height}",
            "resolution_checked_against": resolution_from,
        },
        "seed": {
            "path": str(seed_path),
            "project": result.seed.project_name,
            "timeline": result.seed.timeline_name,
            "items_removed": len(result.seed.items),
            "project_rate_stored": result.seed.project_rate is not None,
        },
        "open": [
            "In Resolve open the Project Manager on this library and open the project.",
            "Media link by path: each clip names its file on the edit PC (media, below).",
        ],
        "operations": [OPERATIONS[name].as_dict() for name in ops],
        "operations_not_opened": unopened(ops),
        "media": result.authored.extra,
        "exported": {
            "segments": sum(c.location.startswith("segments[") for c in layout.clips),
            "outro": any(c.location == "outro" for c in layout.clips),
            "picture_items": sum(i.kind == "picture" for i in items),
            "sound_items": sum(i.kind == "sound" for i in items),
        },
        "not_carried": _not_carried(layout),
        "notes": [item.as_dict() for item in layout.notes],
        "placement": {
            "permissions": result.placed.permissions,
            "staging_leftovers": result.placed.leftovers,
            "cleanup": result.placed.cleanup,
        },
        "lint_warnings": [{"rule": f.rule, "message": f.message} for f in warnings],
        "self_check": {
            "ok": True,
            "integrity_check": "ok",
            "reader_issues": sorted({i.code for i in result.drp.issues}),
            "import": result.check,
        },
    }
