"""The gate: author every measured file's rows from another file of its kind, compare with Resolve.

For every clip of the given Resolve-saved projects whose file has measured facts, :func:`run`
authors the clip's rows from the facts with a *different* clip of the same kind as the template
(leave-one-out: the first other member that is not stale, preferring the same project) and
compares them with Resolve's own rows for that file (:func:`aar.drp.mediapool.fields.compare`).
The template's facts are checked against its own rows on every authoring
(:func:`aar.drp.mediapool.author.author`); a reason the result would not be verified counts
against the kind like a differing field. A file Resolve saw at another modification time than
the one measured (its ``Clip`` field 3 differs) changed after Resolve read it, so its facts are
not the facts Resolve measured: it is counted as ``stale`` and not compared. A ``Clip`` field 3
that does not parse is reported and the file compared anyway (so the field shows the
difference). A kind with one member cannot be checked and is listed ``single``. Each kind's
**envelope** is the set of values the checked files had for each property of
:func:`aar.drp.mediapool.kinds.envelope_of`: what the evidence covers.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from fractions import Fraction
from pathlib import Path

from aar.drp.library import LibraryOptions
from aar.drp.mediapool import blobtree as bt
from aar.drp.mediapool.author import Placement, Template, author
from aar.drp.mediapool.errors import MediaPoolError
from aar.drp.mediapool.facts import MAX_FACTS_BYTES, MediaFacts, facts_from_json, parse_local_time
from aar.drp.mediapool.fields import compare, media_class
from aar.drp.mediapool.kinds import (
    ENVELOPE_PROPERTIES,
    STILL_EXTENSIONS,
    KindKey,
    KindStatus,
    envelope_of,
    kind_of_facts,
)
from aar.drp.mediapool.templates import ClipRows, TemplateSet, read_templates

__all__ = ["KindScore", "Member", "Scoreboard", "load_facts", "run"]

MAX_FACTS_FILE = 256 * 1024 * 1024
MAX_FACT_LINES = 100_000


@dataclass
class Member:
    #: The project's resolved full path: every library project's file is Project.db.
    project: str
    clip: ClipRows
    facts: MediaFacts
    stale: bool
    rate: Fraction | None
    map_id: str | None
    #: ``Clip`` field 3 did not parse: whether the file changed since Resolve read it is unknown.
    seen_unreadable: bool = False


@dataclass
class KindScore:
    kind: KindKey
    members: int = 0
    stale: int = 0
    checked: int = 0
    refused: int = 0
    fields: int = 0
    equal: int = 0
    differing: dict[str, int] = field(default_factory=dict)
    examples: dict[str, str] = field(default_factory=dict)
    refusals: dict[str, int] = field(default_factory=dict)
    #: Excluded fields that did differ, per exclusion pattern: the evidence each one is needed.
    hidden: dict[str, int] = field(default_factory=dict)
    #: For a kind with one member: the member authored from its own rows (not a verification).
    self_check: str | None = None
    #: Reasons an authored result would not be verified (a template whose facts do not
    #: reproduce its own rows, an unproven look), with how often each occurred.
    unverified: dict[str, int] = field(default_factory=dict)
    #: Files whose ``Clip`` field 3 Resolve wrote in a form that does not parse.
    unreadable: int = 0
    #: The values the checked files had for each envelope property.
    envelope: dict[str, set[str]] = field(default_factory=dict)

    @property
    def status(self) -> str:
        """``equal`` (every checked file equal), ``differs``, ``refused`` (author refused every
        file), ``single`` (one member: nothing to check against) or ``stale`` (every file
        changed after Resolve read it)."""
        if self.checked == 0 and self.refused:
            return "refused"
        if self.checked == 0:
            return "single" if self.members == 1 else "stale"
        return "differs" if self.differing or self.refused or self.unverified else "equal"


@dataclass
class Scoreboard:
    kinds: list[KindScore]
    unmatched: list[str]
    skipped: list[str]

    def passed(self, supported: Iterable[KindKey]) -> bool:
        by_kind = {s.kind: s for s in self.kinds}
        return all(k in by_kind and by_kind[k].status == "equal" for k in supported)

    def envelope_mismatches(self, registry: dict[KindKey, KindStatus]) -> list[str]:
        """The verified kinds whose recorded envelope is not what the checked files covered."""
        by_kind = {s.kind: s for s in self.kinds}
        out = []
        for kind, status in registry.items():
            if status.status != "verified":
                continue
            score = by_kind.get(kind)
            measured = {p: frozenset(v) for p, v in (score.envelope if score else {}).items()}
            recorded = dict(status.envelope or {})
            if measured != recorded:
                out.append(
                    f"{kind.label()}: recorded {format_envelope(recorded)}; "
                    f"checked {format_envelope(measured)}"
                )
        return out


def load_facts(path: Path) -> dict[str, MediaFacts]:
    """The facts file (JSON lines), keyed by the path in lower case (Windows paths ignore case).
    Lines for files that did not exist are skipped; a malformed line is an error."""
    size = path.stat().st_size
    if size > MAX_FACTS_FILE:
        raise MediaPoolError(
            "bad_facts", f"{path} is larger than {MAX_FACTS_FILE} bytes", "split it"
        )
    out: dict[str, MediaFacts] = {}
    with path.open("rb") as stream:
        for number, line in enumerate(stream, 1):
            if number > MAX_FACT_LINES:
                raise MediaPoolError(
                    "bad_facts", f"{path} has more than {MAX_FACT_LINES} lines", "split it"
                )
            if len(line) > MAX_FACTS_BYTES:
                raise MediaPoolError("bad_facts", f"{path}:{number} is too long", "measure again")
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except ValueError as exc:
                raise MediaPoolError(
                    "bad_facts", f"{path}:{number} is not JSON ({exc})", "measure again"
                ) from None
            if isinstance(data, dict) and data.get("exists") is False:
                continue
            try:
                facts = facts_from_json(data)
            except MediaPoolError as exc:
                raise MediaPoolError(
                    exc.code, f"{path}:{number}: {exc.message}", exc.hint
                ) from None
            out[facts.path.lower()] = facts
    return out


def _clip_path(clip: ClipRows) -> tuple[str, str]:
    row = clip.video if clip.video is not None else clip.audio[0]
    _, payload = bt.message_of(row["Clip"], "Clip")
    fields = bt.pb_decode(payload, "Clip")
    folder = bt.pb_get(fields, 1, "Clip")[2].decode("utf-8", "replace")
    name = bt.pb_get(fields, 2, "Clip")[2].decode("utf-8", "replace")
    seen = bt.pb_get(fields, 3, "Clip")[2].decode("utf-8", "replace")
    return folder + "\\" + name, seen


def _seen_at(text: str) -> datetime | None:
    """The modification time Resolve recorded (``Clip`` field 3), parsed leniently (any spacing),
    so that the exact text is checked by the field comparison, not here; None if it does not
    parse (the caller reports that, it is never taken as stale)."""
    try:
        return parse_local_time(" ".join(text.split()))
    except ValueError:
        return None


def run(
    projects: list[Path],
    facts: dict[str, MediaFacts],
    *,
    options: LibraryOptions | None = None,
) -> Scoreboard:
    """Author and compare every measured clip of ``projects`` (see the module docstring)."""
    members: dict[KindKey, list[Member]] = defaultdict(list)
    unmatched: list[str] = []
    skipped: list[str] = []
    # Each project once, by its resolved path: a project given twice, or through an alias
    # path, would otherwise offer a second read of every clip as that clip's own template.
    unique: dict[str, Path] = {}
    for given in projects:
        unique.setdefault(str(Path(given).resolve()), Path(given))
    for key, project in unique.items():
        tset: TemplateSet = read_templates(project, options=options)
        skipped.extend(f"{project.name}: {s.name}: {s.reason}" for s in tset.skipped)
        for clip in tset.clips:
            path, seen = _clip_path(clip)
            found = facts.get(path.lower())
            if found is None:
                unmatched.append(f"{project.name}: {clip.media.get('Name')}")
                continue
            seen_at = _seen_at(seen)
            stale = seen_at is not None and seen_at != found.mtime_local
            members[kind_of_facts(found)].append(
                Member(
                    key,
                    clip,
                    found,
                    stale,
                    tset.project_rate,
                    tset.lockable_map_id,
                    seen_unreadable=seen_at is None,
                )
            )
    scores: list[KindScore] = []
    for kind, group in members.items():
        score = KindScore(
            kind,
            members=len(group),
            stale=sum(m.stale for m in group),
            unreadable=sum(m.seen_unreadable for m in group),
        )
        scores.append(score)
        for target in group:
            if target.stale:
                continue
            others = [m for m in group if not _same_clip(m, target)]
            if not others:
                score.self_check = _self_check(kind, target)
                continue
            # A stale member's facts are not the facts its rows were made from: it serves as a
            # template only when no other member is left, and then without its facts. Another
            # clip of the same file (imported twice) is the last choice before that: its facts
            # are the target's, so a value wrongly copied rather than derived would not show.
            file = target.facts.path.casefold()
            others.sort(
                key=lambda m: (
                    m.stale,
                    m.facts.path.casefold() == file,
                    m.project != target.project,
                )
            )
            _check(score, kind, target, others[0])
    scores.sort(key=lambda s: (-s.members, s.kind.label()))
    return Scoreboard(scores, unmatched, skipped)


def _same_clip(a: Member, b: Member) -> bool:
    """The same clip of the same project, whichever read it came from."""
    return a.project == b.project and a.clip.media_id == b.clip.media_id


def _author(kind: KindKey, target: Member, template: Member) -> tuple[ClipRows, tuple[str, ...]]:
    """The target's rows authored from ``template`` (a kind registry of its own, with no
    envelope: the gate measures envelopes, it does not apply them), and the reasons the result
    would not be verified."""
    placement = Placement(
        folder_id=str(target.clip.media["Sm2MpFolder_id"]),
        folder_index=0,
        project_rate=target.rate or Fraction(30000, 1001),
        added_at=0,
        lockable_map_id=target.map_id,
    )
    authored = author(
        target.facts,
        Template(
            template.clip, None if template.stale else template.facts, Path(template.project).name
        ),
        placement,
        kinds={kind: KindStatus("verified", 0)},
        taken=target.clip.ids() if not _same_clip(template, target) else frozenset(),
        allow_unverified=True,
    )
    return ClipRows.from_tables(authored.rows), authored.unverified


def _self_check(kind: KindKey, target: Member) -> str:
    try:
        got, unverified = _author(kind, target, target)
    except MediaPoolError as exc:
        return f"refused: {exc.code}: {exc.message}"[:200]
    compared, diffs, _ = compare(target.clip, got, media_class(target.clip, STILL_EXTENSIONS))
    if not diffs and not unverified:
        return f"equal ({compared} fields)"
    if not diffs:
        return f"unverified: {unverified[0]}"[:200]
    return f"{len(diffs)} of {compared} fields differ, first {diffs[0].path}"


def _check(score: KindScore, kind: KindKey, target: Member, template: Member) -> None:
    try:
        got, unverified = _author(kind, target, template)
    except MediaPoolError as exc:
        score.refused += 1
        key = f"{exc.code}: {exc.message}"[:160]
        score.refusals[key] = score.refusals.get(key, 0) + 1
        return
    for reason in unverified:
        score.unverified[reason[:200]] = score.unverified.get(reason[:200], 0) + 1
    media = media_class(target.clip, STILL_EXTENSIONS)
    compared, diffs, hidden = compare(target.clip, got, media)
    score.checked += 1
    for prop, value in envelope_of(target.facts).items():
        score.envelope.setdefault(prop, set()).add(value)
    score.fields += compared
    score.equal += compared - len(diffs)
    for d in diffs:
        score.differing[d.path] = score.differing.get(d.path, 0) + 1
        score.examples.setdefault(
            d.path, f"{target.facts.name}: Resolve {d.expected} authored {d.got}"[:300]
        )
    for rule, _ in hidden:
        score.hidden[rule.pattern] = score.hidden.get(rule.pattern, 0) + 1


def format_envelope(envelope: dict[str, set[str]] | dict[str, frozenset[str]]) -> str:
    """An envelope as one line, properties in :data:`ENVELOPE_PROPERTIES` order."""
    return "; ".join(
        f"{prop}={','.join(sorted(envelope.get(prop, ())))}" for prop in ENVELOPE_PROPERTIES
    )


def format_scoreboard(board: Scoreboard) -> str:
    """The scoreboard as text: per kind, files checked, fields equal, fields differing with one
    example each. Names of files are shown only in the examples."""
    lines: list[str] = []
    for s in board.kinds:
        lines.append(
            f"{s.status:9} {s.kind.label()}\n"
            f"          members {s.members}, stale {s.stale}, checked {s.checked}, "
            f"refused {s.refused}, "
            f"fields {s.fields}, equal {s.equal}"
        )
        for path, count in sorted(s.differing.items()):
            lines.append(f"          DIFF {path} x{count}: {s.examples[path]}")
        for reason, count in sorted(s.refusals.items()):
            lines.append(f"          REFUSED x{count}: {reason}")
        for reason, count in sorted(s.unverified.items()):
            lines.append(f"          UNVERIFIED x{count}: {reason}")
        if s.unreadable:
            lines.append(f"          Clip #3 unreadable (compared anyway): {s.unreadable}")
        if s.envelope:
            lines.append("          envelope: " + format_envelope(s.envelope))
        if s.self_check is not None:
            lines.append(f"          self-check (not a verification): {s.self_check}")
        for pattern, count in sorted(s.hidden.items()):
            lines.append(f"          excluded and differing: {pattern} in {count} field(s)")
    lines.append(
        f"clips without facts: {len(board.unmatched)}; not templates: {len(board.skipped)}"
    )
    return "\n".join(lines)
