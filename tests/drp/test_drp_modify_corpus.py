"""Opt-in S8 checks over the private corpus (``$AAR_DRP_CORPUS``); see ``test_drp_corpus``.

Test ids and printed lines use numbers, never file or project names. The reader's budgets
apply throughout (every file is read with ``read_drp``'s defaults).
"""

from __future__ import annotations

import json
import os
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
from drp_edit import edit_zip

from aar.drp import blobs
from aar.drp.diff import diff_projects
from aar.drp.modify import compare_round_trip, encode, parse_plan, verify_write, zipraw
from aar.drp.modify.engine import Expected, change_key
from aar.drp.modify.markers import MarkerList
from aar.drp.modify.write import edit_file
from aar.drp.reader import read_drp
from aar.drp.scan import scan

CORPUS = os.environ.get("AAR_DRP_CORPUS")
SUFFIXES = {".drp", ".drt", ".drb"}
FILES = sorted(p for p in Path(CORPUS).rglob("*") if p.suffix.lower() in SUFFIXES) if CORPUS else []
IDS = [f"corpus-{i:03d}" for i in range(len(FILES))]
needs_corpus = pytest.mark.skipif(not CORPUS, reason="set AAR_DRP_CORPUS to run the corpus checks")


def _zip_files() -> list[Path]:
    return [p for p in FILES if zipfile.is_zipfile(p)]


@needs_corpus
@pytest.mark.parametrize("path", FILES, ids=IDS)
def test_no_op_rewrite_returns_the_input_bytes(path: Path, tmp_path: Path) -> None:
    data = path.read_bytes()
    assert zipraw.rewrite(data, {}) == data
    drp = read_drp(path)
    out = tmp_path / f"out{path.suffix}"
    outcome = edit_file(path, parse_plan('{"edits": []}'), out)
    if outcome.status == "refused":
        # Only inputs the gate can never pass for are refused, and only for that reason.
        assert [r.code for r in outcome.refusals] == ["input-has-warnings"]
        assert {i.code for i in drp.issues if i.severity != "info"} <= {"blob-partial"}
        return
    assert out.read_bytes() == data


def _hex_leaves(path: Path) -> list[tuple[str, str, bytes]]:
    out = []
    with zipfile.ZipFile(path) as zf:
        for name in zf.namelist():
            if not name.endswith(".xml"):
                continue
            doc = scan(zf.read(name), name)
            if doc.root is None:
                continue
            for el in doc.root.iter():
                if not el.children and blobs.is_blob_tag(el.tag) and el.text:
                    out.append((name, el.tag, el.inner_raw))
    return out


@needs_corpus
@pytest.mark.parametrize("path", FILES, ids=IDS)
def test_decode_encode_is_the_identity_for_every_blob_kind_we_write(path: Path) -> None:
    """Hex text, Clip envelopes (and their zstd frames), and marker lists, across the corpus."""
    if not zipfile.is_zipfile(path):
        pytest.skip("not a zip")
    budget = blobs.DecodeBudget()
    counts: Counter[str] = Counter()
    for _name, tag, inner in _hex_leaves(path):
        try:
            raw = blobs.parse_hex(inner.decode("ascii"))
        except (blobs.BlobError, UnicodeDecodeError):
            continue
        assert encode.hex_like(inner, raw) == inner
        counts["hex"] += 1
        if tag == "Clip":
            env = blobs.parse_envelope(raw)
            assert env is not None and env.version is not None
            payload = blobs.unwrap_message(raw, budget=budget)
            again = encode.envelope(raw, payload)
            assert blobs.unwrap_message(again) == payload
            if env.codec == "zstd":
                off, _ = blobs._frame_header(env.payload)
                header = int.from_bytes(env.payload[off : off + 3], "little")
                if (header >> 1) & 3 == 0 and header & 1:  # one raw block, as we write them
                    assert again == raw
                    counts["clip-raw-identical"] += 1
            else:
                assert again == raw
            fields = blobs.decode_protobuf(payload, nested_depth=0)
            first = next(f for f in fields if f.number == 1)
            assert isinstance(first.value, bytes)
            assert encode.replace_field(payload, first, first.value) == payload
            counts["clip"] += 1
        elif tag == "FieldsBlob" and b"0042006c006f0062004400610074" in inner:  # "BlobData"
            try:
                ml = MarkerList.decode(raw, budget)
            except (encode.EncodeError, blobs.BlobError):
                continue  # not a marker list (metadata blobs carry BlobData too)
            written = ml.encode()
            again = MarkerList.decode(written, budget)
            assert again.values() == ml.values()
            assert [e.raw for e in again.entries] == [e.raw for e in ml.entries]
            if blobs.parse_envelope(ml.blob_data).codec != "zstd":  # type: ignore[union-attr]
                assert written == raw
            for entry in ml.entries:
                assert entry.updated({}).raw == entry.raw
            counts["markers"] += len(ml.entries)
    print(f"s8-identity file={FILES.index(path):03d} {dict(sorted(counts.items()))}")


def _plan_for(path: Path) -> dict[str, Any]:
    """Every whitelisted edit kind that applies to this file."""
    d = read_drp(path)
    edits: list[dict[str, Any]] = []
    if d.project:
        edits.append({"op": "rename", "target": "project", "new_name": "S8 project"})
    timelines = [t for t in d.timelines if t.id and t.media_clip_id]
    if timelines:
        edits.append(
            {"op": "rename", "target": "timeline", "id": timelines[0].id, "new_name": "S8 timeline"}
        )
    owned = {t.media_clip_id for t in d.timelines}
    clips = [m for m in d.media if m.id not in owned and m.name]
    if clips:
        edits.append({"op": "rename", "target": "clip", "id": clips[-1].id, "new_name": "S8 clip"})
    bins = [b for b in d.bins if b.parent_id and b.id != b.member]
    if bins:
        edits.append({"op": "rename", "target": "bin", "id": bins[0].id, "new_name": "S8 bin"})
    paths = [m.path for m in d.media if m.path]
    drives = sorted({p[0].upper() for p in paths if re.match(r"^[A-Za-z]:", p)})
    volumes = sorted({p.split("/")[2] for p in paths if p.startswith("/Volumes/")})
    if drives:
        table = [{"windows": f"{x}:", "mac": f"/Volumes/Drive{x}"} for x in drives]
        edits.append({"op": "path_map", "direction": "windows_to_mac", "table": table})
    elif volumes:
        table = [
            {"windows": f"{chr(69 + k)}:", "mac": f"/Volumes/{v}"}
            for k, v in enumerate(volumes[:20])
        ]
        edits.append({"op": "path_map", "direction": "mac_to_windows", "table": table})
    marked = [t for t in d.timelines if t.markers]
    if marked:
        t = marked[0]
        frames = sorted({m.frame for m in t.markers})
        edits.append(
            {
                "op": "marker_update",
                "owner": "timeline",
                "owner_id": t.id,
                "frame": frames[0],
                "set": {"name": "S8 marker", "color": "Red", "note": "né"},
            }
        )
        if len(frames) > 1:
            edits.append(
                {"op": "marker_delete", "owner": "timeline", "owner_id": t.id, "frame": frames[-1]}
            )
        edits.append(
            {
                "op": "marker_add",
                "owner": "timeline",
                "owner_id": t.id,
                "frame": frames[-1] + 7,
                "name": "S8 new",
                "custom_data": "cd",
            }
        )
    items = [it for t in d.timelines for tr in t.tracks for it in tr.items if it.markers and it.id]
    if items:
        it = items[0]
        frame = it.markers[0].frame
        edits.append(
            {
                "op": "marker_update",
                "owner": "item",
                "owner_id": it.id,
                "frame": frame,
                "set": {"frame": frame + 1_000_003},
            }
        )
    return {"edits": edits}


@needs_corpus
@pytest.mark.parametrize("path", FILES, ids=IDS)
def test_every_edit_kind_round_trips_on_real_files(path: Path, tmp_path: Path) -> None:
    """Edit, re-read, and require the diff to show exactly the intended changes."""
    if not zipfile.is_zipfile(path):
        pytest.skip("not a zip")
    drp = read_drp(path)
    spec = _plan_for(path)
    out = tmp_path / f"out{path.suffix}"
    outcome = edit_file(path, parse_plan(json.dumps(spec)), out)
    noisy = {i.code for i in drp.issues if i.severity in ("warning", "error")}
    if noisy:
        assert noisy == {"blob-partial"}
        assert [r.code for r in outcome.refusals] == ["input-has-warnings"] and not out.exists()
        return
    assert outcome.status == "written", [r.to_dict() for r in outcome.refusals]
    # Re-read and diff here, outside the writer: exactly the expected changes, gate passing.
    d = diff_projects(read_drp(path), read_drp(out), gate=True)
    assert d.gate.passed, d.gate.reasons
    assert Counter(change_key(c) for c in d.changes) == Counter(
        Expected(**e).key() for e in outcome.expected
    )
    assert read_drp(out).versions == drp.versions or (
        (read_drp(out).versions.app_version, read_drp(out).versions.project_version)
        == (drp.versions.app_version, drp.versions.project_version)
    )
    kinds = Counter(e["op"] for e in spec["edits"])
    print(
        f"s8-edit file={FILES.index(path):03d} edits={dict(kinds)} changes={len(outcome.changes)}"
    )


@needs_corpus
def test_partial_blob_files_refuse_while_their_clean_siblings_edit(tmp_path: Path) -> None:
    partial = [p for p in _zip_files() if "blob-partial" in {i.code for i in read_drp(p).issues}]
    assert partial, "the corpus has blob-partial files"
    clean = next(
        p
        for p in _zip_files()
        if not [i for i in read_drp(p).issues if i.severity != "info"] and p.name.startswith("dat-")
    )
    spec = parse_plan(
        json.dumps({"edits": [{"op": "rename", "target": "project", "new_name": "x"}]})
    )
    for p in partial:
        outcome = edit_file(p, spec, tmp_path / f"{FILES.index(p):03d}.drp")
        assert [r.code for r in outcome.refusals] == ["input-has-warnings"]
        assert outcome.refusals[0].details == {"codes": ["blob-partial"]}
    assert edit_file(clean, spec, tmp_path / "clean.drp").status == "written"
    print(f"s8-partial refused={len(partial)} clean-sibling=written")


@needs_corpus
def test_input_has_warnings_excludes_16_of_the_64_corpus_files(tmp_path: Path) -> None:
    """docs/DESIGN.md (Modify, S8) states these numbers."""
    spec = parse_plan(
        json.dumps({"edits": [{"op": "rename", "target": "project", "new_name": "x"}]})
    )
    refused, codes = 0, Counter()
    for i, p in enumerate(FILES):
        outcome = edit_file(p, spec, tmp_path / f"{i:03d}.drp", dry_run=True)
        if "input-has-warnings" in {r.code for r in outcome.refusals}:
            refused += 1
            codes.update(outcome.refusals[0].details["codes"])
    assert (len(FILES), refused, dict(codes)) == (64, 16, {"blob-partial": 16})
    print(f"s8b-warnings files={len(FILES)} refused={refused} codes={dict(codes)}")


def _pair_002_003() -> tuple[Path, Path]:
    a, b = FILES[2], FILES[3]
    va, vb = read_drp(a).versions, read_drp(b).versions
    if not (va and vb and va.app_version == vb.app_version == "20.0.0.0049"):
        pytest.skip("corpus files 002/003 are not the Resolve 20 round-trip pair")
    return a, b


def _multicam_removed(a: Path, b: Path) -> list[Expected]:
    ta = {t.id: t for t in read_drp(a).timelines}
    tb = {t.id for t in read_drp(b).timelines}
    gone = [t for k, t in ta.items() if k not in tb]
    assert len(gone) == 1 and gone[0].kind == "multicam"
    return [Expected("timelines", "removed", "timeline", gone[0].id, None, gone[0].name, None, 0)]


@needs_corpus
def test_round_trip_allow_list_on_the_resolve_pair(tmp_path: Path) -> None:
    a, b = _pair_002_003()
    expected = _multicam_removed(a, b)
    result = compare_round_trip(a, b, expected)
    assert result.ok, result.to_dict()
    rules = Counter(rule for rule, _ in result.allowed)
    assert rules == Counter(
        {
            "playhead": 3,
            "ptzr-preset-dbid": 3,
            "db-saved-time": 2,
            "project-age": 1,
            "settings-save-date": 1,
            "bin-save-stamp": 1,
            "last-modified": 1,
        }
    )
    assert sum(n for r, n in rules.items() if r != "last-modified") == 11
    print(f"s8-roundtrip pair=002->003 allowed={dict(sorted(rules.items()))}")
    # Without the intended edit it fails; our own writer check never uses the allow-list.
    assert [r.code for r in compare_round_trip(a, b).refusals] == ["unintended-change"]
    members = set(zipfile.ZipFile(a).namelist())
    assert not verify_write(a, b, expected, members).ok


@needs_corpus
@pytest.mark.parametrize(
    ("member", "old", "new"),
    [
        ("MediaPool/Master/MpFolder.xml", b"<ID>100</ID>", b"<ID>101</ID>"),
        (
            "MediaPool/Master/MpFolder.xml",
            b"<IsCustom>true</IsCustom>",
            b"<IsCustom>false</IsCustom>",
        ),
        ("project.xml", b"<DbSavedTime>", b"<DbSavedTime>9x"),
    ],
    ids=["preset-id", "preset-flag", "saved-time-not-decimal"],
)
def test_any_other_opaque_change_still_fails_on_the_pair(
    tmp_path: Path, member: str, old: bytes, new: bytes
) -> None:
    a, b = _pair_002_003()
    with zipfile.ZipFile(b) as zf:
        data = zf.read(member)
    assert old in data
    mutated = edit_zip(b, tmp_path / "mutated.drp", {member: lambda d: d.replace(old, new, 1)})
    result = compare_round_trip(a, mutated, _multicam_removed(a, b))
    assert [r.code for r in result.refusals] == ["unexplained-opaque"]
