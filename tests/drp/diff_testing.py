"""Shared assertions and fixture facts for the ``aar.drp.diff`` tests (test-only code)."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from drp_edit import add_carrier, carrier, edit_zip, markers_fields_blob, member_named

from aar.drp.diff import Change, ProjectDiff, diff_projects
from aar.drp.reader import read_drp

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "drp"
ALL = ["empty-project.drp", "empty-project-r19.drp", "media-clip-h264.drp", "media-clip-r19.drp"]
H264 = FIXTURES / "media-clip-h264.drp"

# DbIds inside media-clip-h264.drp
V1_ITEM = "ad1fd562-da22-4b1f-b79f-f3106f9c2ac5"
A1_ITEM = "78f4666d-9ffc-4c26-a9cd-70bf093a36fa"
TIMELINE = "f75533e4-a99c-4cd9-8280-1f756f0d3bef"
SEQUENCE = "12d57950-0b88-4ccb-8cc5-f7638e046ebd"
CLIP = "2cd8eb8a-ae1b-4fbf-877a-c8dcda96e32b"
SEQ_MEMBER = member_named(H264, "SeqContainer/")
POOL = "MediaPool/Master/MpFolder.xml"
V1 = f'DbId="{V1_ITEM}"'.encode()

SWAP = {"added": "removed", "removed": "added", "opaque-added": "opaque-removed"}
SWAP |= {v: k for k, v in SWAP.items()}


def diff(a: Path, b: Path, *, gate: bool = False) -> ProjectDiff:
    return diff_projects(read_drp(a), read_drp(b), gate=gate)


def sig(c: Change) -> tuple[Any, ...]:
    """Everything structural about a change (not the wording, which names the newer side)."""
    return (
        c.section,
        c.kind,
        c.object_kind,
        c.object_id,
        c.match,
        c.timeline_id,
        c.field_name,
        json.dumps(c.to_dict()["old"], sort_keys=True),
        json.dumps(c.to_dict()["new"], sort_keys=True),
        tuple(sorted((c.delta or {}).items())),
        c.where_a,
        c.where_b,
        c.byte_count,
    )


def mirror_sig(c: Change) -> tuple[Any, ...]:
    """What the same change must look like when A and B are swapped (written independently)."""
    return (
        c.section,
        SWAP.get(c.kind, c.kind),
        c.object_kind,
        c.object_id,
        c.match,
        c.timeline_id,
        c.field_name,
        json.dumps(c.to_dict()["new"], sort_keys=True),
        json.dumps(c.to_dict()["old"], sort_keys=True),
        tuple(sorted((k, -v) for k, v in (c.delta or {}).items())),
        c.where_b,
        c.where_a,
        c.byte_count,
    )


def assert_mirrors(a: Path, b: Path, *, gate: bool = False) -> ProjectDiff:
    ab, ba = diff(a, b, gate=gate), diff(b, a, gate=gate)
    assert sorted(map(sig, ba.changes), key=repr) == sorted(map(mirror_sig, ab.changes), key=repr)
    return ab


def edited(
    tmp_path: Path, name: str, edits: dict[str, Callable[[bytes], bytes]], src: Path | None = None
) -> Path:
    return edit_zip(src or H264, tmp_path / name, edits)


def only_change(a: Path, b: Path, *, gate: bool = False) -> Change:
    d = assert_mirrors(a, b, gate=gate)
    assert len(d.changes) == 1, d.to_text()
    return d.changes[0]


MARKER_BLOB_ID = "0b0b0b0b-1111-2222-3333-444444444444"


def marker_file(tmp_path: Path, name: str, *entries: bytes) -> Path:
    element = carrier(
        "0b0b0b0b-1111-2222-3333-444444444444", SEQUENCE, markers_fields_blob(*entries)
    )
    return edited(tmp_path, name, {"project.xml": add_carrier(element)})
