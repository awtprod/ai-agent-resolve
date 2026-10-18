"""The DRP edit plan: a typed, JSON-serialisable list of whitelisted edits.

``aar drp edit-schema`` prints the JSON Schema of :class:`DrpEditPlan`. Every edit names its
target by DbId (``id``, as ``aar drp inspect --json`` shows it), or by its current name when that
name is unique. Edits apply in order; a plan whose edits touch the same value twice is refused.

Whitelisted ``op`` values: ``rename``, ``path_map``, ``marker_add``, ``marker_update`` and
``marker_delete``. ``metadata`` and ``fader_keyframes`` are known but deferred (see
:data:`DEFERRED_OPS`), and anything else is refused as not whitelisted.

Marker frames: a timeline marker's ``frame`` counts from the first frame of the timeline (frame
0 is the timeline start). An item marker's frame space is not verified yet (see
:class:`aar.drp.model.Marker`).
"""

from __future__ import annotations

import json
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

__all__ = [
    "DEFERRED_OPS",
    "MAX_PLAN_BYTES",
    "PLAN_VERSION",
    "WHITELISTED_OPS",
    "DrpEditPlan",
    "Edit",
    "MarkerAdd",
    "MarkerDelete",
    "MarkerFields",
    "MarkerUpdate",
    "PathMap",
    "PathPair",
    "PlanProblem",
    "Rename",
    "parse_plan",
]

PLAN_VERSION = 1
MAX_PLAN_BYTES = 1024 * 1024
MAX_EDITS = 10_000
MAX_NAME = 1024
MAX_MARKER_TEXT = 64 * 1024
_FRAME_MAX = 2**31 - 1

WHITELISTED_OPS = ("rename", "path_map", "marker_add", "marker_update", "marker_delete")
DEFERRED_OPS: dict[str, str] = {
    "metadata": (
        "deferred: the reader decodes no user-editable clip metadata. Clip Name is a rename "
        "(target 'clip'). Comments, keywords, scene/shot/take and camera fields live in "
        "BtMetadataLockableBlob, which the reader does not decode, so the post-write gate "
        "could not vouch for an edit; the technical fields it does decode (frames, frame rate, "
        "resolution, codec, StartFrame) describe the media file, and StartFrame sits next to "
        "an undecoded Timecode string in the same Time blob"
    ),
    "fader_keyframes": (
        "deferred: drp_tools' keyframe writer is not available (only its codecs and format "
        "notes are), its notes leave the tick encoding, the track mapping and the AdrDatabase "
        "checksum open, and the reader does not decode FLStudioModelBA, so the gate could not "
        "vouch for a write"
    ),
}

MarkerColor = Literal[
    "Blue",
    "Cyan",
    "Green",
    "Yellow",
    "Red",
    "Pink",
    "Purple",
    "Fuchsia",
    "Rose",
    "Lavender",
    "Sky",
    "Mint",
    "Lemon",
    "Sand",
    "Cocoa",
    "Cream",
]
Owner = Literal["timeline", "item"]


class _Model(BaseModel):
    """Unknown keys are errors, so a typo in a plan is refused rather than ignored."""

    model_config = ConfigDict(extra="forbid")


class Rename(_Model):
    """Rename the project, a timeline, a bin, a media pool clip or a timeline item.

    ``project`` needs no selector. The others take exactly one of ``id`` (DbId; for a timeline
    the timeline's ``id`` from ``aar drp inspect --json``) and ``name`` (current name, which must
    be unique). ``expect_name``, when given, must equal the current name.
    """

    op: Literal["rename"]
    target: Literal["project", "timeline", "bin", "clip", "item"]
    id: str | None = Field(default=None, min_length=1, max_length=256)
    name: str | None = Field(default=None, min_length=1, max_length=MAX_NAME)
    new_name: str = Field(min_length=1, max_length=MAX_NAME)
    expect_name: str | None = Field(default=None, max_length=MAX_NAME)

    @model_validator(mode="after")
    def _selector(self) -> Rename:
        given = (self.id is not None) + (self.name is not None)
        if self.target == "project" and given:
            raise ValueError("a project rename takes no id or name")
        if self.target != "project" and given != 1:
            raise ValueError(f"a {self.target} rename needs exactly one of id and name")
        return self


class PathPair(_Model):
    """One row of a path table: a Windows prefix and the Mac prefix it corresponds to."""

    windows: str = Field(min_length=2, max_length=4096, description="e.g. 'E:' or '\\\\nas\\share'")
    mac: str = Field(min_length=2, max_length=4096, description="e.g. '/Volumes/Media'")


class PathMap(_Model):
    """Relink media: rewrite every decoded media path through the table, in one direction."""

    op: Literal["path_map"]
    direction: Literal["windows_to_mac", "mac_to_windows"]
    table: list[PathPair] = Field(min_length=1, max_length=256)


class MarkerFields(_Model):
    """New values for a marker; fields left out keep their value."""

    frame: int | None = Field(default=None, ge=0, le=_FRAME_MAX)
    color: MarkerColor | None = None
    name: str | None = Field(default=None, max_length=MAX_NAME)
    note: str | None = Field(default=None, max_length=MAX_MARKER_TEXT)
    duration: int | None = Field(default=None, ge=1, le=_FRAME_MAX)
    custom_data: str | None = Field(default=None, max_length=MAX_MARKER_TEXT)

    @model_validator(mode="after")
    def _something(self) -> MarkerFields:
        if not self.model_fields_set:
            raise ValueError("marker_update needs at least one field in 'set'")
        for key in self.model_fields_set:
            if getattr(self, key) is None:
                raise ValueError(f"'{key}' cannot be null")
        return self


class _MarkerRef(_Model):
    owner: Owner
    owner_id: str = Field(min_length=1, max_length=256)
    frame: int = Field(ge=0, le=_FRAME_MAX)


class MarkerAdd(_MarkerRef):
    """Add a marker at ``frame`` (no marker may be there yet)."""

    op: Literal["marker_add"]
    color: MarkerColor = "Blue"
    name: str = Field(default="", max_length=MAX_NAME)
    note: str = Field(default="", max_length=MAX_MARKER_TEXT)
    duration: int = Field(default=1, ge=1, le=_FRAME_MAX)
    custom_data: str = Field(default="", max_length=MAX_MARKER_TEXT)


class MarkerUpdate(_MarkerRef):
    """Change the marker at ``frame``; ``set.frame`` moves it."""

    op: Literal["marker_update"]
    set: MarkerFields


class MarkerDelete(_MarkerRef):
    """Delete the marker at ``frame``."""

    op: Literal["marker_delete"]


Edit = Annotated[
    Rename | PathMap | MarkerAdd | MarkerUpdate | MarkerDelete, Field(discriminator="op")
]


class DrpEditPlan(_Model):
    """A list of whitelisted edits for one ``.drp``/``.drt`` file.

    ``input_sha256`` (optional) refuses the plan when the input is not that exact file.
    ``max_project_version`` (optional) is the target host's DbPrjVer: an input stamped newer is
    refused, because the writer never restamps a file.
    """

    plan_version: Literal[1] = PLAN_VERSION
    input_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    max_project_version: int | None = Field(default=None, ge=0, le=10_000)
    edits: list[Edit] = Field(default_factory=list, max_length=MAX_EDITS)


class PlanProblem(ValueError):
    """The plan is refused before anything is read. ``code`` is stable, ``details`` structured."""

    def __init__(self, code: str, message: str, details: list[Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.details = details or []


def parse_plan(text: str | bytes) -> DrpEditPlan:
    """Parse and validate a plan; raises :class:`PlanProblem` with a stable code.

    Codes: ``plan-too-large``, ``plan-not-json``, ``edit-deferred`` (a known op that is not
    implemented yet), ``edit-not-whitelisted`` (any other op) and ``plan-invalid``.
    """
    size = len(text.encode("utf-8") if isinstance(text, str) else text)
    if size > MAX_PLAN_BYTES:
        raise PlanProblem("plan-too-large", f"the plan is {size} bytes, over {MAX_PLAN_BYTES}")
    try:
        raw = json.loads(text)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise PlanProblem("plan-not-json", f"the plan is not JSON: {exc}") from exc
    edits = raw.get("edits") if isinstance(raw, dict) else None
    if isinstance(edits, list):
        deferred, foreign = [], []
        for i, edit in enumerate(edits):
            op = edit.get("op") if isinstance(edit, dict) else None
            if op in DEFERRED_OPS:
                deferred.append({"edit": i, "op": op, "reason": DEFERRED_OPS[op]})
            elif op not in WHITELISTED_OPS:
                foreign.append({"edit": i, "op": op})
        if deferred:
            raise PlanProblem(
                "edit-deferred",
                f"{len(deferred)} edit(s) use an op that is not implemented yet",
                deferred,
            )
        if foreign:
            raise PlanProblem(
                "edit-not-whitelisted",
                f"{len(foreign)} edit(s) are not whitelisted; allowed ops: "
                + ", ".join(WHITELISTED_OPS),
                foreign,
            )
    try:
        return DrpEditPlan.model_validate(raw)
    except ValidationError as exc:
        details = json.loads(
            exc.json(include_url=False, include_context=False, include_input=False)
        )
        raise PlanProblem("plan-invalid", "the plan failed validation", details) from exc
