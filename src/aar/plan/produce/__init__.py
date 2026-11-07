"""Plan producers: analysis outputs and a keep list to an EditPlan v2 that lints clean.

Cold and deterministic: the only media access is the source-audio decode for pauses and joins
(``aar.plan.produce.audio``). The transforms are pure functions, each in its own module:

1. ``pauses``: long pauses (source-audio quiet, never transcript gaps) shrink to a beat.
2. ``joins``: every cut's in point starts in source-audio quiet (QA's join-quiet test).
3. ``slivers``: shots under ``min_shot_s`` fold into a neighbour of their keep range.
4. ``ending``: with a laugh map, the content ends just after the final laugh's onset.
5. ``camera``: punch-ins on jump cuts, and a short's steady camera path over the face track.

Chronological order and no dialogue splice are invariants (``timeline``). Parameters are the
profile's ``[produce]`` block; see docs/DESIGN.md "Plan producers (S13d)". Automatic
speaker-driven camera switching is not here: the cameras come from a schedule, or the profile's
default camera.
"""

from aar.plan.produce.build import ProduceResult, apply_sync, bind_faces, check_profile, produce
from aar.plan.produce.errors import (
    InvariantError,
    LintGateError,
    ProduceError,
    ProduceInputError,
)
from aar.plan.produce.inputs import (
    CameraSchedule,
    FaceTrack,
    KeepList,
    SyncReport,
    load_faces,
    load_keep,
    load_schedule,
    load_stub,
    load_sync,
    read_laughs,
    read_transcript,
)
from aar.plan.produce.report import PlannerWarning, ProduceReport

__all__ = [
    "CameraSchedule",
    "FaceTrack",
    "InvariantError",
    "KeepList",
    "LintGateError",
    "PlannerWarning",
    "ProduceError",
    "ProduceInputError",
    "ProduceReport",
    "ProduceResult",
    "SyncReport",
    "apply_sync",
    "bind_faces",
    "check_profile",
    "load_faces",
    "load_keep",
    "load_schedule",
    "load_stub",
    "load_sync",
    "produce",
    "read_laughs",
    "read_transcript",
]
