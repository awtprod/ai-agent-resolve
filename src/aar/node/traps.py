"""Known Resolve scripting traps, each with a stable id, its symptom, its cause and its fix.

Ops cite these ids in their errors and results (``OpError.trap_id``, ``Note.trap_id``), so an
agent that hits one can look the fix up instead of rediscovering it. Every entry was observed
on DaVinci Resolve Studio 21.1; ``verified`` says how.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Trap:
    id: str
    title: str
    symptom: str
    cause: str
    fix: str
    verified: str = "live, Resolve 21.1"

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


_ALL: tuple[Trap, ...] = (
    Trap(
        id="append-endframe-exclusive",
        title="AppendClipInfo endFrame is exclusive",
        symptom="Appended items are one frame shorter than the range that was meant.",
        cause="MediaPool.AppendToTimeline treats endFrame as exclusive: [startFrame, endFrame).",
        fix="Pass half-open ranges straight through: endFrame is the first frame not wanted. "
        "Do not subtract 1.",
    ),
    Trap(
        id="append-fps-short",
        title="Rate-converted appends can land one frame short",
        symptom="A source at a different frame rate (e.g. 24 fps into a 29.97 timeline) lands "
        "1 frame short of its converted length, leaving a 1-frame gap before the next item.",
        cause="AppendClipInfo start/end frames are in the source's frame rate, and the "
        "conversion to timeline frames can round down.",
        fix="Read every appended item back. If it is 1 frame short, delete it and re-append "
        "with endFrame + 1, then read it back again.",
    ),
    Trap(
        id="append-still-ignores-range",
        title="Appended stills ignore their range",
        symptom="A still image appends with a 150-frame default length and an odd offset, "
        "whatever startFrame/endFrame say.",
        cause="AppendToTimeline does not apply source ranges to still images.",
        fix="Do not append stills. Render the still to a ProRes 4444 MOV of the wanted length "
        "and append that.",
    ),
    Trap(
        id="append-ungraded",
        title="Appended items come in ungraded",
        symptom="Items placed with AppendToTimeline have no grade, even when the same clip is "
        "graded on another timeline.",
        cause="AppendToTimeline creates fresh timeline items with an empty node graph.",
        fix="Copy the grade from a graded reference item of the same camera with "
        "TimelineItem.CopyGrades([...]); this works across timelines. CopyGrades replaces the "
        "whole node graph, so re-apply per-clip LUTs afterwards.",
    ),
    Trap(
        id="av-append-video-only",
        title="An audio+video append of BRAW returns only the video item",
        symptom="AppendToTimeline returns one item for a BRAW clip appended with audio and "
        "video, although an audio item was also placed.",
        cause="The returned list only holds the video item for BRAW 'av' appends.",
        fix="Find the audio items by walking Timeline.GetItemListInTrack('audio', n) for items "
        "of the same media at the same start.",
    ),
    Trap(
        id="volume-key",
        title="The audio gain key is AudioVolume",
        symptom="Setting 'Volume' on a timeline item silently returns False and nothing changes.",
        cause="The item property for clip gain is 'AudioVolume' (dB, -100 to 30); keys are "
        "validated before anything is applied.",
        fix="Use SetProperties({'AudioVolume': dB}). Check every property setter's return "
        "value and read the property back.",
    ),
    Trap(
        id="setproperty-after-color",
        title="Property setters fail after returning to Edit from Color",
        symptom="After switching from the Color page back to Edit, every item property setter "
        "returns False.",
        cause="A page-state problem in Resolve; not documented.",
        fix="OpenPage to another page and back, then retry once.",
    ),
    Trap(
        id="rcm-input-colorspace",
        title="Imported Rec.709 media takes the camera-log input default",
        symptom="In a colour-managed project, Rec.709 media imported into the media pool "
        "looks wrong because it is decoded as camera log.",
        cause="New media inherits the project's input colour space default, which is set for "
        "the camera originals.",
        fix="Set the clip property 'Input Color Space' on each affected media-pool item, "
        "listed explicitly. Never on BRAW clips, never by sweeping a bin tree, and never by "
        "changing the project's colour settings.",
    ),
    Trap(
        id="custom-res-resets-colour-science",
        title="A custom timeline resolution resets the timeline's colour science",
        symptom="After giving one timeline its own resolution (e.g. a vertical cut), that "
        "timeline renders with plain DaVinci YRGB colour.",
        cause="Turning on per-timeline settings for a resolution change resets the timeline's "
        "colour science to plain davinciYRGB.",
        fix="Record colorScienceMode, isAutoColorManage, rcmPresetMode, colorSpaceTimeline, "
        "colorSpaceOutput, inputDRT, outputDRT, useInverseDRT, useColorSpaceAwareGradingTools "
        "and the other timeline colour settings first, set them back afterwards, and read "
        "them back.",
    ),
    Trap(
        id="multicam-angles",
        title="Multicam angles cannot be set, and appends lose them",
        symptom="Appended multicam clips come in on the wrong angle; the API has no call to "
        "choose an angle.",
        cause="The scripting API does not expose multicam angle selection.",
        fix="Do not append multicam clips. Duplicate the timeline to a scratch copy, flatten "
        "it (TimelineItem.FlattenMulticam), and measure the constant offset of each camera "
        "from the flattened items; then append the camera files with those offsets.",
    ),
    Trap(
        id="deliver-clobber",
        title="Loading a render preset clobbers the Deliver settings",
        symptom="The Deliver page's target folder, file name and format are replaced after a "
        "render preset is loaded.",
        cause="LoadRenderPreset replaces every Deliver setting, and the API has no "
        "GetRenderSettings to read them first.",
        fix="Save the Deliver settings as a temporary render preset before and load it back "
        "afterwards. The broker does this for every borrowed context.",
    ),
    Trap(
        id="render-uses-original-media",
        title="Renders use original media, not proxies",
        symptom="A timeline that plays fine fails to export with 'frames offline'.",
        cause="Playback can come from proxies, but a render always reads the original media.",
        fix="Before rendering, check every item's media 'Online Status'; relink or sync the "
        "original media rather than relying on proxies.",
    ),
    Trap(
        id="pm-folder-scoped-load",
        title="LoadProject only looks in the current Project Manager folder",
        symptom="LoadProject(name) returns None for a project that exists, or opens a "
        "different project with the same name.",
        cause="ProjectManager.LoadProject resolves the name inside the current Project Manager "
        "folder only, and names are not unique across folders.",
        fix="Address projects by ProjectLocator: navigate to the folder path from the root "
        "(GotoRootFolder, then OpenFolder per segment), load, and verify the project's unique "
        "id.",
    ),
    Trap(
        id="source-frame-readback-rounding",
        title="Source frame readback can be one frame off",
        symptom="TimelineItem.GetSourceStartFrame/GetSourceEndFrame can differ by one frame "
        "from the in point used and from the item's duration (end - start is sometimes "
        "duration - 1, duration or duration + 1).",
        cause="Source positions are kept as times and rounded to frames on readback.",
        fix="Measure placement on timeline frames (GetStart/GetEnd, end exclusive). Treat "
        "source readback as accurate to +/- 1 frame and report the measured value.",
        verified="live, Resolve 21.1, read-only probe of four items on one timeline",
    ),
)

TRAPS: dict[str, Trap] = {trap.id: trap for trap in _ALL}


def get_trap(trap_id: str) -> Trap:
    """The trap with ``trap_id``. Raises ``KeyError`` for an unknown id."""
    return TRAPS[trap_id]


def trap_ids() -> list[str]:
    return list(TRAPS)
