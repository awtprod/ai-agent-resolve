"""``aar.plan.render``: an EditPlan v2 rendered to MP4 with ffmpeg on the hub, no Resolve (S18b).

:func:`render_plan` lints the plan (and refuses one with lint errors, as the FCPXML export does),
resolves every used source to a probed hub file (:mod:`~aar.plan.render.media`), builds the FCPXML
export's own :class:`~aar.plan.export.layout.Layout` with those hub paths (so every placement,
handle, alignment and fade is S18's), and renders it in a work dir beside the output:

1. the mix (:func:`~aar.plan.render.graph.mix_graph`) to a 32-bit float WAV of the exact length;
2. its loudness to the profile's target (:mod:`~aar.plan.render.loudness`);
3. the picture (:func:`~aar.plan.render.graph.video_graph`, from a ``-filter_complex_script``)
   encoded with that sound: ``review`` is H.264 High yuv420p + AAC with faststart, ``master`` a
   higher-quality H.264, or ProRes 422 HQ in a .mov when ``[render] master_codec`` says so;
4. :func:`aar.qa.check` on the result with the plan's kind;
5. the file and its report (``<stem>.render.json``) placed as one pair by S18's pair writer.

The work dir is removed afterwards, on success and failure. Every ffmpeg call has a timeout
scaled to the timeline's length, passes argument lists only, and keeps a bounded stderr tail.
Nothing touches ``/tmp`` or the network.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import secrets
import shutil
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal

from aar.media import MediaToolMissing, input_url, resolve_tool, timeout_for
from aar.plan.apply.inputs import plan_hash
from aar.plan.export.errors import ExportError
from aar.plan.export.files import _refuse_existing, write_pair
from aar.plan.export.layout import Layout, build_layout
from aar.plan.export.media import MediaMap, MediaRef
from aar.plan.inputs import LaughMap, Transcript
from aar.plan.lint import Finding, lint
from aar.plan.model import EditPlan, Source
from aar.plan.profile import Profile
from aar.plan.render.errors import RENDER_CODES, RUNTIME_CODES, RenderError
from aar.plan.render.fonts import FontChoice, find_font
from aar.plan.render.graph import SAMPLE_RATE, mix_graph, sample_at, video_graph
from aar.plan.render.loudness import normalize
from aar.plan.render.media import HubSource, resolve_source, used_source_ids
from aar.plan.render.report import not_rendered, notes, rendered
from aar.plan.render.runner import run_ffmpeg
from aar.plan.render.timeline import Timeline, build_timeline

__all__ = [
    "RENDER_CODES",
    "RUNTIME_CODES",
    "Preset",
    "RenderError",
    "RenderResult",
    "render_layout",
    "render_plan",
    "render_report_path",
]

Preset = Literal["review", "master"]
LUT_SUFFIXES = (".cube", ".3dl", ".dat", ".m3d", ".csp")
LUT_MAX_BYTES = 64 * 1024 * 1024
_HASH_CHUNK = 1024 * 1024


@dataclass
class RenderResult:
    """The placed file, its report, and the lint warnings that did not block it."""

    out: Path
    report_path: Path
    report: dict[str, Any]
    warnings: list[Finding] = field(default_factory=list)

    @property
    def qa_ok(self) -> bool:
        return bool(self.report["qa"]["ok"])


def render_report_path(out: Path) -> Path:
    """Where the render report goes: beside the file, ``<stem>.render.json`` (a name the FCPXML
    export's ``<stem>.report.json`` never takes)."""
    return out.with_name(f"{out.stem}.render.json")


def _container(out: Path, preset: Preset, profile: Profile) -> str:
    suffix = out.suffix.lower()
    prores = preset == "master" and profile.render.master_codec == "prores"
    allowed = (".mov",) if prores else (".mp4", ".mov")
    if suffix not in allowed:
        raise RenderError(
            "bad_output_path",
            f"--out {out.name!r}: a {'ProRes ' if prores else ''}{preset} render is written as "
            f"{' or '.join(allowed)}",
            f"give an output name ending in {allowed[0]}",
            path=str(out),
        )
    if not out.parent.is_dir():
        raise RenderError(
            "bad_output_path",
            f"the output's folder {str(out.parent)!r} does not exist",
            "create the folder first, or give another --out",
            path=str(out),
        )
    return "mov" if suffix == ".mov" else "mp4"


def _codec_args(preset: Preset, profile: Profile, container: str) -> tuple[list[str], str]:
    """Encoder arguments and the picture format the graph ends in."""
    settings = profile.render
    if preset == "master" and settings.master_codec == "prores":
        args = ["-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le"]
        args += ["-c:a", "pcm_s24le"]
        return [*args, "-movflags", "+faststart", "-f", container], "yuv422p10le"
    crf = settings.review_crf if preset == "review" else settings.master_crf
    speed = settings.review_x264_preset if preset == "review" else settings.master_x264_preset
    kbps = settings.review_audio_kbps if preset == "review" else settings.master_audio_kbps
    args = ["-c:v", "libx264", "-profile:v", "high", "-pix_fmt", "yuv420p"]
    args += ["-preset", speed, "-crf", str(crf)]
    args += ["-c:a", "aac", "-b:a", f"{kbps}k"]
    return [*args, "-movflags", "+faststart", "-f", container], "yuv420p"


def _hub_ref(hub: dict[str, HubSource]) -> Any:
    def resolve(source: Source) -> MediaRef:
        media = hub[source.id]
        path = str(media.path)
        return MediaRef(source.id, source.media, path, "hub-path", path, media.path.name)

    return resolve


def render_layout(plan: EditPlan, hub: dict[str, HubSource], profile: Profile) -> Layout:
    """S18's layout with hub paths. Start timecodes do not matter to a render (no clip links to
    a file), so they are left out: none is refused, unknown or off the frame grid."""
    untimed = plan.model_copy(
        update={
            "sources": [
                s.model_copy(update={"start_timecode": None, "start_timecode_rate": None})
                for s in plan.sources
            ]
        }
    )
    try:
        return build_layout(
            untimed,
            timeline_start=profile.export.timeline_start,
            allow_unknown_timecode=True,
            resolve_media=_hub_ref(hub),
        )
    except ExportError as exc:
        raise RenderError.from_export(exc) from None


def _luts(plan: EditPlan, used: list[str], media_root: Path | None) -> dict[str, Path]:
    """The LUT file of each used source whose ``color.lut`` names a LUT file on the hub."""
    found: dict[str, Path] = {}
    for source in plan.sources:
        lut = source.color.lut if source.color is not None else None
        if source.id not in used or lut is None or not lut.lower().endswith(LUT_SUFFIXES):
            continue
        path = Path(lut)
        if not path.is_absolute():
            if media_root is None:
                continue
            path = media_root / path
        with contextlib.suppress(OSError):
            if path.is_file() and os.access(path, os.R_OK) and path.stat().st_size < LUT_MAX_BYTES:
                found[source.id] = path
    return found


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_HASH_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _qa_plan(plan: EditPlan, hub: dict[str, HubSource]) -> EditPlan:
    """The plan with each used source's media replaced by its hub path, for QA's join checks."""
    return plan.model_copy(
        update={
            "sources": [
                s.model_copy(update={"media": str(hub[s.id].path)}) if s.id in hub else s
                for s in plan.sources
            ]
        }
    )


def _work_dir(out: Path) -> Path:
    """A fresh work dir beside the output (so the result is placed by a rename on its own file
    system, never through /tmp)."""
    while True:
        work = out.parent / f".{out.name}.aar-render-{secrets.token_hex(4)}"
        try:
            work.mkdir(mode=0o700)
        except FileExistsError:
            continue
        return work


def render_plan(
    plan: EditPlan,
    profile: Profile,
    *,
    out: Path,
    preset: Preset = "review",
    media_map: MediaMap | None = None,
    media_root: Path | None = None,
    transcript: Transcript | None = None,
    laughs: LaughMap | None = None,
    overwrite: bool = False,
) -> RenderResult:
    """Render ``plan`` to ``out`` (module docstring); a typed :class:`RenderError` on refusal or
    failure. Relative source media resolve against ``media_root``."""
    if preset not in ("review", "master"):
        raise ValueError("preset must be 'review' or 'master'")
    out = Path(os.path.abspath(out))
    container = _container(out, preset, profile)
    report_file = render_report_path(out)
    try:
        _refuse_existing([out, report_file], overwrite)
    except ExportError as exc:
        raise RenderError.from_export(exc) from None
    try:
        resolve_tool("ffmpeg")
        resolve_tool("ffprobe")
    except MediaToolMissing as exc:
        raise RenderError("tool_missing", str(exc), "install ffmpeg (with ffprobe)") from None
    findings = lint(plan, profile, transcript=transcript, laughs=laughs)
    if not findings.ok:
        errors = [f for f in findings.findings if f.severity == "error"]
        raise RenderError(
            "lint_errors",
            f"the plan has {len(errors)} lint error(s)",
            "fix the lint errors (aar plan lint) before rendering",
            errors=[{"rule": f.rule, "message": f.message} for f in errors],
        )
    warnings = [f for f in findings.findings if f.severity == "warn"]
    used = used_source_ids(plan)
    sources = plan.source_map()
    hub = {
        sid: resolve_source(
            sources[sid],
            plan.fps,
            media_map=media_map,
            media_root=media_root,
            fps_tolerance=profile.render.fps_tolerance,
        )
        for sid in used
    }
    layout = render_layout(plan, hub, profile)
    tl = build_timeline(layout, hub)
    font = find_font(profile.render.font)
    luts = _luts(plan, used, media_root)
    codec, out_fmt = _codec_args(preset, profile, container)
    duration_s = float(Fraction(tl.total_frames * plan.fps.den, plan.fps.num))
    timeout_s = timeout_for(
        duration_s,
        base_s=profile.render.timeout_base_s,
        per_media_s=profile.render.timeout_per_media_s,
    )
    work = _work_dir(out)
    try:
        return _render(
            plan,
            profile,
            tl,
            hub,
            font,
            luts,
            work=work,
            out=out,
            report_file=report_file,
            preset=preset,
            container=container,
            codec=codec,
            out_fmt=out_fmt,
            timeout_s=timeout_s,
            transcript=transcript,
            warnings=warnings,
            overwrite=overwrite,
        )
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _render(
    plan: EditPlan,
    profile: Profile,
    tl: Timeline,
    hub: dict[str, HubSource],
    font: FontChoice,
    luts: dict[str, Path],
    *,
    work: Path,
    out: Path,
    report_file: Path,
    preset: Preset,
    container: str,
    codec: list[str],
    out_fmt: str,
    timeout_s: float,
    transcript: Transcript | None,
    warnings: list[Finding],
    overwrite: bool,
) -> RenderResult:
    from aar.qa import check as qa_check  # the QA library imports the plan package

    font_name = None
    if font.path is not None:
        font_name = f"font{font.path.suffix.lower()}"
        shutil.copyfile(font.path, work / font_name)
    lut_names: dict[str, str] = {}
    for n, (sid, path) in enumerate(sorted(luts.items())):
        lut_names[sid] = f"lut{n}{path.suffix.lower()}"
        shutil.copyfile(path, work / lut_names[sid])
    graph = video_graph(tl, hub, font=font_name, luts=lut_names, out_pix_fmt=out_fmt)
    for text in graph.titles:
        (work / text.name).write_text(text.text, encoding="utf-8")
    (work / "video.ffgraph").write_text(graph.text, encoding="utf-8")
    (work / "mix.ffgraph").write_text(mix_graph(tl, hub), encoding="utf-8")
    inputs = [arg for path in tl.inputs for arg in ("-i", input_url(path))]
    samples = sample_at(plan.fps, tl.total_frames)
    run_ffmpeg(
        [
            *("-v", "error", "-y", *inputs, "-filter_complex_script", "mix.ffgraph"),
            *("-map", "[mix]", "-c:a", "pcm_f32le", "-ar", str(SAMPLE_RATE), "-ac", "2"),
            *("-rf64", "auto", "-f", "wav", "mix.wav"),
        ],
        "mix",
        timeout_s,
        work,
    )
    loudness = normalize(
        work / "mix.wav",
        work / "final.wav",
        kind=plan.kind,
        qa=profile.qa,
        settings=profile.render,
        samples=samples,
        timeout_s=timeout_s,
        cwd=work,
    )
    rendered_file = work / f"render.{container}"
    rate = f"{plan.fps.num}/{plan.fps.den}"
    run_ffmpeg(
        [
            *("-v", "error", "-y", *inputs, "-i", "final.wav"),
            *("-filter_complex_script", "video.ffgraph", "-map", "[vout]"),
            *("-map", f"{len(tl.inputs)}:a:0", "-r", rate, "-fps_mode", "cfr", *codec),
            rendered_file.name,
        ],
        "encode",
        timeout_s,
        work,
    )
    rendered_file.chmod(0o644)  # as the pair writer's report (and S18's files) get
    try:
        qa = qa_check(
            rendered_file,
            kind=plan.kind,
            profile=profile,
            plan=_qa_plan(plan, hub),
            transcript=transcript,
        )
    except MediaToolMissing as exc:
        raise RenderError("tool_missing", str(exc), "install ffmpeg (with ffprobe)") from None
    qa_dict = qa.model_dump(mode="json")
    qa_dict["file"] = out.name
    if isinstance(qa_dict.get("media"), dict) and "path" in qa_dict["media"]:
        qa_dict["media"]["path"] = out.name
    res = plan.resolution
    report: dict[str, Any] = {
        "format": container,
        "preset": preset,
        "file": out.name,
        "sha256": _sha256(rendered_file),
        "size_bytes": rendered_file.stat().st_size,
        "plan": {
            "project": plan.project,
            "timeline": plan.timeline,
            "kind": plan.kind,
            "sha256": plan_hash(plan),
        },
        "timeline": {
            "fps": rate,
            "total_frames": tl.total_frames,
            "duration_s": round(float(Fraction(tl.total_frames * plan.fps.den, plan.fps.num)), 6),
            "resolution": f"{res.width}x{res.height}",
            "audio_samples": samples,
            "sample_rate": SAMPLE_RATE,
        },
        "encode": {"args": codec, "pix_fmt": out_fmt, "frame_rate": rate},
        "media": [media.as_dict() for media in hub.values()],
        "rendered": rendered(tl, font, {sid: str(path) for sid, path in luts.items()}),
        "loudness": loudness.as_dict(),
        "not_rendered": [item.as_dict() for item in not_rendered(plan, tl.layout, font, luts)],
        "notes": [item.as_dict() for item in notes(tl, font, hub)],
        "lint_warnings": [{"rule": f.rule, "message": f.message} for f in warnings],
        "qa": qa_dict,
    }
    body = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    try:
        write_pair([(out, rendered_file), (report_file, body.encode("utf-8"))], overwrite=overwrite)
    except ExportError as exc:
        raise RenderError.from_export(exc) from None
    return RenderResult(out, report_file, report, warnings)
