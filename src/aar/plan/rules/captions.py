"""Caption rules: line length and house style."""

from __future__ import annotations

import colorsys
import re

from aar.plan.lint import Finding, LintContext, Ref, rule
from aar.plan.model import CaptionStyle

_NAMED = {
    "yellow": (255, 255, 0),
    "gold": (255, 215, 0),
    "orange": (255, 165, 0),
    "red": (255, 0, 0),
    "green": (0, 128, 0),
    "blue": (0, 0, 255),
    "white": (255, 255, 255),
    "black": (0, 0, 0),
}
_MAX_REFS = 20


def _hsv(color: str) -> tuple[float, float, float] | None:
    """Hue (degrees), saturation and value of '#RGB', '#RRGGBB' or a basic colour name."""
    text = color.strip().lower()
    if text in _NAMED:
        red, green, blue = _NAMED[text]
    elif re.fullmatch(r"#[0-9a-f]{6}", text):
        red, green, blue = (int(text[i : i + 2], 16) for i in (1, 3, 5))
    elif re.fullmatch(r"#[0-9a-f]{3}", text):
        red, green, blue = (int(ch * 2, 16) for ch in text[1:])
    else:
        return None
    hue, sat, val = colorsys.rgb_to_hsv(red / 255, green / 255, blue / 255)
    return hue * 360, sat, val


def _capped(refs: list[Ref]) -> tuple[list[Ref], str]:
    if len(refs) <= _MAX_REFS:
        return refs, ""
    return refs[:_MAX_REFS], f" (the first {_MAX_REFS} are listed)"


@rule(
    "caption-length",
    severity="error",
    summary="Caption lines are short enough to read at a glance.",
    params=("max_chars",),
)
def caption_length(ctx: LintContext) -> list[Finding]:
    limit = int(ctx.params["max_chars"])
    out: list[Finding] = []
    for number, caption in enumerate(ctx.plan.captions):
        for line in caption.lines:
            if len(line) <= limit:
                continue
            out.append(
                ctx.finding(
                    "error",
                    f"Caption {number} has a {len(line)}-character line {line!r}; "
                    f"the limit is {limit}.",
                    f"Split it into lines of at most {limit} characters.",
                    [ctx.record_ref(caption.record_in, f"caption[{number}]")],
                )
            )
    return out


@rule(
    "caption-style",
    severity="info",
    summary="Captions use the house font and highlight the active word.",
    params=("font", "active_hue_min", "active_hue_max", "active_min_saturation"),
)
def caption_style(ctx: LintContext) -> list[Finding]:
    font = str(ctx.params["font"])
    hue_min = float(ctx.params["active_hue_min"])
    hue_max = float(ctx.params["active_hue_max"])
    min_sat = float(ctx.params["active_min_saturation"])
    bad_font: list[Ref] = []
    bad_color: list[Ref] = []
    for number, caption in enumerate(ctx.plan.captions):
        style = caption.style or ctx.plan.caption_style or CaptionStyle()
        ref = ctx.record_ref(caption.record_in, f"caption[{number}]")
        if style.font is None or style.font.lower() != font.lower():
            bad_font.append(ref)
        hsv = _hsv(style.active_word_color) if style.active_word_color else None
        if hsv is None or not (hue_min <= hsv[0] <= hue_max and hsv[1] >= min_sat):
            bad_color.append(ref)
    out: list[Finding] = []
    if bad_font:
        refs, note = _capped(bad_font)
        out.append(
            ctx.finding(
                "info",
                f"{len(bad_font)} caption(s) do not use the font {font!r}{note}.",
                f'Set caption_style.font (or each caption\'s style) to "{font}".',
                refs,
            )
        )
    if bad_color:
        refs, note = _capped(bad_color)
        out.append(
            ctx.finding(
                "info",
                f"{len(bad_color)} caption(s) do not highlight the active word in yellow{note}.",
                'Set active_word_color to a yellow such as "#FFE600".',
                refs,
            )
        )
    return out
