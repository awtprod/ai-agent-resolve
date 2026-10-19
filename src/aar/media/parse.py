"""Pure parsers for the text ffmpeg filters print on stderr.

Each takes the whole captured stderr (other filters' lines may be mixed in) and returns typed
values. They never run anything, so they are tested on captured samples as well as live runs.

- ``ebur128`` prints a ``Summary:`` block when the filter closes (``framelog=quiet`` drops the
  per-frame lines). Values are printed with one decimal.
- ``astats`` prints ``key: value`` lines per channel and an ``Overall`` section.
- ``silencedetect`` prints ``silence_start`` and ``silence_end | silence_duration`` events as
  they happen; a silence still open when the stream ends may have no end line.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass


class ParseError(ValueError):
    """The expected filter output is not in the text."""


_NUMBER = r"([-+]?(?:inf|nan|\d+(?:\.\d*)?(?:[eE][-+]?\d+)?|\.\d+))"


def _float(text: str) -> float:
    return float(text)  # accepts "inf", "-inf" and "nan" as printed by ffmpeg


@dataclass(frozen=True)
class Ebur128Summary:
    """EBU R128 summary. ``true_peak_dbtp``/``sample_peak_dbfs`` are None unless requested."""

    integrated_lufs: float
    integrated_threshold_lufs: float
    lra_lu: float
    lra_threshold_lufs: float
    lra_low_lufs: float
    lra_high_lufs: float
    sample_peak_dbfs: float | None
    true_peak_dbtp: float | None


_EBU_LINE = re.compile(r"^\s*(I|Threshold|LRA|LRA low|LRA high|Peak):\s*" + _NUMBER + r"\s*(\S+)?")
_EBU_SECTION = re.compile(r"^\s*(Integrated loudness|Loudness range|Sample peak|True peak):\s*$")


def parse_ebur128(stderr: str) -> Ebur128Summary:
    """Parse the last ebur128 ``Summary:`` block in ``stderr``."""
    lines = stderr.splitlines()
    start = None
    for i in range(len(lines) - 1, -1, -1):
        if "Parsed_ebur128" in lines[i] and lines[i].rstrip().endswith("Summary:"):
            start = i
            break
    if start is None:
        raise ParseError("no ebur128 summary in the output")
    values: dict[tuple[str, str], float] = {}
    section = ""
    for line in lines[start + 1 :]:
        if line.startswith("["):  # another component's log line ends the block
            break
        sec = _EBU_SECTION.match(line)
        if sec:
            section = sec.group(1)
            continue
        m = _EBU_LINE.match(line)
        if m and section:
            values[(section, m.group(1))] = _float(m.group(2))
    try:
        return Ebur128Summary(
            integrated_lufs=values[("Integrated loudness", "I")],
            integrated_threshold_lufs=values[("Integrated loudness", "Threshold")],
            lra_lu=values[("Loudness range", "LRA")],
            lra_threshold_lufs=values[("Loudness range", "Threshold")],
            lra_low_lufs=values[("Loudness range", "LRA low")],
            lra_high_lufs=values[("Loudness range", "LRA high")],
            sample_peak_dbfs=values.get(("Sample peak", "Peak")),
            true_peak_dbtp=values.get(("True peak", "Peak")),
        )
    except KeyError as exc:
        raise ParseError(f"ebur128 summary is missing {exc.args[0][1]!r}") from None


@dataclass(frozen=True)
class AStats:
    """``astats`` Overall values, keyed by the printed name (e.g. ``"Peak level dB"``)."""

    overall: dict[str, float]

    def get(self, key: str) -> float | None:
        return self.overall.get(key)

    @property
    def peak_level_dbfs(self) -> float | None:
        return self.overall.get("Peak level dB")

    @property
    def non_finite(self) -> int:
        """NaN plus infinite samples, as astats prints them in Overall: the per-channel average
        (0 when astats was not asked to count them or the format cannot hold them)."""
        nan = self.overall.get("Number of NaNs") or 0.0
        inf = self.overall.get("Number of Infs") or 0.0
        return int(nan + inf)


_ASTATS_LINE = re.compile(r"^\[Parsed_astats_\d+ @ [^\]]*\]\s*(.*?):\s*" + _NUMBER + r"\s*$")
_ASTATS_HEAD = re.compile(r"^\[Parsed_astats_\d+ @ [^\]]*\]\s*(Overall|Channel: \d+)\s*$")


def parse_astats(stderr: str) -> AStats:
    """Parse the ``Overall`` section of the last astats instance in ``stderr``."""
    overall: dict[str, float] | None = None
    current: dict[str, float] | None = None
    for line in stderr.splitlines():
        head = _ASTATS_HEAD.match(line)
        if head:
            if head.group(1) == "Overall":
                overall = {}
                current = overall
            else:
                current = None
            continue
        m = _ASTATS_LINE.match(line)
        if m and current is not None:
            current[m.group(1)] = _float(m.group(2))
    if overall is None:
        raise ParseError("no astats Overall section in the output")
    return AStats(overall=overall)


@dataclass(frozen=True)
class Silence:
    """A silent stretch in seconds. ``end_s`` is None if the stream ended inside it."""

    start_s: float
    end_s: float | None

    def duration(self, stream_end_s: float | None = None) -> float | None:
        end = self.end_s if self.end_s is not None else stream_end_s
        if end is None:
            return None
        return max(0.0, end - self.start_s)


_SIL_START = re.compile(r"^\[silencedetect @ [^\]]*\]\s*silence_start:\s*" + _NUMBER)
_SIL_END = re.compile(r"^\[silencedetect @ [^\]]*\]\s*silence_end:\s*" + _NUMBER)


def parse_silencedetect(stderr: str) -> list[Silence]:
    """All silences in order. Start times are clamped at 0 (ffmpeg can print -0.0001)."""
    found: list[Silence] = []
    open_start: float | None = None
    for line in stderr.splitlines():
        m = _SIL_START.match(line)
        if m:
            if open_start is None:  # a repeated start keeps the earlier one
                open_start = max(0.0, _float(m.group(1)))
            continue
        m = _SIL_END.match(line)
        if m:
            end = _float(m.group(1))
            start = open_start if open_start is not None else 0.0
            if math.isfinite(end):
                found.append(Silence(start, max(start, end)))
            open_start = None
    if open_start is not None:
        found.append(Silence(open_start, None))
    return found
