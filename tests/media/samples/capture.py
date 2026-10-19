"""Regenerate the captured ffmpeg stderr samples next to this file (ffmpeg 6.1 on Linux).

Run by hand: ``python tests/media/samples/capture.py``. Inputs are lavfi sources written to
a scratch directory under this folder and deleted again. Lines naming a file are dropped so
no local path ends up in the samples.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHAIN = (
    "astats=measure_overall=Peak_level+Number_of_NaNs+Number_of_Infs+Flat_factor+Peak_count"
    "+Number_of_samples:measure_perchannel=none,silencedetect=n=-42dB:d=0.2,"
    "ebur128=peak=sample+true:framelog=quiet"
)


def run(*args: str) -> str:
    out = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostdin", "-nostats", *args],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    return out.stderr


def keep(text: str) -> str:
    lines = [ln for ln in text.splitlines() if "/" not in ln and "from '" not in ln]
    return "\n".join(lines) + "\n"


def main() -> None:
    scratch = HERE / "_scratch"
    scratch.mkdir(exist_ok=True)
    try:
        gap = scratch / "gap.wav"
        expr = "if(between(t\\,0.7\\,1.7)\\,0\\,0.1*sin(2*PI*440*t))"
        run("-y", "-f", "lavfi", "-i", f"aevalsrc={expr}|{expr}:s=48000:d=2.5", str(gap))
        (HERE / "chain_gap.txt").write_text(
            keep(run("-v", "info", "-i", str(gap), "-map", "0:a:0", "-af", CHAIN, "-f", "null", "-"))
        )
        quiet = scratch / "quiet.wav"
        run("-y", "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo:d=1", str(quiet))
        (HERE / "chain_digital_silence.txt").write_text(
            keep(run("-v", "info", "-i", str(quiet), "-af", CHAIN, "-f", "null", "-"))
        )
        nan = scratch / "nan.wav"
        run("-y", "-f", "lavfi", "-i", "aevalsrc=0/0|0:s=48000:d=0.5", "-c:a", "pcm_f32le", str(nan))
        (HERE / "astats_nan_perchannel.txt").write_text(
            keep(
                run(
                    "-v", "info", "-i", str(nan), "-af",
                    "astats=measure_overall=Peak_level+Number_of_NaNs+Number_of_Infs",
                    "-f", "null", "-",
                )
            )
        )
        tone = scratch / "tone.wav"
        run("-y", "-f", "lavfi", "-i", "sine=f=1000:r=48000:d=1", str(tone))
        (HERE / "ebur128_framelog.txt").write_text(
            keep(run("-v", "info", "-i", str(tone), "-af", "ebur128=peak=true", "-f", "null", "-"))
        )
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    main()
