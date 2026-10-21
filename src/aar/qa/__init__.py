"""Media QA on rendered files: ``check(...)`` returns a report with one entry per check.

See ``aar.qa.check`` for the checks and docs/DESIGN.md "Media QA" for their definitions.
"""

from aar.qa.check import (
    JoinQuiet,
    agreement,
    check,
    evaluate_duration,
    evaluate_loudness,
    evaluate_quiet_run,
    evaluate_silence,
    evaluate_size,
    judge_join_quiet,
    resolve_media,
    suggest_src_in,
    tokens,
)
from aar.qa.planmap import Cut, KeptWord, StraddledWord, cuts, kept_words, straddled_words
from aar.qa.report import CHECK_IDS, Kind, QACheck, QAReport, SkipReason, Status
from aar.qa.transcriber import Transcriber

__all__ = [
    "CHECK_IDS",
    "Cut",
    "JoinQuiet",
    "KeptWord",
    "Kind",
    "QACheck",
    "QAReport",
    "SkipReason",
    "Status",
    "StraddledWord",
    "Transcriber",
    "agreement",
    "check",
    "cuts",
    "evaluate_duration",
    "evaluate_loudness",
    "evaluate_quiet_run",
    "evaluate_silence",
    "evaluate_size",
    "judge_join_quiet",
    "kept_words",
    "resolve_media",
    "straddled_words",
    "suggest_src_in",
    "tokens",
]
