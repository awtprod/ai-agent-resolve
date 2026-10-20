"""Cold analysis workers: audio sync offsets (``aar.analyze.sync``) and word-level transcripts
(``aar.analyze.whisper``), with the ``aar analyze`` CLI group.

They run on the hub host on local files and register no node ops (the job wiring comes later).
numpy (extra ``analyze``) and faster-whisper (extra ``whisper``) are imported lazily, so this
package imports without them; a command that needs a missing one fails with a typed
``missing-extra`` error naming the extra to install.
"""

from aar.analyze.errors import AnalyzeError
from aar.analyze.sync import SyncSettings, sync_files
from aar.analyze.whisper import (
    CCN_PRESET,
    RecordingOffset,
    WhisperSettings,
    WhisperTranscriber,
    transcribe_file,
)

__all__ = [
    "CCN_PRESET",
    "AnalyzeError",
    "RecordingOffset",
    "SyncSettings",
    "WhisperSettings",
    "WhisperTranscriber",
    "sync_files",
    "transcribe_file",
]
