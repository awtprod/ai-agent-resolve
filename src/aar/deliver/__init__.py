"""Hub-side delivery: carry a finished render to a stored, hashed, QA'd and (review) sent artifact.

R6 steps 9 to 12. ``deliver_review`` makes the under-cap review MP4, QA's it and requests a human
review; ``deliver_master`` stores and QA's the master. Both take their transport, transcoder, QA,
event log and artifact store by injection. Recipe wiring is a later slice.
"""

from __future__ import annotations

from aar.deliver.errors import ERROR_CODES, DeliverError
from aar.deliver.model import DeliveryResult, RenderOutput
from aar.deliver.service import EditHostLike, deliver_master, deliver_review
from aar.deliver.transcode import FfmpegTranscoder, Transcoder

__all__ = [
    "ERROR_CODES",
    "DeliverError",
    "DeliveryResult",
    "EditHostLike",
    "FfmpegTranscoder",
    "RenderOutput",
    "Transcoder",
    "deliver_master",
    "deliver_review",
]
