"""Read, diff, modify and create .drp, .drt and .drb files; needs nothing from Resolve.

Reading (this slice) is offline and read-only::

    from aar.drp import read_drp, describe

    drp = read_drp("project.drp")
    print(describe(drp))

Every decoded value carries a :class:`~aar.drp.model.Provenance` (zip member + byte span) so a
later writer can splice edits into the original text.
"""

from aar.drp.container import Container, open_container
from aar.drp.describe import describe
from aar.drp.errors import BlobError, ContainerError, DrpError, ZstdUnavailableError
from aar.drp.model import DrpFile, Provenance
from aar.drp.reader import read_drp

__all__ = [
    "BlobError",
    "Container",
    "ContainerError",
    "DrpError",
    "DrpFile",
    "Provenance",
    "ZstdUnavailableError",
    "describe",
    "open_container",
    "read_drp",
]
