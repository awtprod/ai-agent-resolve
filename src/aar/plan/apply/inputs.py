"""The real result files the later phases compile from.

A phase is submittable only from the real results of the phases before it (no placeholders). The
recipe runner (S13l) produces these from the live op results; the compiler reads them as its
typed input contract. ``PrepareResult`` is phase 1's output (the branch identity, one imported
pool item per source, one found pool item per title template). ``AssembleResult`` wraps phase
2's append results, from which phase 3 resolves item ids by ``(track, record_in)``.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from aar.node.ops.framework import MediaRef, TimelineIdentity, TimelineRef
from aar.node.ops.write import AppendResult


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ImportedSource(_Model):
    """One plan source, imported into the pool by phase 1."""

    source_id: str = Field(min_length=1)
    media: MediaRef
    file_path: str = Field(min_length=1, description="The real clip path, for the BRAW re-check.")


class FoundTitle(_Model):
    """One title/sting template, located in the pool by phase 1's ``media.find``."""

    template: str = Field(min_length=1)
    media: MediaRef


class PrepareResult(_Model):
    """Phase 1's output: the branched working timeline and the resolved pool items."""

    branch: TimelineIdentity
    sources: list[ImportedSource] = Field(default_factory=list)
    titles: list[FoundTitle] = Field(default_factory=list)

    def branch_ref(self) -> TimelineRef:
        return TimelineRef(name=self.branch.name, unique_id=self.branch.unique_id)

    def source_media(self) -> dict[str, ImportedSource]:
        return {s.source_id: s for s in self.sources}

    def title_media(self) -> dict[str, MediaRef]:
        return {t.template: t.media for t in self.titles}


class AssembleResult(_Model):
    """Phase 2's output: the append results that carry the real item ids."""

    branch: TimelineIdentity
    appends: list[AppendResult] = Field(default_factory=list)

    def branch_ref(self) -> TimelineRef:
        return TimelineRef(name=self.branch.name, unique_id=self.branch.unique_id)
