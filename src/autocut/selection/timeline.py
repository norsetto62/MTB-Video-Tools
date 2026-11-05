"""Assemble a deterministic output timeline from retained and mandatory clips."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
from .selection import MandatoryClip, SelectionCandidate, SelectionResult

@dataclass(frozen=True)
class TimelineClip:
    """One continuous source interval positioned on the output timeline."""
    video_name: str
    source_start: float
    source_end: float
    output_start: float
    output_end: float
    mandatory: bool
    score: float | None = None
    annotation_order: int | None = None
    @property
    def duration(self) -> float: return self.output_end - self.output_start

@dataclass(frozen=True)
class Timeline:
    """Ordered clips and their derived total duration."""
    clips: tuple[TimelineClip, ...]
    duration: float
    def to_dict(self) -> dict:
        """Return a JSON-serializable timeline representation."""
        return {"schema_version": 1, "duration": self.duration, "clips": [
            {"video_name": c.video_name, "source_start": c.source_start, "source_end": c.source_end,
             "output_start": c.output_start, "output_end": c.output_end, "duration": c.duration,
             "mandatory": c.mandatory, "score": c.score, "annotation_order": c.annotation_order}
            for c in self.clips]}

def assemble_timeline(selection: SelectionResult, mandatory_clips: Sequence[MandatoryClip] = ()) -> Timeline:
    """Place clips in manifest order, then source-time order; concatenate without overlap."""
    if not isinstance(selection, SelectionResult): raise TypeError("selection must be a SelectionResult instance.")
    mandatory = tuple(mandatory_clips)
    for item in mandatory:
        if not isinstance(item, MandatoryClip): raise TypeError("mandatory_clips must contain MandatoryClip instances.")
    items: list[tuple[tuple[int, float, int, str], str, object]] = []
    for candidate in selection.selected:
        order = candidate.annotation_order if candidate.annotation_order is not None else candidate.video_order
        items.append(((order, candidate.start, 1, candidate.video_name), "candidate", candidate))
    for item in mandatory:
        items.append(((item.sequence_order, item.start, 0, item.video_name), "mandatory", item))
    items.sort(key=lambda row: row[0])
    clips: list[TimelineClip] = []
    cursor = 0.0
    for _key, kind, item in items:
        if kind == "candidate":
            c = item
            assert isinstance(c, SelectionCandidate)
            video_name, start, end = c.video_name, c.start, c.end
            mandatory_flag, score, annotation_order = False, c.score, c.annotation_order
        else:
            c = item
            assert isinstance(c, MandatoryClip)
            video_name, start, end = c.video_name, c.start, c.end
            mandatory_flag, score, annotation_order = True, None, c.sequence_order
        duration = end - start
        clips.append(TimelineClip(video_name, start, end, cursor, cursor + duration, mandatory_flag, score, annotation_order))
        cursor += duration
    return Timeline(tuple(clips), cursor)
