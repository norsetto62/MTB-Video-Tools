"""Assemble a deterministic output timeline from selected and mandatory clips."""
from __future__ import annotations

from dataclasses import dataclass
import logging
import math
from typing import Sequence

from .budget import MandatoryClip, SelectionCandidate, SelectionResult

logger = logging.getLogger(__name__)


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
    tier: str | None = None

    @property
    def duration(self) -> float:
        return self.output_end - self.output_start


@dataclass(frozen=True)
class Timeline:
    """Ordered clips and final-duration diagnostics for the render stage."""

    clips: tuple[TimelineClip, ...]
    target_duration: float
    duration: float
    over_budget_by: float

    def to_dict(self) -> dict:
        """Return a JSON-serializable timeline representation."""
        return {
            "schema_version": 1,
            "target_duration": self.target_duration,
            "duration": self.duration,
            "over_budget_by": self.over_budget_by,
            "clips": [
                {
                    "video_name": clip.video_name,
                    "source_start": clip.source_start,
                    "source_end": clip.source_end,
                    "output_start": clip.output_start,
                    "output_end": clip.output_end,
                    "duration": clip.duration,
                    "mandatory": clip.mandatory,
                    "score": clip.score,
                    "tier": clip.tier,
                }
                for clip in self.clips
            ],
        }


def assemble_timeline(
    selection: SelectionResult,
    mandatory_clips: Sequence[MandatoryClip] = (),
) -> Timeline:
    """Place selected clips in source order, with mandatory clips preserved.

    Candidate video_order is the order of that source video in the user's
    input. Within each video, source time determines order. Mandatory clips
    use sequence_order from the user's manifest; for source videos with a
    matching candidate order, the mandatory clip is placed in that video's
    source-time sequence. For mandatory-only sources, sequence_order provides
    a deterministic position relative to the other mandatory entries.

    Output is concatenated without transitions or overlaps in this phase.
    """
    if not isinstance(selection, SelectionResult):
        raise TypeError("selection must be a SelectionResult instance.")
    mandatory = tuple(mandatory_clips)
    for item in mandatory:
        if not isinstance(item, MandatoryClip):
            raise TypeError("mandatory_clips must contain MandatoryClip instances.")

    video_order: dict[str, int] = {}
    for candidate in selection.selected:
        previous = video_order.get(candidate.video_name)
        if previous is None or candidate.video_order < previous:
            video_order[candidate.video_name] = candidate.video_order

    # Build sequence keys. A source video uses its source-order index; within
    # a video, source time preserves chronological order. A mandatory interval
    # for a known video is inserted at its source time. A mandatory-only video
    # is placed using its manifest sequence order.
    items: list[tuple[tuple[int, float, int, str], str, object]] = []
    for candidate in selection.selected:
        items.append((
            (candidate.video_order, candidate.start, 1, candidate.video_name),
            "candidate",
            candidate,
        ))
    for item in mandatory:
        order = video_order.get(item.video_name)
        if order is None:
            # No selected candidate survived from this source. sequence_order
            # is the stable manifest-order fallback for the mandatory entry.
            order = item.sequence_order
        items.append((
            (order, item.start, 0, item.video_name),
            "mandatory",
            item,
        ))

    # Preserve original source sequence. Equal source times put mandatory
    # material first, then use stable source names as a final tie-breaker.
    items.sort(key=lambda row: row[0])

    timeline_clips: list[TimelineClip] = []
    cursor = 0.0
    for _key, kind, item in items:
        if kind == "candidate":
            candidate = item
            assert isinstance(candidate, SelectionCandidate)
            source_start, source_end = candidate.start, candidate.end
            mandatory_flag = False
            score = candidate.score
            tier = candidate.tier
            video_name = candidate.video_name
        else:
            clip = item
            assert isinstance(clip, MandatoryClip)
            source_start, source_end = clip.start, clip.end
            mandatory_flag = True
            score = None
            tier = None
            video_name = clip.video_name

        duration = source_end - source_start
        timeline_clips.append(TimelineClip(
            video_name=video_name,
            source_start=source_start,
            source_end=source_end,
            output_start=cursor,
            output_end=cursor + duration,
            mandatory=mandatory_flag,
            score=score,
            tier=tier,
        ))
        cursor += duration

    over_budget_by = max(0.0, cursor - selection.target_duration)
    if over_budget_by > 0:
        logger.warning(
            "Assembled timeline duration %.3fs exceeds target %.3fs by %.3fs; "
            "preserving all selected and mandatory clips.",
            cursor, selection.target_duration, over_budget_by,
        )

    return Timeline(
        clips=tuple(timeline_clips),
        target_duration=selection.target_duration,
        duration=cursor,
        over_budget_by=over_budget_by,
    )
