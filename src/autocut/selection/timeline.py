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
    annotation_order: int | None = None

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
                    "annotation_order": clip.annotation_order,
                }
                for clip in self.clips
            ],
        }


def assemble_timeline(
    selection: SelectionResult,
    mandatory_clips: Sequence[MandatoryClip] = (),
    *,
    max_clip_duration: float = 0.0,
) -> Timeline:
    """Place clips in annotation-manifest order, then source-time order.

    Pipeline candidates carry the physical manifest line number of their
    source interval. Mandatory clips use their sequence_order, which is set to
    that same line number. For compatibility, candidates without
    annotation_order fall back to video_order. Within an annotation, source time
    determines order. Output is concatenated without transitions/overlaps.
    """
    if not isinstance(selection, SelectionResult):
        raise TypeError("selection must be a SelectionResult instance.")
    if (
        isinstance(max_clip_duration, bool)
        or not isinstance(max_clip_duration, (int, float))
        or not math.isfinite(max_clip_duration)
        or max_clip_duration < 0
    ):
        raise ValueError("max_clip_duration must be finite and non-negative.")
    mandatory = tuple(mandatory_clips)
    for item in mandatory:
        if not isinstance(item, MandatoryClip):
            raise TypeError("mandatory_clips must contain MandatoryClip instances.")

    video_order: dict[str, int] = {}
    for candidate in selection.selected:
        previous = video_order.get(candidate.video_name)
        if previous is None or candidate.video_order < previous:
            video_order[candidate.video_name] = candidate.video_order
    for item in mandatory:
        previous = video_order.get(item.video_name)
        if previous is None or item.video_order < previous:
            video_order[item.video_name] = item.video_order

    # The annotation index is the primary order key. Source timestamps only
    # order clips belonging to the same annotation interval.
    items: list[tuple[tuple[int, float, int, str], str, object]] = []
    for candidate in selection.selected:
        order = (
            candidate.annotation_order
            if candidate.annotation_order is not None
            else candidate.video_order
        )
        items.append((
            (order, candidate.start, 1, candidate.video_name),
            "candidate",
            candidate,
        ))
    for item in mandatory:
        items.append((
            (item.sequence_order, item.start, 0, item.video_name),
            "mandatory",
            item,
        ))

    items.sort(key=lambda row: row[0])

    timeline_clips: list[TimelineClip] = []
    cursor = 0.0
    for _key, kind, item in items:
        if kind == "candidate":
            candidate = item
            assert isinstance(candidate, SelectionCandidate)
            if max_clip_duration > 0 and candidate.duration > max_clip_duration + 1e-9:
                raise ValueError(
                    f"Optional clip {candidate.video_name} duration "
                    f"{candidate.duration:.3f}s exceeds max_clip_duration "
                    f"{max_clip_duration:.3f}s."
                )
            source_start, source_end = candidate.start, candidate.end
            mandatory_flag = False
            score = candidate.score
            tier = candidate.tier
            video_name = candidate.video_name
            annotation_order = candidate.annotation_order
        else:
            clip = item
            assert isinstance(clip, MandatoryClip)
            source_start, source_end = clip.start, clip.end
            mandatory_flag = True
            score = None
            tier = None
            video_name = clip.video_name
            annotation_order = clip.sequence_order

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
            annotation_order=annotation_order,
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
