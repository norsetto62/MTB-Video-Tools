"""Top-window selection and temporal merging for LightGBM LambdaRank scores.

Raw ranking scores are not probabilities. This module intentionally keeps the
LightGBM candidate path separate from the legacy P/U probability pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class RankedWindow:
    """A scored source-time window."""
    start: float
    end: float
    score: float
    window_index: int


@dataclass(frozen=True)
class RankedSegment:
    """Merged segment with evidence and hierarchical ranking statistics."""
    start: float
    end: float
    max_score: float
    window_count: int
    mean_score: float
    window_indices: tuple[int, ...]

    @property
    def duration_seconds(self) -> float:
        return self.end - self.start


def select_ranked_segments(
    timestamps: np.ndarray,
    scores: Sequence[float] | np.ndarray,
    *,
    top_k: int = 10,
    max_gap_seconds: float = 2.0,
) -> list[RankedSegment]:
    """Keep top-k windows by score, merge overlaps/gaps, and rank segments.

    Ties are resolved deterministically by earlier start time then input index.
    Merging is performed in chronological order. Segment ranking uses max score,
    window count, mean score, and duration, all descending.
    """
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer.")
    if (isinstance(max_gap_seconds, bool)
            or not isinstance(max_gap_seconds, (int, float))
            or not math.isfinite(max_gap_seconds) or max_gap_seconds < 0):
        raise ValueError("max_gap_seconds must be finite and non-negative.")

    try:
        spans = np.asarray(timestamps, dtype=np.float64)
        values = np.asarray(scores, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("timestamps and scores must be numeric.") from exc
    if spans.size == 0 and spans.shape == (0,):
        spans = spans.reshape(0, 2)
    if spans.ndim != 2 or spans.shape[1] != 2:
        raise ValueError("timestamps must have shape (N, 2).")
    if values.ndim != 1 or values.shape[0] != spans.shape[0]:
        raise ValueError("scores must have shape (N,) matching timestamps.")
    if not np.isfinite(spans).all() or not np.isfinite(values).all():
        raise ValueError("timestamps and scores must be finite.")
    if np.any(spans[:, 0] < 0) or np.any(spans[:, 1] <= spans[:, 0]):
        raise ValueError("windows must have non-negative starts and end > start.")
    if len(spans) > 1 and np.any(np.diff(spans[:, 0]) < 0):
        raise ValueError("timestamps must be chronologically ordered.")
    if len(spans) == 0:
        return []

    ranked_indices = sorted(
        range(len(values)),
        key=lambda i: (-float(values[i]), float(spans[i, 0]), i),
    )[:top_k]
    chosen = [
        RankedWindow(float(spans[i, 0]), float(spans[i, 1]), float(values[i]), i)
        for i in ranked_indices
    ]
    chosen.sort(key=lambda item: (item.start, item.end, item.window_index))

    groups: list[list[RankedWindow]] = []
    for window in chosen:
        if not groups or window.start - max(item.end for item in groups[-1]) > max_gap_seconds:
            groups.append([window])
        else:
            groups[-1].append(window)

    segments = [
        RankedSegment(
            start=min(item.start for item in group),
            end=max(item.end for item in group),
            max_score=max(item.score for item in group),
            window_count=len(group),
            mean_score=float(np.mean([item.score for item in group])),
            window_indices=tuple(sorted(item.window_index for item in group)),
        )
        for group in groups
    ]
    segments.sort(key=lambda item: (
        -item.max_score,
        -item.window_count,
        -item.mean_score,
        -item.duration_seconds,
        item.start,
    ))
    return segments



def build_candidate_clips(
    timestamps: np.ndarray,
    scores: Sequence[float] | np.ndarray,
    *,
    video_duration: float,
    top_k: int = 10,
    max_gap_seconds: float = 2.0,
    min_clip_seconds: float = 3.0,
    padding_seconds: float = 2.0,
):
    """Select ranked segments and convert them to production CandidateClip values.

    The minimum duration is applied to the unpadded merged segment. Padding is
    then applied symmetrically and clipped to the real source-video duration.
    The returned pairs preserve the originating RankedSegment for diagnostics.
    """
    from .candidate_types import CandidateClip, WindowEvidence

    numeric_values = {
        "video_duration": video_duration,
        "max_gap_seconds": max_gap_seconds,
        "min_clip_seconds": min_clip_seconds,
        "padding_seconds": padding_seconds,
    }
    for name, value in numeric_values.items():
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0):
            raise ValueError(f"{name} must be finite and non-negative.")
    if video_duration <= 0:
        raise ValueError("video_duration must be greater than zero.")

    spans = np.asarray(timestamps, dtype=np.float64)
    values = np.asarray(scores, dtype=np.float64)
    segments = select_ranked_segments(
        spans, values, top_k=top_k, max_gap_seconds=max_gap_seconds,
    )
    candidates = []
    for segment in segments:
        if segment.duration_seconds + 1e-9 < min_clip_seconds:
            continue
        start = max(0.0, segment.start - padding_seconds)
        end = min(float(video_duration), segment.end + padding_seconds)
        if end <= start:
            continue
        evidence = tuple(
            WindowEvidence(
                start=float(spans[index, 0]),
                end=float(spans[index, 1]),
                probability=float(values[index]),
            )
            for index in segment.window_indices
        )
        candidate = CandidateClip(
            start=start,
            end=end,
            duration=end - start,
            mean_probability=segment.mean_score,
            max_probability=segment.max_score,
            window_evidence=evidence,
        )
        candidates.append((segment, candidate))
    return candidates
