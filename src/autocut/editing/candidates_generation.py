"""Generate temporally coherent highlight candidates from window inference.

This module implements Phase 9 only: window scoring/gating and temporal
candidate formation. It does not rank candidates for a duration budget or
select the final edit (Phase 10).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from ..ml.inference import InferenceResult


@dataclass(frozen=True)
class CandidateGenerationConfig:
    """Thresholds and temporal rules used to form highlight candidates."""

    class_score_weights: tuple[float, float, float] = (0.0, 1.0, 2.5)
    junk_threshold: float = 0.55
    action_threshold: float = 0.50
    score_threshold: float = 0.60
    min_candidate_duration: float = 4.0
    max_gap_to_merge: float = 3.0
    padding_seconds: float = 2.0

    def __post_init__(self) -> None:
        if len(self.class_score_weights) != 3:
            raise ValueError("class_score_weights must contain exactly three weights.")
        if not all(math.isfinite(value) for value in self.class_score_weights):
            raise ValueError("class_score_weights must contain only finite values.")
        if any(value < 0 for value in self.class_score_weights):
            raise ValueError("class_score_weights must be non-negative.")

        for name in ("junk_threshold", "action_threshold"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1.")

        if not math.isfinite(self.score_threshold):
            raise ValueError("score_threshold must be finite.")
        if self.score_threshold < 0:
            raise ValueError("score_threshold must be non-negative.")

        for name in ("min_candidate_duration", "max_gap_to_merge", "padding_seconds"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative.")
        if self.min_candidate_duration <= 0:
            raise ValueError("min_candidate_duration must be greater than zero.")


@dataclass(frozen=True)
class CandidateClip:
    """A temporally coherent candidate for later budget-based selection.

    start, end and duration describe the final padded and clamped interval.
    mean_score and max_score use unique accepted inference windows contributing
    to the candidate, not padded or bridged gap time.
    """

    start: float
    end: float
    duration: float
    tier: str
    mean_score: float
    max_score: float


@dataclass
class _Span:
    """Internal interval plus the accepted windows contributing to it."""

    start: float
    end: float
    window_indices: list[int]


def _validate_inference(result: InferenceResult) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(result, InferenceResult):
        raise TypeError("inference must be an InferenceResult instance.")

    try:
        timestamps = np.asarray(result.timestamps, dtype=np.float64)
        probabilities = np.asarray(result.probabilities, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Inference timestamps and probabilities must be numeric.") from exc

    # An empty Python list becomes shape (0,) under np.asarray([]).
    # Normalize that one unambiguous empty representation to (0, 2).
    if timestamps.size == 0 and timestamps.shape == (0,):
        timestamps = timestamps.reshape(0, 2)
    if timestamps.ndim != 2 or timestamps.shape[1:] != (2,):
        raise ValueError("inference.timestamps must have shape (N, 2).")
    if probabilities.shape != (timestamps.shape[0], 3):
        raise ValueError("inference.probabilities must have shape (N, 3).")
    if not np.isfinite(timestamps).all():
        raise ValueError("inference.timestamps must contain only finite values.")
    if not np.isfinite(probabilities).all():
        raise ValueError("inference.probabilities must contain only finite values.")
    if np.any(timestamps[:, 1] <= timestamps[:, 0]):
        raise ValueError("Every window end must be after its start.")
    if len(timestamps) > 1 and np.any(np.diff(timestamps[:, 0]) <= 0):
        raise ValueError("Window start timestamps must be strictly increasing.")
    if np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError("Class probabilities must be between 0 and 1.")
    if len(probabilities) and not np.allclose(
        probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1e-5
    ):
        raise ValueError("Each row of class probabilities must sum to 1.")

    return timestamps, probabilities


def _merge_spans(
    spans: Sequence[_Span],
    *,
    max_gap: float,
) -> list[_Span]:
    """Merge sorted spans when their gap is <= 0 or strictly below max_gap."""
    if not spans:
        return []

    ordered = sorted(spans, key=lambda span: (span.start, span.end))
    merged: list[_Span] = [
        _Span(ordered[0].start, ordered[0].end, list(ordered[0].window_indices))
    ]

    for span in ordered[1:]:
        current = merged[-1]
        gap = span.start - current.end
        should_merge = gap <= 0.0 or (max_gap > 0.0 and gap < max_gap)
        if should_merge:
            current.end = max(current.end, span.end)
            current.window_indices = sorted(
                set(current.window_indices).union(span.window_indices)
            )
        else:
            merged.append(_Span(span.start, span.end, list(span.window_indices)))

    return merged


def generate_candidates(
    inference: InferenceResult,
    clip_duration: float,
    *,
    config: CandidateGenerationConfig | None = None,
) -> list[CandidateClip]:
    """Convert window probabilities into chronologically ordered candidate clips.

    Window scoring and gating rules:
      1. Reject as junk if P0 >= junk_threshold.
      2. Otherwise accept as high_action if P2 >= action_threshold.
      3. Otherwise accept as moderate_interest if weighted score >= score_threshold.
      4. Reject all other windows.

    Accepted windows are unioned, brief gaps are bridged, short spans are
    removed before padding, and padded spans are merged and clamped to the
    video interval [0, clip_duration].

    Args:
        inference: Window timestamps and three-class probabilities from
            infer_windows.
        clip_duration: Duration of the source video in seconds.
        config: Optional candidate-generation settings. Defaults to the
            agreed Phase 9 starting values.

    Returns:
        CandidateClip objects in chronological order. An empty list means no
        windows passed gating or all resulting spans were too short.
    """
    if isinstance(clip_duration, bool) or not isinstance(clip_duration, (int, float)):
        raise ValueError("clip_duration must be a finite positive number.")
    if not math.isfinite(clip_duration) or clip_duration <= 0:
        raise ValueError("clip_duration must be a finite positive number.")

    settings = config if config is not None else CandidateGenerationConfig()
    if not isinstance(settings, CandidateGenerationConfig):
        raise TypeError("config must be a CandidateGenerationConfig instance.")

    timestamps, probabilities = _validate_inference(inference)
    if len(timestamps) == 0:
        return []

    weights = np.asarray(settings.class_score_weights, dtype=np.float64)
    scores = probabilities @ weights

    accepted: list[_Span] = []
    accepted_tiers: dict[int, str] = {}
    for index, (interval, row, score) in enumerate(zip(timestamps, probabilities, scores)):
        p0, _p1, p2 = row
        if p0 >= settings.junk_threshold:
            continue
        if p2 >= settings.action_threshold:
            tier = "high_action"
        elif score >= settings.score_threshold:
            tier = "moderate_interest"
        else:
            continue

        accepted_tiers[index] = tier
        accepted.append(_Span(float(interval[0]), float(interval[1]), [index]))

    if not accepted:
        return []

    # Stage 2: dissolve overlapping and touching sliding-window intervals.
    raw_spans = _merge_spans(accepted, max_gap=0.0)

    # Stage 3: bridge only gaps strictly shorter than max_gap_to_merge.
    gap_merged = _merge_spans(raw_spans, max_gap=settings.max_gap_to_merge)

    # Stage 4: minimum-duration filtering happens before padding.
    long_enough = [
        span for span in gap_merged
        if span.end - span.start >= settings.min_candidate_duration
    ]

    # Stage 5: pad surviving spans, retaining contributing window indices.
    padded = [
        _Span(
            span.start - settings.padding_seconds,
            span.end + settings.padding_seconds,
            list(span.window_indices),
        )
        for span in long_enough
    ]

    # Stage 6a: merge padded intervals that overlap or touch.
    final_spans = _merge_spans(padded, max_gap=0.0)

    # Stage 6b: clamp to video bounds. The duration reflects the final interval.
    candidates: list[CandidateClip] = []
    for span in final_spans:
        start = min(max(span.start, 0.0), float(clip_duration))
        end = min(max(span.end, 0.0), float(clip_duration))
        if end <= start:
            continue

        unique_indices = sorted(set(span.window_indices))
        contributing_scores = scores[unique_indices]
        tier = (
            "high_action"
            if any(accepted_tiers[index] == "high_action" for index in unique_indices)
            else "moderate_interest"
        )
        candidates.append(
            CandidateClip(
                start=start,
                end=end,
                duration=end - start,
                tier=tier,
                mean_score=float(np.mean(contributing_scores)),
                max_score=float(np.max(contributing_scores)),
            )
        )

    return candidates
