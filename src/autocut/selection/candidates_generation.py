"""Generate temporally coherent highlight candidates from binary PU inference windows.

Candidate generation applies probability thresholding, gap-aware merging,
padding, and a minimum-duration filter for binary mountain biking highlights.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from ..ml.inference import InferenceResult


@dataclass(frozen=True)
class CandidateGenerationConfig:
    """Thresholds and temporal rules for binary candidate generation."""

    positive_threshold: float = 0.5
    min_candidate_duration: float = 3.0
    max_gap_to_merge: float = 3.0
    padding_seconds: float = 2.0

    def __post_init__(self) -> None:
        for name in ("positive_threshold",):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1.")
        for name in ("min_candidate_duration", "max_gap_to_merge", "padding_seconds"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative.")
        if self.min_candidate_duration <= 0:
            raise ValueError("min_candidate_duration must be greater than zero.")


@dataclass(frozen=True)
class WindowEvidence:
    """Window-level inference evidence retained for scoring and excerpt trimming."""

    start: float
    end: float
    probability: float


@dataclass(frozen=True)
class CandidateClip:
    """A padded, merged candidate clip with source-relative window evidence."""

    start: float
    end: float
    duration: float
    mean_probability: float
    max_probability: float
    window_evidence: tuple[WindowEvidence, ...] = ()


@dataclass
class _Span:
    start: float
    end: float
    window_indices: list[int]


def _validate_binary_inference(result: InferenceResult) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(result, InferenceResult):
        raise TypeError("inference must be an InferenceResult instance.")
    try:
        timestamps = np.asarray(result.timestamps, dtype=np.float64)
        probabilities = np.asarray(result.probabilities, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Inference timestamps and probabilities must be numeric.") from exc

    if timestamps.size == 0 and timestamps.shape == (0,):
        timestamps = timestamps.reshape(0, 2)
    if timestamps.ndim != 2 or timestamps.shape[1] != 2:
        raise ValueError("inference.timestamps must have shape (N, 2).")
    
    # Handle both 1D probability arrays or 2D (N, 2) where column 1 is the positive class probability
    if probabilities.ndim == 2 and probabilities.shape[1] == 2:
        probabilities = probabilities[:, 1]
    elif probabilities.ndim == 2 and probabilities.shape[1] == 1:
        probabilities = probabilities[:, 0]

    if probabilities.shape[0] != timestamps.shape[0]:
        raise ValueError("Number of probability scores must match number of window timestamps.")
    if not np.isfinite(timestamps).all() or not np.isfinite(probabilities).all():
        raise ValueError("Inference timestamps and probabilities must be finite.")
    if np.any(timestamps[:, 1] <= timestamps[:, 0]):
        raise ValueError("Every window end must be after its start.")
    if len(timestamps) > 1 and np.any(np.diff(timestamps[:, 0]) <= 0):
        raise ValueError("Window start timestamps must be strictly increasing.")
    if np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError("Probabilities must be between 0 and 1.")

    return timestamps, probabilities


def _merge_spans(spans: Sequence[_Span], *, max_gap: float) -> list[_Span]:
    """Merge chronologically adjacent spans within max_gap."""
    ordered = sorted(spans, key=lambda span: (span.start, span.end))
    if not ordered:
        return []
    merged = [_Span(ordered[0].start, ordered[0].end, list(ordered[0].window_indices))]
    for span in ordered[1:]:
        current = merged[-1]
        gap = span.start - current.end
        if gap <= max_gap:
            current.end = max(current.end, span.end)
            current.window_indices = sorted(set(current.window_indices).union(span.window_indices))
        else:
            merged.append(_Span(span.start, span.end, list(span.window_indices)))
    return merged


def generate_candidates(
    inference: InferenceResult,
    clip_duration: float,
    *,
    config: CandidateGenerationConfig | None = None,
) -> list[CandidateClip]:
    """Convert binary inference windows into chronologically ordered candidate clips."""
    if not isinstance(clip_duration, (int, float)) or not math.isfinite(clip_duration) or clip_duration <= 0:
        raise ValueError("clip_duration must be a finite positive number.")

    settings = config if config is not None else CandidateGenerationConfig()
    timestamps, probabilities = _validate_binary_inference(inference)
    if len(timestamps) == 0:
        return []

    # 1. Select windows meeting the positive threshold
    accepted: list[_Span] = []
    for index, (interval, prob) in enumerate(zip(timestamps, probabilities)):
        if prob >= settings.positive_threshold:
            accepted.append(_Span(float(interval[0]), float(interval[1]), [index]))

    if not accepted:
        return []

    # 2. Merge close intervals
    merged = _merge_spans(accepted, max_gap=settings.max_gap_to_merge)

    # 3. Apply padding
    padded: list[_Span] = []
    for span in merged:
        padded_start = max(0.0, span.start - settings.padding_seconds)
        padded_end = min(float(clip_duration), span.end + settings.padding_seconds)
        padded.append(_Span(padded_start, padded_end, list(span.window_indices)))

    final_spans = _merge_spans(padded, max_gap=0.0)

    # 4. Build final CandidateClips and apply min duration filter
    candidates: list[CandidateClip] = []
    for span in final_spans:
        start = min(max(span.start, 0.0), float(clip_duration))
        end = min(max(span.end, 0.0), float(clip_duration))
        duration = end - start
        if duration < settings.min_candidate_duration:
            continue

        unique_indices = sorted(set(span.window_indices))
        window_probs = probabilities[unique_indices]

        candidates.append(CandidateClip(
            start=start,
            end=end,
            duration=duration,
            mean_probability=float(np.mean(window_probs)),
            max_probability=float(np.max(window_probs)),
            window_evidence=tuple(
                WindowEvidence(
                    start=float(timestamps[idx, 0]),
                    end=float(timestamps[idx, 1]),
                    probability=float(probabilities[idx]),
                )
                for idx in unique_indices
            ),
        ))

    return candidates