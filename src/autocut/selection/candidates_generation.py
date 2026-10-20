"""Generate temporally coherent highlight candidates from inference windows.

Candidate generation is local to one annotated source interval. It applies
class-aware eligibility, optional asymmetric P2 envelope smoothing, tier-aware
padding and temporal merging, and a final minimum-duration filter.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from ..ml.inference import InferenceResult


@dataclass(frozen=True)
class CandidateGenerationConfig:
    """Binary confidence thresholds and temporal rules for candidate generation."""

    class_score_weights: tuple[float, float] = (0.0, 1.0)
    junk_threshold: float = 0.20
    action_threshold: float = 0.60
    flow_threshold: float = 0.40
    min_candidate_duration: float = 3.0
    max_gap_to_merge: float = 3.0
    padding_seconds: float = 2.0
    # Legacy setting name retained for config compatibility. The envelope is
    # applied to P(good), not a third-class probability.
    p2_envelope_decay: float = 0.0

    def __post_init__(self) -> None:
        if len(self.class_score_weights) != 2:
            raise ValueError("class_score_weights must contain exactly two weights (bad, good).")
        if not all(math.isfinite(value) for value in self.class_score_weights):
            raise ValueError("class_score_weights must contain only finite values.")
        if any(value < 0 for value in self.class_score_weights):
            raise ValueError("class_score_weights must be non-negative.")
        for name in ("junk_threshold", "action_threshold", "flow_threshold", "p2_envelope_decay"):
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
    score: float
    tier: str
    p0: float = 0.0
    p1: float = 0.0
    p2: float = 0.0

@dataclass(frozen=True)
class CandidateClip:
    """A padded, merged candidate clip, with source-relative window evidence."""

    start: float
    end: float
    duration: float
    tier: str
    mean_score: float
    max_score: float
    window_evidence: tuple[WindowEvidence, ...] = ()

@dataclass
class _Span:
    start: float
    end: float
    window_indices: list[int]
    tier: str

def _validate_inference(result: InferenceResult) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(result, InferenceResult):
        raise TypeError("inference must be an InferenceResult instance.")
    try:
        timestamps = np.asarray(result.timestamps, dtype=np.float64)
        probabilities = np.asarray(result.probabilities, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Inference timestamps and probabilities must be numeric.") from exc
    if timestamps.size == 0 and timestamps.shape == (0,):
        timestamps = timestamps.reshape(0, 2)
    if timestamps.ndim != 2 or timestamps.shape[1:] != (2,):
        raise ValueError("inference.timestamps must have shape (N, 2).")
    if probabilities.shape != (timestamps.shape[0], 2):
        raise ValueError("inference.probabilities must have shape (N, 2) for bad/good classes.")
    if not np.isfinite(timestamps).all() or not np.isfinite(probabilities).all():
        raise ValueError("Inference timestamps and probabilities must be finite.")
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

def _merge_spans(spans: Sequence[_Span], *, max_gap: float) -> list[_Span]:
    """Merge chronologically adjacent spans only when their semantic tiers match."""
    ordered = sorted(spans, key=lambda span: (span.start, span.end, span.tier))
    if not ordered:
        return []
    merged = [
        _Span(ordered[0].start, ordered[0].end,
              list(ordered[0].window_indices), ordered[0].tier)
    ]
    for span in ordered[1:]:
        current = merged[-1]
        gap = span.start - current.end
        if span.tier == current.tier and gap <= max_gap:
            current.end = max(current.end, span.end)
            current.window_indices = sorted(
                set(current.window_indices).union(span.window_indices)
            )
        else:
            merged.append(
                _Span(span.start, span.end, list(span.window_indices), span.tier)
            )
    return merged


def _p2_envelope(values: np.ndarray, decay: float) -> np.ndarray:
    """Follow P2 rises immediately and let falls decay asymmetrically."""
    if decay <= 0.0 or len(values) == 0:
        return values.copy()
    envelope = values.copy()
    for index in range(1, len(values)):
        envelope[index] = max(values[index], envelope[index - 1] * decay)
    return envelope


def generate_candidates(
    inference: InferenceResult,
    clip_duration: float,
    *,
    config: CandidateGenerationConfig | None = None,
) -> list[CandidateClip]:
    """Convert binary bad/good probabilities into temporally coherent candidates.

    Class 0 is bad and class 1 is good. A good probability above the higher
    threshold receives the high-priority tier; probability above the lower
    threshold receives the moderate-priority tier. These are confidence tiers,
    not distinct learned action/flow classes. Temporal merging, padding,
    clipping and minimum-duration rules remain unchanged.
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

    p_bad, p_good = probabilities.T
    good_envelope = _p2_envelope(p_good, settings.p2_envelope_decay)
    scores = probabilities @ np.asarray(settings.class_score_weights, dtype=np.float64)

    accepted: list[_Span] = []
    accepted_tiers: dict[int, str] = {}
    for index, interval in enumerate(timestamps):
        start, end = float(interval[0]), float(interval[1])
        if good_envelope[index] >= settings.action_threshold:
            tier = "high_action"
        elif good_envelope[index] >= settings.flow_threshold:
            tier = "moderate_interest"
        else:
            tier = None
        if tier is None:
            continue
        accepted_tiers[index] = tier
        accepted.append(_Span(start, end, [index], tier))

    if not accepted:
        return []

    merged = _merge_spans(accepted, max_gap=settings.max_gap_to_merge)
    padded: list[_Span] = []
    for index, span in enumerate(merged):
        previous_other = [
            (other.end + span.start) / 2.0 for other in merged[:index]
            if other.tier != span.tier
        ]
        next_other = [
            (span.end + other.start) / 2.0 for other in merged[index + 1:]
            if other.tier != span.tier
        ]
        padded_start = max(0.0, span.start - settings.padding_seconds)
        padded_end = min(float(clip_duration), span.end + settings.padding_seconds)
        if previous_other:
            padded_start = max(padded_start, max(previous_other))
        if next_other:
            padded_end = min(padded_end, min(next_other))
        padded.append(_Span(padded_start, padded_end, list(span.window_indices), span.tier))
    final_spans = _merge_spans(padded, max_gap=0.0)
    candidates: list[CandidateClip] = []
    for span in final_spans:
        start = min(max(span.start, 0.0), float(clip_duration))
        end = min(max(span.end, 0.0), float(clip_duration))
        if end - start < settings.min_candidate_duration:
            continue
        unique_indices = sorted(set(span.window_indices))
        contributing_scores = scores[unique_indices]
        candidates.append(CandidateClip(
            start=start,
            end=end,
            duration=end - start,
            tier=span.tier,
            mean_score=float(np.mean(contributing_scores)),
            max_score=float(np.max(contributing_scores)),
            window_evidence=tuple(
                WindowEvidence(
                    start=float(timestamps[i, 0]),
                    end=float(timestamps[i, 1]),
                    score=float(scores[i]),
                    tier=accepted_tiers[i],
                    p0=float(p_bad[i]),
                    p1=float(p_good[i]),
                    p2=0.0,
                )
                for i in unique_indices
            ),
        ))
    return candidates
