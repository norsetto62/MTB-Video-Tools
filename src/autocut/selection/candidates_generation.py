"""Generate temporally coherent highlight candidates from inference windows.

Candidate generation is local to one annotated source interval. It applies
class-aware eligibility, optional asymmetric P2 envelope smoothing, padding,
temporal merging, and a final minimum-duration filter. Global budget selection
belongs to Phase 10.
"""
from __future__ import annotations

from dataclasses import dataclass
import logging
import math
from typing import Sequence

import numpy as np

from ..ml.inference import InferenceResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CandidateGenerationConfig:
    """Class-aware thresholds and temporal rules for candidate generation."""

    class_score_weights: tuple[float, float, float] = (0.0, 1.0, 2.5)
    junk_threshold: float = 0.20
    action_threshold: float = 0.60
    flow_threshold: float = 0.40
    min_candidate_duration: float = 3.0
    max_gap_to_merge: float = 3.0
    padding_seconds: float = 2.0
    # 0 disables smoothing. Otherwise a falling P2 envelope decays by this
    # factor per inference window, while rises are followed immediately.
    p2_envelope_decay: float = 0.0

    def __post_init__(self) -> None:
        if len(self.class_score_weights) != 3:
            raise ValueError("class_score_weights must contain exactly three weights.")
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
    if probabilities.shape != (timestamps.shape[0], 3):
        raise ValueError("inference.probabilities must have shape (N, 3).")
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
    """Merge spans of the same semantic tier only; the gap rule is inclusive."""
    merged: list[_Span] = []
    for tier in ("high_action", "moderate_interest"):
        ordered = sorted(
            (span for span in spans if span.tier == tier),
            key=lambda span: (span.start, span.end),
        )
        if not ordered:
            continue
        tier_merged = [
            _Span(ordered[0].start, ordered[0].end,
                  list(ordered[0].window_indices), tier)
        ]
        for span in ordered[1:]:
            current = tier_merged[-1]
            gap = span.start - current.end
            if gap <= max_gap:
                current.end = max(current.end, span.end)
                current.window_indices = sorted(
                    set(current.window_indices).union(span.window_indices)
                )
            else:
                tier_merged.append(
                    _Span(span.start, span.end, list(span.window_indices), tier)
                )
        merged.extend(tier_merged)
    return sorted(merged, key=lambda span: (span.start, span.end, span.tier))


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
    """Convert inference windows into chronologically ordered candidate clips.

    Eligibility is class-aware: high action requires P2 >= action_threshold,
    P0 < junk_threshold, and class 2 to remain the strongest class after the
    optional P2 envelope. Flow requires P0 < junk_threshold, P1 >=
    flow_threshold, and either class 1 to be the raw argmax or P1 > P2.

    Accepted windows are padded, merged using the inclusive max-gap rule, and
    clamped to the source interval. min_candidate_duration is applied only to
    the final padded-and-merged clip boundaries.
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

    logger.info(
        "Candidate generation started: windows=%d clip_duration=%.3fs "
        "junk_threshold=%.3f action_threshold=%.3f flow_threshold=%.3f "
        "max_gap_to_merge=%.3fs padding_seconds=%.3fs p2_envelope_decay=%.3f.",
        len(timestamps), float(clip_duration), settings.junk_threshold,
        settings.action_threshold, settings.flow_threshold,
        settings.max_gap_to_merge, settings.padding_seconds,
        settings.p2_envelope_decay,
    )

    p0, p1, p2 = probabilities.T
    p2_envelope = _p2_envelope(p2, settings.p2_envelope_decay)
    smoothed_class = np.argmax(np.column_stack((p0, p1, p2_envelope)), axis=1)
    raw_class = np.argmax(probabilities, axis=1)
    scores = probabilities @ np.asarray(settings.class_score_weights, dtype=np.float64)

    accepted: list[_Span] = []
    accepted_tiers: dict[int, str] = {}
    rejected_junk = 0
    rejected_threshold = 0
    high_action_count = 0
    moderate_count = 0
    for index, interval in enumerate(timestamps):
        start, end = float(interval[0]), float(interval[1])
        action_class = bool(
            p2_envelope[index] >= settings.action_threshold
            and smoothed_class[index] == 2
        )
        flow_class = bool(
            p1[index] >= settings.flow_threshold
            and (raw_class[index] == 1 or p1[index] > p2[index])
        )
        if p0[index] >= settings.junk_threshold:
            rejected_junk += 1
            reason = "junk_probability"
            tier = None
        elif action_class:
            tier = "high_action"
            reason = "accepted"
        elif flow_class:
            tier = "moderate_interest"
            reason = "accepted"
        else:
            rejected_threshold += 1
            tier = None
            reason = "below_action_and_flow_eligibility"
        logger.info(
            "Candidate window #%d [%.3f, %.3f): P0=%.6f P1=%.6f P2=%.6f "
            "P2_envelope=%.6f raw_class=%d envelope_class=%d tier=%s reason=%s.",
            index, start, end, p0[index], p1[index], p2[index],
            p2_envelope[index], raw_class[index], smoothed_class[index],
            tier or "-", reason,
        )
        if tier is None:
            continue
        accepted_tiers[index] = tier
        accepted.append(_Span(start, end, [index], tier))
        if tier == "high_action":
            high_action_count += 1
        else:
            moderate_count += 1

    logger.info(
        "Candidate window classification: total=%d accepted=%d high_action=%d "
        "moderate_interest=%d rejected_junk=%d rejected_below_eligibility=%d.",
        len(timestamps), len(accepted), high_action_count, moderate_count,
        rejected_junk, rejected_threshold,
    )

    if not accepted:
        logger.info("Candidate generation produced no accepted inference windows.")
        return []

    # First form semantic regions using max_gap. Padding is context extension,
    # not a way to enlarge the configured gap tolerance.
    merged = _merge_spans(accepted, max_gap=settings.max_gap_to_merge)
    logger.info(
        "Candidate spans after semantic merge (gap <= %.3fs): %d.",
        settings.max_gap_to_merge, len(merged),
    )
    for index, span in enumerate(merged, start=1):
        logger.info(
            "Semantic span #%d: [%.3f, %.3f) duration=%.3fs evidence_windows=%d.",
            index, span.start, span.end, span.end - span.start,
            len(set(span.window_indices)),
        )
    # Padding may extend into a gap, but never across a different-tier region.
    padded: list[_Span] = []
    for index, span in enumerate(merged):
        previous_other = [
            other.end for other in merged[:index]
            if other.tier != span.tier and other.end <= span.start
        ]
        next_other = [
            other.start for other in merged[index + 1:]
            if other.tier != span.tier and other.start >= span.end
        ]
        padded_start = max(0.0, span.start - settings.padding_seconds)
        padded_end = min(float(clip_duration), span.end + settings.padding_seconds)
        if previous_other:
            padded_start = max(padded_start, max(previous_other))
        if next_other:
            padded_end = min(padded_end, min(next_other))
        padded.append(
            _Span(padded_start, padded_end, list(span.window_indices), span.tier)
        )
    for index, span in enumerate(padded, start=1):
        logger.info(
            "Padded span #%d: [%.3f, %.3f) duration=%.3fs evidence_windows=%d.",
            index, span.start, span.end, span.end - span.start,
            len(set(span.window_indices)),
        )
    # Padding may join regions of the same tier, but must not erase a tier boundary.
    final_spans = _merge_spans(padded, max_gap=0.0)
    logger.info(
        "Candidate spans after padding merge: %d (padding=%.3fs).",
        len(final_spans), settings.padding_seconds,
    )
    for index, span in enumerate(final_spans, start=1):
        logger.info(
            "Final candidate span #%d: [%.3f, %.3f) duration=%.3fs "
            "evidence_windows=%d.",
            index, span.start, span.end, span.end - span.start,
            len(set(span.window_indices)),
        )

    candidates: list[CandidateClip] = []
    for span in final_spans:
        start = min(max(span.start, 0.0), float(clip_duration))
        end = min(max(span.end, 0.0), float(clip_duration))
        if end - start < settings.min_candidate_duration:
            continue
        unique_indices = sorted(set(span.window_indices))
        contributing_scores = scores[unique_indices]
        tier = span.tier
        candidates.append(CandidateClip(
            start=start,
            end=end,
            duration=end - start,
            tier=tier,
            mean_score=float(np.mean(contributing_scores)),
            max_score=float(np.max(contributing_scores)),
            window_evidence=tuple(
                WindowEvidence(
                    start=float(timestamps[index, 0]),
                    end=float(timestamps[index, 1]),
                    score=float(scores[index]),
                    tier=accepted_tiers[index],
                    p0=float(p0[index]),
                    p1=float(p1[index]),
                    p2=float(p2[index]),
                )
                for index in unique_indices
            ),
        ))
    logger.info(
        "Candidate generation complete: accepted_windows=%d semantic_spans=%d "
        "final_spans=%d emitted_candidates=%d.",
        len(accepted), len(merged), len(final_spans), len(candidates),
    )
    return candidates
