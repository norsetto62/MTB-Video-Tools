"""Deterministic tier-prioritized selection within an optional duration budget."""
from __future__ import annotations
from dataclasses import dataclass
import logging
import math
from typing import Sequence
from .candidates_generation import CandidateClip, WindowEvidence
logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class SelectionCandidate:
    """A generated candidate with source identity and annotation ordering."""
    video_name: str
    video_order: int
    candidate: CandidateClip
    annotation_order: int | None = None
    @property
    def start(self) -> float: return self.candidate.start
    @property
    def end(self) -> float: return self.candidate.end
    @property
    def duration(self) -> float: return self.candidate.duration
    @property
    def score(self) -> float: return self.candidate.max_score
    @property
    def mean_score(self) -> float: return self.candidate.mean_score
    @property
    def tier(self) -> str: return self.candidate.tier
    @property
    def window_evidence(self) -> tuple[WindowEvidence, ...]: return self.candidate.window_evidence

@dataclass(frozen=True)
class MandatoryClip:
    """A user-requested source interval that must be preserved in the edit."""
    video_name: str
    start: float
    end: float
    video_order: int
    sequence_order: int = 0
    @property
    def duration(self) -> float: return self.end - self.start

@dataclass(frozen=True)
class SelectionResult:
    """Optional candidates retained after priority, duration, and overlap filtering."""
    selected: tuple[SelectionCandidate, ...]

def _validate_interval(start: float, end: float, label: str) -> None:
    if isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
        raise ValueError(f"{label} interval must have numeric endpoints.")
    if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
        raise ValueError(f"{label} interval must be finite, non-negative, and end after start.")

def _trim_to_duration(item: SelectionCandidate, duration: float, minimum: float) -> SelectionCandidate | None:
    """Honor an explicit per-clip cap around strongest evidence, not an output budget."""
    if duration < minimum or duration >= item.duration: return None
    evidence = sorted(item.window_evidence, key=lambda w: (-w.score, w.start, w.end))
    midpoint = (evidence[0].start + evidence[0].end) / 2.0 if evidence else (item.start + item.end) / 2.0
    start = min(max(midpoint - duration / 2.0, item.start), item.end - duration)
    end = start + duration
    retained = tuple(w for w in item.window_evidence if w.end > start and w.start < end)
    if evidence and not retained: return None
    if retained:
        scores = [w.score for w in retained]
        tier = "high_action" if any(w.tier == "high_action" for w in retained) else "moderate_interest"
        mean_score, max_score = sum(scores) / len(scores), max(scores)
    else:
        tier, mean_score, max_score = item.tier, item.mean_score, item.score
    candidate = CandidateClip(start, end, end - start, tier, mean_score, max_score, retained)
    return SelectionCandidate(item.video_name, item.video_order, candidate, item.annotation_order)

def select_candidates(candidates: Sequence[SelectionCandidate], mandatory_clips: Sequence[MandatoryClip] = (), *, min_clip_duration: float = 3.0, max_clip_duration: float = 0.0, max_total_duration: float | None = None) -> SelectionResult:
    """Select mandatory clips first, then high-action and moderate candidates.

    When supplied, max_total_duration is the total edit-duration limit,
    normally the soundtrack duration. Mandatory material is never dropped;
    if it exhausts the budget, optional candidates are not added. Optional
    candidates are ranked by descending max score within each tier, and only
    candidates that fit the remaining duration are retained. Overlapping
    source intervals are suppressed to avoid repeating frames.
    max_clip_duration remains an independent per-clip constraint.
    """
    for value, name, allow_zero in ((min_clip_duration, "min_clip_duration", False), (max_clip_duration, "max_clip_duration", True)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or (not allow_zero and value == 0):
            qualifier = "non-negative" if allow_zero else "positive"
            raise ValueError(f"{name} must be finite and {qualifier}.")
    if max_total_duration is not None and (
        isinstance(max_total_duration, bool)
        or not isinstance(max_total_duration, (int, float))
        or not math.isfinite(max_total_duration)
        or max_total_duration <= 0
    ):
        raise ValueError("max_total_duration must be finite and positive, or None.")
    candidate_list, mandatory_list = tuple(candidates), tuple(mandatory_clips)
    for item in candidate_list:
        if not isinstance(item, SelectionCandidate): raise TypeError("candidates must contain SelectionCandidate instances.")
        if not item.video_name: raise ValueError("Candidate video_name must not be empty.")
        if isinstance(item.video_order, bool) or not isinstance(item.video_order, int) or item.video_order < 0: raise ValueError("Candidate video_order must be a non-negative integer.")
        if item.annotation_order is not None and (isinstance(item.annotation_order, bool) or not isinstance(item.annotation_order, int) or item.annotation_order < 0): raise ValueError("Candidate annotation_order must be a non-negative integer or None.")
        _validate_interval(item.start, item.end, "Candidate")
        if not math.isfinite(item.score) or not math.isfinite(item.mean_score) or item.score < 0 or item.mean_score < 0: raise ValueError("Candidate scores must be finite and non-negative.")
        if not math.isclose(item.duration, item.end - item.start, rel_tol=0.0, abs_tol=1e-6): raise ValueError("Candidate duration must equal end - start.")
    for item in mandatory_list:
        if not isinstance(item, MandatoryClip): raise TypeError("mandatory_clips must contain MandatoryClip instances.")
        if not item.video_name: raise ValueError("Mandatory video_name must not be empty.")
        _validate_interval(item.start, item.end, "Mandatory")
        if isinstance(item.video_order, bool) or not isinstance(item.video_order, int) or item.video_order < 0: raise ValueError("Mandatory video_order must be a non-negative integer.")
        if isinstance(item.sequence_order, bool) or not isinstance(item.sequence_order, int) or item.sequence_order < 0: raise ValueError("Mandatory sequence_order must be a non-negative integer.")
    mandatory_duration = sum(item.duration for item in mandatory_list)
    remaining = None if max_total_duration is None else max(0.0, max_total_duration - mandatory_duration)
    if max_total_duration is not None and mandatory_duration > max_total_duration + 1e-9:
        logger.warning(
            "Mandatory clips total %.3fs, exceeding soundtrack duration %.3fs; "
            "preserving mandatory clips and selecting no optional candidates.",
            mandatory_duration, max_total_duration,
        )
    tier_order = {"high_action": 0, "moderate_interest": 1}
    ranked = sorted(candidate_list, key=lambda item: (
        tier_order.get(item.tier, 2), -item.score, item.duration,
        item.video_order, item.start, item.video_name,
        item.annotation_order if item.annotation_order is not None else -1,
    ))
    selected: list[SelectionCandidate] = []
    for original in ranked:
        if max_total_duration is not None and mandatory_duration > max_total_duration + 1e-9:
            break
        if any(original.video_name == m.video_name and original.start < m.end and m.start < original.end for m in mandatory_list):
            logger.info("Skipping candidate %s [%.3f, %.3f): overlaps mandatory material.", original.video_name, original.start, original.end)
            continue
        item = original
        if max_clip_duration > 0 and item.duration > max_clip_duration + 1e-9:
            item = _trim_to_duration(item, max_clip_duration, min_clip_duration)
            if item is None:
                logger.info("Skipping candidate: cannot satisfy maximum clip duration %.3fs and minimum duration %.3fs.", max_clip_duration, min_clip_duration)
                continue
        if any(item.video_name == previous.video_name and item.start < previous.end and previous.start < item.end for previous in selected):
            logger.info("Skipping duplicate source-time candidate %s [%.3f, %.3f): overlaps a retained candidate.", item.video_name, item.start, item.end)
            continue
        if remaining is not None and item.duration > remaining + 1e-9:
            logger.info("Skipping candidate %s [%.3f, %.3f): %.3fs exceeds remaining duration %.3fs.", item.video_name, item.start, item.end, item.duration, remaining)
            continue
        selected.append(item)
        if remaining is not None:
            remaining = max(0.0, remaining - item.duration)
        logger.info("Retained candidate: video=%s annotation_order=%s tier=%s range=[%.3f, %.3f) duration=%.3fs score=%.6f evidence=%d.", item.video_name, item.annotation_order, item.tier, item.start, item.end, item.duration, item.score, len(item.window_evidence))
    return SelectionResult(tuple(selected))
