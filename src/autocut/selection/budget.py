"""Greedy global selection of candidate clips under a duration budget.

Selection priority and playback order are deliberately separate. This module
chooses candidates; timeline assembly restores the user's source order.
"""
from __future__ import annotations

from dataclasses import dataclass
import logging
import math
from typing import Sequence

from .candidates_generation import CandidateClip, WindowEvidence

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SelectionCandidate:
    """A Phase 9 candidate annotated with source identity and input ordering."""

    video_name: str
    video_order: int
    candidate: CandidateClip

    @property
    def start(self) -> float:
        return self.candidate.start

    @property
    def end(self) -> float:
        return self.candidate.end

    @property
    def duration(self) -> float:
        return self.candidate.duration

    @property
    def score(self) -> float:
        return self.candidate.max_score

    @property
    def mean_score(self) -> float:
        return self.candidate.mean_score

    @property
    def tier(self) -> str:
        return self.candidate.tier

    @property
    def window_evidence(self) -> tuple[WindowEvidence, ...]:
        return self.candidate.window_evidence


@dataclass(frozen=True)
class MandatoryClip:
    """A user-requested source interval that must be preserved in the edit."""

    video_name: str
    start: float
    end: float
    video_order: int
    sequence_order: int = 0

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class SelectionResult:
    """Selected optional clips and duration accounting."""

    selected: tuple[SelectionCandidate, ...]
    target_duration: float
    mandatory_duration: float
    optional_budget: float
    selected_duration: float
    unused_optional_budget: float
    mandatory_over_budget: bool

    @property
    def projected_duration(self) -> float:
        return self.mandatory_duration + self.selected_duration

    @property
    def over_budget_by(self) -> float:
        return max(0.0, self.projected_duration - self.target_duration)


def _finite_nonnegative(value: float, name: str, *, allow_zero: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number.")
    value = float(value)
    if not math.isfinite(value) or value < 0 or (not allow_zero and value == 0):
        qualifier = "positive" if not allow_zero else "non-negative"
        raise ValueError(f"{name} must be finite and {qualifier}.")
    return value


def _validate_interval(start: float, end: float, label: str) -> None:
    if isinstance(start, bool) or isinstance(end, bool):
        raise ValueError(f"{label} interval must have numeric endpoints.")
    if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
        raise ValueError(f"{label} interval must have numeric endpoints.")
    if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
        raise ValueError(f"{label} interval must be finite, non-negative, and end after start.")


def _trim_to_budget(item: SelectionCandidate, available: float, minimum: float) -> SelectionCandidate | None:
    """Trim an oversized clip around its strongest retained inference window."""
    if available < minimum or not item.window_evidence:
        return None

    evidence = sorted(
        item.window_evidence,
        key=lambda window: (-window.score, window.start, window.end),
    )
    anchor = evidence[0]
    duration = min(available, item.duration)
    midpoint = (anchor.start + anchor.end) / 2.0
    start = min(max(midpoint - duration / 2.0, item.start), item.end - duration)
    end = start + duration
    retained = tuple(
        window for window in item.window_evidence
        if window.end > start and window.start < end
    )
    if not retained:
        return None

    # The trimmed clip's score must be derived from evidence overlapping the
    # retained interval, not copied from the untrimmed candidate.
    scores = [window.score for window in retained]
    tiers = {window.tier for window in retained}
    tier = "high_action" if "high_action" in tiers else "moderate_interest"
    trimmed = CandidateClip(
        start=start,
        end=end,
        duration=end - start,
        tier=tier,
        mean_score=sum(scores) / len(scores),
        max_score=max(scores),
        window_evidence=retained,
    )
    return SelectionCandidate(
        video_name=item.video_name,
        video_order=item.video_order,
        candidate=trimmed,
    )


def select_highlight_budget(
    candidates: Sequence[SelectionCandidate],
    target_duration: float,
    mandatory_clips: Sequence[MandatoryClip] = (),
    *,
    min_trimmed_duration: float = 3.0,
) -> SelectionResult:
    """Greedily choose highest-scoring candidates that fit the optional budget.

    Ranking is max_score descending, duration ascending, then source order and
    start time for deterministic ties. Oversized candidates are skipped rather
    than terminating the scan. Mandatory intervals consume budget first and
    are not eligible for removal. If mandatory content exceeds the target,
    a warning is emitted and the optional budget becomes zero.

    If the next candidate is too long for the remaining budget, it may be
    shortened around its highest-scoring retained inference window, provided
    the remainder is at least min_trimmed_duration and evidence is available.
    No inference is rerun and no candidates are generated recursively.
    """
    target = _finite_nonnegative(target_duration, "target_duration", allow_zero=False)
    min_trim = _finite_nonnegative(min_trimmed_duration, "min_trimmed_duration", allow_zero=False)
    candidate_list = tuple(candidates)
    mandatory_list = tuple(mandatory_clips)

    for item in candidate_list:
        if not isinstance(item, SelectionCandidate):
            raise TypeError("candidates must contain SelectionCandidate instances.")
        if not item.video_name:
            raise ValueError("Candidate video_name must not be empty.")
        if isinstance(item.video_order, bool) or not isinstance(item.video_order, int) or item.video_order < 0:
            raise ValueError("Candidate video_order must be a non-negative integer.")
        _validate_interval(item.start, item.end, "Candidate")
        if not math.isfinite(item.score) or not math.isfinite(item.mean_score):
            raise ValueError("Candidate scores must be finite.")
        if item.score < 0 or item.mean_score < 0:
            raise ValueError("Candidate scores must be non-negative.")
        if not math.isclose(item.duration, item.end - item.start, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError("Candidate duration must equal end - start.")

    for item in mandatory_list:
        if not isinstance(item, MandatoryClip):
            raise TypeError("mandatory_clips must contain MandatoryClip instances.")
        if not item.video_name:
            raise ValueError("Mandatory video_name must not be empty.")
        _validate_interval(item.start, item.end, "Mandatory")
        if isinstance(item.video_order, bool) or not isinstance(item.video_order, int) or item.video_order < 0:
            raise ValueError("Mandatory video_order must be a non-negative integer.")
        if isinstance(item.sequence_order, bool) or not isinstance(item.sequence_order, int) or item.sequence_order < 0:
            raise ValueError("Mandatory sequence_order must be a non-negative integer.")

    mandatory_duration = sum(item.duration for item in mandatory_list)
    optional_budget = max(0.0, target - mandatory_duration)
    over = mandatory_duration > target
    if over:
        logger.warning(
            "Mandatory clips total %.3fs, exceeding target duration %.3fs by %.3fs; "
            "preserving mandatory clips and selecting no optional clips.",
            mandatory_duration, target, mandatory_duration - target,
        )

    ranked = sorted(
        candidate_list,
        key=lambda item: (-item.score, item.duration, item.video_order, item.start, item.video_name),
    )
    selected: list[SelectionCandidate] = []
    selected_duration = 0.0
    remaining = optional_budget
    for item in ranked:
        # A tiny tolerance avoids rejecting exact fits due to float roundoff.
        chosen = item
        if item.duration > remaining + 1e-9:
            chosen = _trim_to_budget(item, remaining, min_trim)  # type: ignore[assignment]
            if chosen is None:
                continue
        selected.append(chosen)
        selected_duration += chosen.duration
        remaining = max(0.0, optional_budget - selected_duration)
        if remaining <= 1e-9:
            break

    return SelectionResult(
        selected=tuple(selected),
        target_duration=target,
        mandatory_duration=mandatory_duration,
        optional_budget=optional_budget,
        selected_duration=selected_duration,
        unused_optional_budget=max(0.0, optional_budget - selected_duration),
        mandatory_over_budget=over,
    )
