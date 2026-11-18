"""Framework-independent candidate value objects shared by selection paths."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WindowEvidence:
    """Window-level score evidence retained for candidate ranking."""

    start: float
    end: float
    probability: float


@dataclass(frozen=True)
class CandidateClip:
    """A source-time candidate segment with its supporting window evidence."""

    start: float
    end: float
    duration: float
    mean_probability: float
    max_probability: float
    window_evidence: tuple[WindowEvidence, ...] = ()
