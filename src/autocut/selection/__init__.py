"""Candidate generation, deterministic selection, and timeline assembly."""

from .candidates_generation import (
    CandidateClip,
    CandidateGenerationConfig,
    WindowEvidence,
    generate_candidates,
)
from .selection import (
    MandatoryClip,
    SelectionCandidate,
    SelectionResult,
    select_candidates,
)
from .timeline import Timeline, TimelineClip, assemble_timeline

__all__ = [
    "CandidateClip",
    "CandidateGenerationConfig",
    "WindowEvidence",
    "generate_candidates",
    "MandatoryClip",
    "SelectionCandidate",
    "SelectionResult",
    "select_candidates",
    "Timeline",
    "TimelineClip",
    "assemble_timeline",
]
