"""Candidate generation, greedy selection, and timeline assembly."""

from .candidates_generation import (
    CandidateClip,
    CandidateGenerationConfig,
    WindowEvidence,
    generate_candidates,
)
from .budget import (
    MandatoryClip,
    SelectionCandidate,
    SelectionResult,
    select_highlight_budget,
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
    "select_highlight_budget",
    "Timeline",
    "TimelineClip",
    "assemble_timeline",
]
