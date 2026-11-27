"""LightGBM candidate ranking, deterministic selection, and timeline assembly."""

from .candidate_types import CandidateClip, WindowEvidence
from .selection import MandatoryClip, SelectionCandidate, SelectionResult, select_candidates
from .candidates import RankedSegment, RankedWindow, build_candidate_clips, select_ranked_segments
from .timeline import Timeline, TimelineClip, assemble_timeline

__all__ = [
    "CandidateClip", "WindowEvidence",
    "MandatoryClip", "SelectionCandidate", "SelectionResult", "select_candidates",
    "RankedSegment", "RankedWindow", "select_ranked_segments", "build_candidate_clips",
    "Timeline", "TimelineClip", "assemble_timeline",
]
