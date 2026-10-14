"""Tests for budget-free candidate selection."""
import pytest
from autocut.selection.selection import MandatoryClip, SelectionCandidate, select_candidates
from autocut.selection.candidates_generation import CandidateClip, WindowEvidence

def candidate(video="v1.mp4", order=0, start=0.0, end=4.0, score=1.0, evidence=(), annotation_order=None):
    return SelectionCandidate(video, order, CandidateClip(start, end, end-start, "high_action", score, score, tuple(evidence)), annotation_order)

def test_preserves_all_non_overlapping_candidates_regardless_of_total_duration():
    long = candidate(start=0, end=284.68, score=2.2)
    other = candidate(video="v2.mp4", order=1, start=0, end=35, score=2.0)
    third = candidate(video="v3.mp4", order=2, start=0, end=19, score=1.0)
    result = select_candidates([long, other, third])
    assert set(result.selected) == {long, other, third}
    assert sum(item.duration for item in result.selected) == pytest.approx(338.68)

def test_deterministic_priority_and_duplicate_source_overlap_suppression():
    lower = candidate(start=10, end=14, score=1.5)
    higher = candidate(start=0, end=4, score=2.0)
    duplicate = candidate(start=2, end=6, score=1.0)
    different_video = candidate(video="v2.mp4", order=1, start=2, end=6, score=1.0)
    result = select_candidates([lower, duplicate, different_video, higher])
    assert result.selected == (higher, lower, different_video)

def test_candidate_overlapping_mandatory_interval_is_not_selected():
    mandatory = [MandatoryClip("v1.mp4", 5, 10, video_order=0, sequence_order=0)]
    overlapping = candidate(start=3, end=7, score=2.5)
    independent = candidate(start=11, end=15, score=1.0)
    result = select_candidates([overlapping, independent], mandatory)
    assert result.selected == (independent,)

def test_max_clip_duration_is_independent_per_clip_constraint():
    evidence = (
        WindowEvidence(0, 4, 0.8, "moderate_interest"),
        WindowEvidence(4, 8, 1.2, "moderate_interest"),
        WindowEvidence(8, 12, 2.2, "high_action"),
    )
    item = candidate(start=0, end=12, score=2.2, evidence=evidence)
    result = select_candidates([item], min_clip_duration=3, max_clip_duration=6)
    assert len(result.selected) == 1
    assert result.selected[0].duration == 6
    assert result.selected[0].end == 12
    assert result.selected[0].window_evidence == evidence[1:]

def test_max_clip_duration_below_minimum_drops_only_that_candidate():
    result = select_candidates([candidate(start=0, end=12)], min_clip_duration=3, max_clip_duration=2)
    assert result.selected == ()

def test_selection_validates_types_and_durations():
    with pytest.raises(TypeError):
        select_candidates([object()])
    with pytest.raises(ValueError):
        select_candidates([], min_clip_duration=0)
    with pytest.raises(ValueError):
        select_candidates([], max_clip_duration=-1)



def test_duration_budget_prioritizes_high_action_then_moderate_by_score():
    high_low = candidate(start=0, end=4, score=1.5)
    high_best = candidate(start=10, end=14, score=2.2)
    moderate_best = SelectionCandidate(
        "v2.mp4", 1,
        CandidateClip(0, 4, 4, "moderate_interest", 1.0, 1.0),
    )
    moderate_next = SelectionCandidate(
        "v3.mp4", 2,
        CandidateClip(0, 4, 4, "moderate_interest", 0.8, 0.8),
    )
    result = select_candidates(
        [moderate_next, high_low, moderate_best, high_best],
        max_total_duration=12.0,
    )
    assert result.selected == (high_best, high_low, moderate_best)


def test_duration_budget_preserves_mandatory_and_uses_remaining_space():
    mandatory = MandatoryClip("v1.mp4", 0, 6, video_order=0)
    fits = candidate(video="v2.mp4", order=1, start=0, end=4, score=2.0)
    too_long = candidate(video="v3.mp4", order=2, start=0, end=5, score=2.5)
    result = select_candidates(
        [too_long, fits], [mandatory], max_total_duration=10.0,
    )
    assert result.selected == (fits,)


def test_mandatory_duration_over_budget_keeps_mandatory_and_selects_no_candidates(caplog):
    mandatory = MandatoryClip("v1.mp4", 0, 8, video_order=0)
    optional = candidate(video="v2.mp4", order=1, start=0, end=4, score=2.0)
    with caplog.at_level("WARNING"):
        result = select_candidates(
            [optional], [mandatory], max_total_duration=5.0,
        )
    assert result.selected == ()
    assert "exceeding soundtrack duration" in caplog.text
