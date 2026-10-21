"""Tests for probability-ranked binary candidate selection."""

import pytest

from autocut.selection.selection import MandatoryClip, SelectionCandidate, select_candidates
from autocut.selection.candidates_generation import CandidateClip, WindowEvidence


def candidate(video="v1.mp4", order=0, start=0.0, end=4.0, score=0.8,
              mean_score=None, evidence=(), annotation_order=None):
    mean = score if mean_score is None else mean_score
    clip = CandidateClip(start, end, end-start, mean, score, tuple(evidence))
    return SelectionCandidate(video, order, clip, annotation_order)


def test_preserves_all_non_overlapping_candidates_without_duration_budget():
    long = candidate(start=0, end=284.68, score=0.9)
    other = candidate(video="v2.mp4", order=1, start=0, end=35, score=0.8)
    third = candidate(video="v3.mp4", order=2, start=0, end=19, score=0.7)
    result = select_candidates([long, other, third])
    assert set(result.selected) == {long, other, third}
    assert sum(item.duration for item in result.selected) == pytest.approx(338.68)


def test_deterministic_priority_and_duplicate_source_overlap_suppression():
    lower = candidate(start=10, end=14, score=0.75)
    higher = candidate(start=0, end=4, score=0.95)
    duplicate = candidate(start=2, end=6, score=0.6)
    different_video = candidate(video="v2.mp4", order=1, start=2, end=6, score=0.6)
    result = select_candidates([lower, duplicate, different_video, higher])
    assert result.selected == (higher, lower, different_video)


def test_candidate_overlapping_mandatory_interval_is_not_selected():
    mandatory = [MandatoryClip("v1.mp4", 5, 10, video_order=0, sequence_order=0)]
    overlapping = candidate(start=3, end=7, score=0.95)
    independent = candidate(start=11, end=15, score=0.6)
    result = select_candidates([overlapping, independent], mandatory)
    assert result.selected == (independent,)


def test_max_clip_duration_trims_around_strongest_evidence():
    evidence = (
        WindowEvidence(0, 4, 0.4),
        WindowEvidence(4, 8, 0.7),
        WindowEvidence(8, 12, 0.95),
    )
    item = candidate(start=0, end=12, score=0.95, mean_score=0.6833, evidence=evidence)
    result = select_candidates([item], min_clip_duration=3, max_clip_duration=6)
    assert len(result.selected) == 1
    trimmed = result.selected[0]
    assert trimmed.duration == pytest.approx(6)
    assert trimmed.end == pytest.approx(12)
    assert trimmed.window_evidence == evidence[1:]


def test_max_clip_duration_below_minimum_drops_only_that_candidate():
    result = select_candidates([candidate(start=0, end=12)], min_clip_duration=3,
                               max_clip_duration=2)
    assert result.selected == ()


def test_selection_validates_types_and_durations():
    with pytest.raises(TypeError):
        select_candidates([object()])
    with pytest.raises(ValueError):
        select_candidates([], min_clip_duration=0)
    with pytest.raises(ValueError):
        select_candidates([], max_clip_duration=-1)


def test_blended_max_and_mean_probability_determine_priority():
    # Candidate A has a higher peak, but B has stronger sustained evidence.
    peak_only = candidate(start=0, end=4, score=0.95, mean_score=0.2)
    sustained = candidate(start=10, end=14, score=0.8, mean_score=0.7)
    result = select_candidates([peak_only, sustained])
    assert result.selected == (sustained, peak_only)


def test_duration_budget_uses_probability_priority():
    lower = candidate(start=0, end=4, score=0.7)
    higher = candidate(start=10, end=14, score=0.95)
    moderate = candidate(video="v2.mp4", order=1, start=0, end=4, score=0.8)
    result = select_candidates([moderate, lower, higher], max_total_duration=8.0)
    assert result.selected == (higher, moderate)


def test_duration_budget_preserves_mandatory_and_uses_remaining_space():
    mandatory = MandatoryClip("v1.mp4", 0, 6, video_order=0)
    fits = candidate(video="v2.mp4", order=1, start=0, end=4, score=0.8)
    too_long = candidate(video="v3.mp4", order=2, start=0, end=5, score=0.99)
    result = select_candidates([too_long, fits], [mandatory], max_total_duration=10.0)
    assert result.selected == (fits,)


def test_mandatory_duration_over_budget_keeps_mandatory_and_selects_no_candidates(caplog):
    mandatory = MandatoryClip("v1.mp4", 0, 8, video_order=0)
    optional = candidate(video="v2.mp4", order=1, start=0, end=4, score=0.9)
    with caplog.at_level("WARNING"):
        result = select_candidates([optional], [mandatory], max_total_duration=5.0)
    assert result.selected == ()
    assert "exceeding soundtrack duration" in caplog.text
