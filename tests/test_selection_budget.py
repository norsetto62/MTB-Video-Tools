"""Tests for Phase 10 greedy budget selection."""
import logging

import pytest

from autocut.selection.budget import (
    MandatoryClip,
    SelectionCandidate,
    select_highlight_budget,
)
from autocut.selection.candidates_generation import CandidateClip, WindowEvidence


def candidate(video="v1.mp4", order=0, start=0.0, end=4.0, score=1.0, mean=None, evidence=()):
    duration = end - start
    return SelectionCandidate(
        video_name=video,
        video_order=order,
        candidate=CandidateClip(
            start=start, end=end, duration=duration, tier="high_action",
            mean_score=score if mean is None else mean, max_score=score,
            window_evidence=tuple(evidence),
        ),
    )


def test_greedy_selects_by_score_but_skips_oversized_candidate():
    a = candidate(start=0, end=12, score=2.2)
    b = candidate(video="v2.mp4", order=1, start=0, end=4, score=2.0)
    c = candidate(video="v3.mp4", order=2, start=0, end=3, score=2.0)
    result = select_highlight_budget([a, b, c], 10)
    assert result.selected == (c, b)
    assert result.selected_duration == 7
    assert result.unused_optional_budget == 3


def test_equal_score_prefers_shorter_then_source_order():
    long = candidate(start=10, end=14, score=1.5)
    short = candidate(video="v2.mp4", order=1, start=0, end=2, score=1.5)
    earlier = candidate(video="v1.mp4", order=0, start=0, end=2, score=1.5)
    result = select_highlight_budget([long, short, earlier], 4)
    assert result.selected == (earlier, short)


def test_zero_optional_budget_when_mandatory_exceeds_target(caplog):
    mandatory = [MandatoryClip("v1.mp4", 0, 12, video_order=0, sequence_order=0)]
    with caplog.at_level(logging.WARNING):
        result = select_highlight_budget([candidate()], 10, mandatory)
    assert result.selected == ()
    assert result.optional_budget == 0
    assert result.mandatory_over_budget
    assert "exceeding target duration" in caplog.text


def test_mandatory_duration_is_subtracted_from_target():
    mandatory = [MandatoryClip("v1.mp4", 0, 3, video_order=0, sequence_order=0)]
    result = select_highlight_budget(
        [candidate(start=0, end=4, score=2), candidate(video="v2.mp4", order=1, start=0, end=5, score=1)],
        10,
        mandatory,
    )
    assert result.optional_budget == 7
    # The first candidate overlaps the mandatory interval in v1.mp4 and is
    # excluded to avoid duplicating that source footage. The v2.mp4 candidate
    # therefore wins and fits within the remaining 7-second budget.
    assert result.selected_duration == 5
    assert [item.video_name for item in result.selected] == ["v2.mp4"]


def test_selector_validates_duration_and_types():
    with pytest.raises(ValueError):
        select_highlight_budget([], 0)
    with pytest.raises(TypeError):
        select_highlight_budget([object()], 10)


def test_candidate_window_evidence_is_carried_for_later_refinement():
    evidence = (WindowEvidence(0, 4, 2.0, "high_action"),)
    item = candidate(start=0, end=4, score=2.0, evidence=evidence)
    assert item.window_evidence == evidence



def test_oversized_candidate_is_trimmed_around_best_window():
    evidence = (
        WindowEvidence(0, 4, 0.8, "moderate_interest"),
        WindowEvidence(4, 8, 1.2, "moderate_interest"),
        WindowEvidence(8, 12, 2.2, "high_action"),
    )
    item = candidate(start=0, end=12, score=2.2, evidence=evidence)
    result = select_highlight_budget([item], 10)
    assert len(result.selected) == 1
    trimmed = result.selected[0]
    assert (trimmed.start, trimmed.end, trimmed.duration) == (2.0, 12.0, 10.0)
    assert trimmed.score == 2.2
    assert result.selected_duration == 10.0
    assert result.unused_optional_budget == 0.0


def test_oversized_candidate_without_evidence_is_skipped():
    result = select_highlight_budget([candidate(start=0, end=12, score=2.2)], 10)
    assert result.selected == ()
    assert result.unused_optional_budget == 10.0


def test_too_small_remaining_budget_does_not_create_tiny_trim():
    evidence = (WindowEvidence(0, 4, 2.2, "high_action"),)
    item = candidate(start=0, end=4, score=2.2, evidence=evidence)
    result = select_highlight_budget([item], 2)
    assert result.selected == ()



def test_candidate_overlapping_mandatory_interval_is_not_selected():
    mandatory = [MandatoryClip("v1.mp4", 5, 10, video_order=0)]
    overlapping = candidate(start=3, end=7, score=2.5)
    independent = candidate(start=11, end=15, score=1.0)
    result = select_highlight_budget([overlapping, independent], 12, mandatory)
    assert result.selected == (independent,)



def test_max_clip_duration_trims_around_strongest_evidence():
    evidence = (
        WindowEvidence(0, 4, 0.8, "moderate_interest"),
        WindowEvidence(4, 8, 1.2, "moderate_interest"),
        WindowEvidence(8, 12, 2.2, "high_action"),
    )
    result = select_highlight_budget(
        [candidate(start=0, end=12, score=2.2, evidence=evidence)],
        20, max_clip_duration=6,
    )
    trimmed = result.selected[0]
    assert (trimmed.start, trimmed.end, trimmed.duration) == (6.0, 12.0, 6.0)
    assert trimmed.score == 2.2
    assert trimmed.window_evidence == evidence[1:]


def test_max_clip_duration_uses_centered_fallback_without_evidence():
    result = select_highlight_budget(
        [candidate(start=0, end=12, score=2.2)], 20, max_clip_duration=6,
    )
    trimmed = result.selected[0]
    assert (trimmed.start, trimmed.end, trimmed.duration) == (3.0, 9.0, 6.0)
    assert trimmed.window_evidence == ()


def test_max_clip_duration_below_minimum_skips_optional_candidate():
    result = select_highlight_budget(
        [candidate(start=0, end=12, score=2.2)], 20,
        min_trimmed_duration=3, max_clip_duration=2,
    )
    assert result.selected == ()
    assert result.unused_optional_budget == 20


def test_max_clip_duration_zero_disables_limit():
    item = candidate(start=0, end=12, score=2.2)
    result = select_highlight_budget([item], 20, max_clip_duration=0)
    assert result.selected == (item,)


def test_mandatory_clips_are_exempt_from_max_clip_duration():
    mandatory = [MandatoryClip("must.mp4", 2, 12, video_order=0)]
    result = select_highlight_budget(
        [candidate(video="optional.mp4", order=1, start=0, end=12, score=2.2)],
        20, mandatory, max_clip_duration=4,
    )
    assert result.mandatory_duration == 10
    assert len(result.selected) == 1
    assert result.selected[0].duration == 4


def test_max_clip_trim_does_not_rescue_candidate_overlapping_mandatory():
    mandatory = [MandatoryClip("v1.mp4", 5, 10, video_order=0)]
    evidence = (WindowEvidence(0, 4, 2.2, "high_action"),)
    item = candidate(start=0, end=12, score=2.2, evidence=evidence)

    result = select_highlight_budget(
        [item], 20, mandatory, max_clip_duration=4,
    )

    assert result.selected == ()
    assert result.mandatory_duration == 5
