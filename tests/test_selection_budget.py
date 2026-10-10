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
    mandatory = [MandatoryClip("v1.mp4", 0, 12, sequence_order=0)]
    with caplog.at_level(logging.WARNING):
        result = select_highlight_budget([candidate()], 10, mandatory)
    assert result.selected == ()
    assert result.optional_budget == 0
    assert result.mandatory_over_budget
    assert "exceeding target duration" in caplog.text


def test_mandatory_duration_is_subtracted_from_target():
    mandatory = [MandatoryClip("v1.mp4", 0, 3, sequence_order=0)]
    result = select_highlight_budget(
        [candidate(start=0, end=4, score=2), candidate(video="v2.mp4", order=1, start=0, end=5, score=1)],
        10,
        mandatory,
    )
    assert result.optional_budget == 7
    assert result.selected_duration == 4


def test_selector_validates_duration_and_types():
    with pytest.raises(ValueError):
        select_highlight_budget([], 0)
    with pytest.raises(TypeError):
        select_highlight_budget([object()], 10)


def test_candidate_window_evidence_is_carried_for_later_refinement():
    evidence = (WindowEvidence(0, 4, 2.0, "high_action"),)
    item = candidate(start=0, end=4, score=2.0, evidence=evidence)
    assert item.window_evidence == evidence
