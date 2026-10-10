"""Tests for Phase 9 candidate generation."""

import numpy as np
import pytest

from autocut.candidates_generation import (
    CandidateGenerationConfig,
    generate_candidates,
)
from autocut.ml.inference import InferenceResult


def _inference(intervals, probabilities):
    return InferenceResult(
        timestamps=np.asarray(intervals, dtype=np.float64),
        probabilities=np.asarray(probabilities, dtype=np.float32),
    )


def _action():
    return [0.20, 0.20, 0.60]


def _moderate():
    return [0.20, 0.70, 0.10]


def test_overlapping_accepted_windows_union_and_padding_clamps_to_video():
    result = _inference(
        [[0, 4], [2, 6], [4, 8]],
        [_action(), _action(), _action()],
    )

    clips = generate_candidates(result, clip_duration=8)

    assert len(clips) == 1
    assert (clips[0].start, clips[0].end, clips[0].duration) == (0.0, 8.0, 8.0)
    assert clips[0].tier == "high_action"
    assert clips[0].mean_score == pytest.approx(1.7)
    assert clips[0].max_score == pytest.approx(1.7)


def test_junk_gate_takes_precedence_over_high_action_and_score():
    result = _inference(
        [[0, 4]],
        [[0.55, 0.05, 0.40]],
    )

    assert generate_candidates(result, 10) == []


def test_action_threshold_equality_is_high_action():
    result = _inference([[0, 4]], [[0.20, 0.30, 0.50]])

    clips = generate_candidates(result, 10)

    assert len(clips) == 1
    assert clips[0].tier == "high_action"


def test_moderate_interest_is_accepted_when_score_meets_threshold():
    # S = 0.8 * 1.0 + 0.0 * 2.5 = 0.8; P0 remains below junk threshold.
    result = _inference([[0, 4]], [[0.20, 0.80, 0.0]])

    clips = generate_candidates(result, 10)

    assert len(clips) == 1
    assert clips[0].tier == "moderate_interest"
    assert clips[0].mean_score == pytest.approx(0.8)


def test_short_span_is_removed_before_padding():
    result = _inference([[3, 5]], [_action()])
    config = CandidateGenerationConfig(
        min_candidate_duration=4.0,
        padding_seconds=2.0,
    )

    assert generate_candidates(result, 10, config=config) == []


def test_gap_shorter_than_limit_is_bridged():
    result = _inference(
        [[0, 2], [4, 6]],
        [_action(), _action()],
    )
    config = CandidateGenerationConfig(
        min_candidate_duration=4.0,
        max_gap_to_merge=3.0,
        padding_seconds=0.0,
    )

    clips = generate_candidates(result, 10, config=config)

    assert len(clips) == 1
    assert (clips[0].start, clips[0].end, clips[0].duration) == (0.0, 6.0, 6.0)


def test_gap_equal_to_limit_is_not_bridged():
    result = _inference(
        [[0, 2], [5, 7]],
        [_action(), _action()],
    )
    config = CandidateGenerationConfig(
        min_candidate_duration=4.0,
        max_gap_to_merge=3.0,
        padding_seconds=0.0,
    )

    assert generate_candidates(result, 10, config=config) == []


def test_padding_can_merge_surviving_clips_and_scores_are_combined():
    result = _inference(
        [[0, 4], [7, 11]],
        [_moderate(), _action()],
    )
    config = CandidateGenerationConfig(
        min_candidate_duration=4.0,
        max_gap_to_merge=0.0,
        padding_seconds=2.0,
    )

    clips = generate_candidates(result, 15, config=config)

    assert len(clips) == 1
    assert (clips[0].start, clips[0].end, clips[0].duration) == (0.0, 13.0, 13.0)
    assert clips[0].tier == "high_action"
    assert clips[0].mean_score == pytest.approx((0.8 + 1.7) / 2)
    assert clips[0].max_score == pytest.approx(1.7)


def test_candidates_are_chronological_and_boundary_clamped():
    result = _inference(
        [[20, 24], [2, 6]],
        [_action(), _action()],
    )
    # Input windows must be sorted chronologically, so this malformed order
    # is rejected rather than silently reordered.
    with pytest.raises(ValueError, match="strictly increasing"):
        generate_candidates(result, 30)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"junk_threshold": -0.1},
        {"junk_threshold": 1.1},
        {"action_threshold": float("nan")},
        {"score_threshold": -0.1},
        {"min_candidate_duration": 0},
        {"max_gap_to_merge": -1},
        {"padding_seconds": float("inf")},
        {"class_score_weights": (1.0, 2.0)},
    ],
)
def test_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        CandidateGenerationConfig(**kwargs)


@pytest.mark.parametrize("duration", [0, -1, float("nan"), float("inf")])
def test_rejects_invalid_video_duration(duration):
    with pytest.raises(ValueError, match="clip_duration"):
        generate_candidates(_inference([[0, 4]], [_action()]), duration)


def test_empty_inference_returns_no_candidates():
    result = _inference([], np.empty((0, 3), dtype=np.float32))
    assert generate_candidates(result, 10) == []


def test_invalid_probability_rows_are_rejected():
    result = _inference([[0, 4]], [[0.2, 0.2, 0.2]])
    with pytest.raises(ValueError, match="sum to 1"):
        generate_candidates(result, 10)
