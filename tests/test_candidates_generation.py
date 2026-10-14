"""Tests for class-aware Phase 9 candidate generation."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from autocut.ml.inference import InferenceResult
from autocut.selection.candidates_generation import (
    CandidateGenerationConfig,
    generate_candidates,
)


def _inference(intervals, probabilities):
    return InferenceResult(
        timestamps=np.asarray(intervals, dtype=np.float64),
        probabilities=np.asarray(probabilities, dtype=np.float32),
    )


def _action():
    return [0.10, 0.20, 0.70]


def _flow():
    return [0.10, 0.60, 0.30]


def test_overlapping_accepted_windows_union_and_padding_clamps_to_video():
    result = _inference([[0, 4], [2, 6], [4, 8]], [_action()] * 3)
    clips = generate_candidates(result, clip_duration=8)
    assert len(clips) == 1
    assert (clips[0].start, clips[0].end, clips[0].duration) == (0.0, 8.0, 8.0)
    assert clips[0].tier == "high_action"
    assert clips[0].mean_score == pytest.approx(1.95)
    assert clips[0].max_score == pytest.approx(1.95)
    assert clips[0].window_evidence[0].p2 == pytest.approx(0.7)


def test_high_action_requires_clean_window_and_action_as_argmax():
    accepted = _inference([[0, 4]], [[0.10, 0.20, 0.70]])
    rejected_junk = _inference([[0, 4]], [[0.20, 0.10, 0.70]])
    rejected_not_argmax = _inference([[0, 4]], [[0.10, 0.35, 0.55]])
    config = CandidateGenerationConfig(padding_seconds=0.0)
    assert len(generate_candidates(accepted, 10, config=config)) == 1
    assert generate_candidates(rejected_junk, 10, config=config) == []
    assert generate_candidates(rejected_not_argmax, 10, config=config) == []


def test_flow_eligibility_is_class_aware_not_weighted_score_only():
    good_flow = _inference([[0, 4]], [_flow()])
    junk_dominated = _inference([[0, 4]], [[0.60, 0.30, 0.10]])
    weak_flow = _inference([[0, 4]], [[0.10, 0.35, 0.55]])
    config = CandidateGenerationConfig(padding_seconds=0.0)
    clips = generate_candidates(good_flow, 10, config=config)
    assert len(clips) == 1
    assert clips[0].tier == "moderate_interest"
    assert generate_candidates(junk_dominated, 10, config=config) == []
    assert generate_candidates(weak_flow, 10, config=config) == []


def test_flow_accepts_p1_greater_than_p2_even_when_class_argmax_is_not_flow():
    result = _inference([[0, 4]], [[0.15, 0.44, 0.41]])
    config = CandidateGenerationConfig(padding_seconds=0.0)
    clips = generate_candidates(result, 10, config=config)
    assert len(clips) == 1
    assert clips[0].tier == "moderate_interest"


def test_min_clip_is_applied_after_padding_and_clamping():
    result = _inference([[3, 5]], [_action()])
    config = CandidateGenerationConfig(min_candidate_duration=5.0, padding_seconds=2.0)
    clips = generate_candidates(result, 10, config=config)
    assert len(clips) == 1
    assert (clips[0].start, clips[0].end, clips[0].duration) == (1.0, 7.0, 6.0)


def test_min_clip_filters_final_short_clip():
    result = _inference([[0, 2]], [_action()])
    config = CandidateGenerationConfig(
        min_candidate_duration=3.0, padding_seconds=0.0, max_gap_to_merge=0.0
    )
    assert generate_candidates(result, 10, config=config) == []


def test_gap_equal_to_max_gap_is_bridged_inclusively():
    result = _inference([[0, 2], [5, 7]], [_action(), _action()])
    config = CandidateGenerationConfig(
        min_candidate_duration=3.0, max_gap_to_merge=3.0, padding_seconds=0.0
    )
    clips = generate_candidates(result, 10, config=config)
    assert len(clips) == 1
    assert (clips[0].start, clips[0].end, clips[0].duration) == (0.0, 7.0, 7.0)


def test_padding_does_not_merge_different_interest_tiers():
    result = _inference([[0, 4], [7, 11]], [_flow(), _action()])
    config = CandidateGenerationConfig(
        min_candidate_duration=3.0, max_gap_to_merge=0.0, padding_seconds=2.0
    )
    clips = generate_candidates(result, 15, config=config)
    assert len(clips) == 2
    assert [(clip.start, clip.end, clip.tier) for clip in clips] == [
        (0.0, 5.5, "moderate_interest"),
        (5.5, 13.0, "high_action"),
    ]
    assert clips[0].max_score == pytest.approx(1.35)
    assert clips[1].max_score == pytest.approx(1.95)


def test_moderate_region_before_high_action_remains_separate():
    intervals = [[0, 4], [2, 6], [4, 8], [6, 10], [8, 12], [10, 14],
                 [12, 16], [14, 18], [16, 20], [18, 22], [20, 24],
                 [22, 26], [24, 28], [28, 32], [30, 34], [32, 36]]
    probabilities = [_flow()] * 13 + [
        [0.10, 0.40, 0.50],
        [0.015, 0.329, 0.656],
        [0.007, 0.224, 0.770],
    ]
    # Window #13 is ineligible; it must not let padding or merging promote
    # the preceding moderate-interest run to high_action.
    config = CandidateGenerationConfig(
        min_candidate_duration=3.0, max_gap_to_merge=3.0, padding_seconds=2.0
    )
    clips = generate_candidates(_inference(intervals, probabilities), 40, config=config)
    assert any(clip.tier == "moderate_interest" for clip in clips)
    assert any(clip.tier == "high_action" for clip in clips)
    assert all(
        not (left.tier != right.tier and left.start < right.end and right.start < left.end)
        for left, right in zip(clips, clips[1:])
    )


def test_asymmetric_p2_envelope_can_preserve_a_short_action_dropout():
    result = _inference(
        [[0, 2], [2, 4]],
        [[0.10, 0.10, 0.80], [0.10, 0.45, 0.45]],
    )
    config = CandidateGenerationConfig(
        action_threshold=0.60, flow_threshold=0.40,
        p2_envelope_decay=0.80, padding_seconds=0.0,
        max_gap_to_merge=0.0, min_candidate_duration=3.0,
    )
    clips = generate_candidates(result, 10, config=config)
    assert len(clips) == 1
    assert clips[0].tier == "high_action"
    # Scoring/evidence retains the original, unsmoothed probabilities.
    assert clips[0].window_evidence[1].p2 == pytest.approx(0.45)


def test_candidates_reject_unsorted_window_timestamps():
    result = _inference([[20, 24], [2, 6]], [_action(), _action()])
    with pytest.raises(ValueError, match="strictly increasing"):
        generate_candidates(result, 30)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"junk_threshold": -0.1},
        {"junk_threshold": 1.1},
        {"action_threshold": float("nan")},
        {"flow_threshold": 1.1},
        {"p2_envelope_decay": -0.1},
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
