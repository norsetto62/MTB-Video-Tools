"""Tests for binary PU candidate generation."""

import numpy as np
import pytest

from autocut.ml.inference import InferenceResult
from autocut.selection.candidates_generation import CandidateGenerationConfig, generate_candidates


def _inference(intervals, probabilities):
    timestamps = np.asarray(intervals, dtype=np.float64).reshape(-1, 2)
    scores = np.asarray(probabilities, dtype=np.float32).reshape(-1)
    return InferenceResult(timestamps=timestamps, probabilities=scores)


def test_overlapping_accepted_windows_merge_and_padding_clamps_to_video():
    result = _inference([[0, 4], [2, 6], [4, 8]], [0.8, 0.9, 0.7])
    clips = generate_candidates(result, clip_duration=8)
    assert len(clips) == 1
    clip = clips[0]
    assert (clip.start, clip.end, clip.duration) == (0.0, 8.0, 8.0)
    assert clip.mean_probability == pytest.approx(0.8, abs=1e-6)
    assert clip.max_probability == pytest.approx(0.9, abs=1e-6)
    assert [w.probability for w in clip.window_evidence] == pytest.approx([0.8, 0.9, 0.7])


def test_threshold_is_inclusive_and_low_probability_window_is_rejected():
    result = _inference([[0, 4], [5, 9], [10, 14]], [0.5, 0.499, 0.9])
    config = CandidateGenerationConfig(positive_threshold=0.5, padding_seconds=0.0,
                                       max_gap_to_merge=0.0, min_candidate_duration=1.0)
    clips = generate_candidates(result, 20, config=config)
    assert [(c.start, c.end) for c in clips] == [(0.0, 4.0), (10.0, 14.0)]


def test_minimum_duration_is_applied_after_padding_and_clamping():
    result = _inference([[3, 5]], [0.9])
    config = CandidateGenerationConfig(min_candidate_duration=5.0, padding_seconds=2.0)
    clips = generate_candidates(result, 10, config=config)
    assert len(clips) == 1
    assert (clips[0].start, clips[0].end, clips[0].duration) == (1.0, 7.0, 6.0)


def test_minimum_duration_filters_final_short_clip():
    result = _inference([[0, 2]], [0.9])
    config = CandidateGenerationConfig(min_candidate_duration=3.0, padding_seconds=0.0)
    assert generate_candidates(result, 10, config=config) == []


def test_gap_equal_to_max_gap_is_bridged_inclusively():
    result = _inference([[0, 2], [5, 7]], [0.9, 0.8])
    config = CandidateGenerationConfig(min_candidate_duration=3.0, max_gap_to_merge=3.0,
                                      padding_seconds=0.0)
    clips = generate_candidates(result, 10, config=config)
    assert len(clips) == 1
    assert (clips[0].start, clips[0].end, clips[0].duration) == (0.0, 7.0, 7.0)
    assert clips[0].window_evidence[0].probability == pytest.approx(0.9)


def test_padding_can_merge_candidates_that_become_touching():
    result = _inference([[0, 4], [7, 11]], [0.8, 0.9])
    config = CandidateGenerationConfig(min_candidate_duration=3.0, max_gap_to_merge=0.0,
                                       padding_seconds=2.0)
    clips = generate_candidates(result, 15, config=config)
    assert len(clips) == 1
    assert (clips[0].start, clips[0].end) == (0.0, 13.0)


def test_candidates_are_returned_in_chronological_order():
    result = _inference([[0, 2], [10, 12]], [0.9, 0.8])
    config = CandidateGenerationConfig(min_candidate_duration=1.0, padding_seconds=0.0,
                                       max_gap_to_merge=0.0)
    clips = generate_candidates(result, 20, config=config)
    assert [c.start for c in clips] == [0.0, 10.0]


def test_candidates_reject_unsorted_window_timestamps():
    result = _inference([[20, 24], [2, 6]], [0.8, 0.9])
    with pytest.raises(ValueError, match="strictly increasing"):
        generate_candidates(result, 30)


@pytest.mark.parametrize("kwargs", [
    {"positive_threshold": -0.1},
    {"positive_threshold": 1.1},
    {"positive_threshold": float("nan")},
    {"min_candidate_duration": 0},
    {"max_gap_to_merge": -1},
    {"padding_seconds": float("inf")},
])
def test_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        CandidateGenerationConfig(**kwargs)


@pytest.mark.parametrize("duration", [0, -1, float("nan"), float("inf")])
def test_rejects_invalid_video_duration(duration):
    with pytest.raises(ValueError, match="clip_duration"):
        generate_candidates(_inference([[0, 4]], [0.8]), duration)


def test_empty_inference_returns_no_candidates():
    result = _inference([], [])
    assert generate_candidates(result, 10) == []


@pytest.mark.parametrize("probability", [-0.01, 1.01, float("nan"), float("inf")])
def test_invalid_probability_values_are_rejected(probability):
    result = _inference([[0, 4]], [probability])
    with pytest.raises(ValueError, match="finite|between 0 and 1"):
        generate_candidates(result, 10)


def test_probability_count_must_match_timestamp_count():
    result = InferenceResult(
        timestamps=np.asarray([[0.0, 4.0], [4.0, 8.0]]),
        probabilities=np.asarray([0.8]),
    )
    with pytest.raises(ValueError, match="match"):
        generate_candidates(result, 10)


def test_binary_two_column_probability_output_uses_positive_column():
    result = InferenceResult(
        timestamps=np.asarray([[0.0, 4.0], [4.0, 8.0]]),
        probabilities=np.asarray([[0.9, 0.1], [0.1, 0.9]]),
    )
    config = CandidateGenerationConfig(positive_threshold=0.5, padding_seconds=0.0,
                                       max_gap_to_merge=0.0, min_candidate_duration=1.0)
    clips = generate_candidates(result, 10, config=config)
    assert [(c.start, c.end) for c in clips] == [(4.0, 8.0)]
