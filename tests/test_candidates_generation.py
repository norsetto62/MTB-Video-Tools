"""Regression tests for binary bad/good candidate generation."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from autocut.ml.inference import InferenceResult
from autocut.selection.candidates_generation import CandidateGenerationConfig, generate_candidates


def _inference(intervals, probabilities):
    return InferenceResult(
        timestamps=np.asarray(intervals, dtype=np.float64).reshape(-1, 2),
        probabilities=np.asarray(probabilities, dtype=np.float32).reshape(-1, 2),
    )


def _config(**kwargs):
    values = dict(
        action_threshold=0.70,
        flow_threshold=0.40,
        padding_seconds=0.0,
        min_candidate_duration=1.0,
        max_gap_to_merge=0.0,
    )
    values.update(kwargs)
    return CandidateGenerationConfig(**values)


def test_binary_good_probability_generates_high_confidence_candidate():
    result = _inference([[0, 4]], [[0.1, 0.9]])
    clips = generate_candidates(result, 10, config=_config())
    assert len(clips) == 1
    assert clips[0].tier == "high_action"
    assert clips[0].max_score == pytest.approx(0.9)
    assert clips[0].window_evidence[0].p0 == pytest.approx(0.1)
    assert clips[0].window_evidence[0].p1 == pytest.approx(0.9)
    assert clips[0].window_evidence[0].p2 == 0.0


def test_binary_good_probability_generates_moderate_confidence_candidate():
    result = _inference([[0, 4]], [[0.45, 0.55]])
    clips = generate_candidates(result, 10, config=_config())
    assert len(clips) == 1
    assert clips[0].tier == "moderate_interest"


def test_bad_dominant_window_is_rejected():
    result = _inference([[0, 4]], [[0.75, 0.25]])
    assert generate_candidates(result, 10, config=_config()) == []


def test_overlapping_good_windows_merge_and_clip_to_video_bounds():
    result = _inference([[0, 4], [2, 6], [4, 8]], [[0.1, 0.9]] * 3)
    clips = generate_candidates(result, 8, config=_config(padding_seconds=2.0))
    assert len(clips) == 1
    assert (clips[0].start, clips[0].end, clips[0].duration) == (0.0, 8.0, 8.0)


def test_confidence_tiers_remain_separate_during_padding():
    result = _inference([[0, 4], [7, 11]], [[0.1, 0.9], [0.4, 0.6]])
    clips = generate_candidates(result, 15, config=_config(padding_seconds=2.0))
    assert len(clips) == 2
    assert {clip.tier for clip in clips} == {"high_action", "moderate_interest"}
    assert clips[0].end <= clips[1].start


def test_minimum_duration_applies_after_padding_and_merging():
    result = _inference([[0, 2]], [[0.1, 0.9]])
    config = _config(min_candidate_duration=3.0, padding_seconds=0.0)
    assert generate_candidates(result, 10, config=config) == []


def test_optional_good_probability_envelope_preserves_short_dropout():
    result = _inference([[0, 2], [2, 4]], [[0.1, 0.8], [0.1, 0.45]])
    config = _config(
        action_threshold=0.6,
        flow_threshold=0.4,
        p2_envelope_decay=0.8,
        min_candidate_duration=3.0,
        max_gap_to_merge=0.0,
    )
    clips = generate_candidates(result, 10, config=config)
    assert len(clips) == 1
    assert clips[0].tier == "high_action"
    assert clips[0].window_evidence[1].p1 == pytest.approx(0.45)


def test_empty_inference_returns_no_candidates():
    assert generate_candidates(_inference([], []), 10) == []


def test_binary_probability_contract_rejects_three_columns():
    result = InferenceResult(
        timestamps=np.asarray([[0.0, 4.0]]),
        probabilities=np.asarray([[0.1, 0.2, 0.7]]),
    )
    with pytest.raises(ValueError, match=r"shape \(N, 2\)"):
        generate_candidates(result, 10)


def test_invalid_probability_rows_are_rejected():
    result = _inference([[0, 4]], [[0.2, 0.2]])
    with pytest.raises(ValueError, match="sum to 1"):
        generate_candidates(result, 10)
