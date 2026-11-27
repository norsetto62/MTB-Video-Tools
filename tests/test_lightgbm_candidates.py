import numpy as np
import pytest

from autocut.selection.candidates import build_candidate_clips, select_ranked_segments


def test_lightgbm_segments_use_hierarchical_ranking_and_allow_negative_scores():
    # The top four windows create three segments: first has strongest peak,
    # second has more evidence windows, and third has the strongest mean only.
    timestamps = np.array([
        [0.0,  2.0],
        [2.0,  4.0],
        [10.0, 12.0],
        [12.0, 14.0],
        [20.0, 22.0],
    ])
    scores = np.array([-0.1, -0.2, -0.3, -0.4, -0.15])
    segments = select_ranked_segments(timestamps, scores, top_k=4, max_gap_seconds=0.0)

    assert len(segments) == 3
    assert segments[0].max_score == pytest.approx(-0.1)
    assert segments[0].window_count == 2
    assert segments[0].mean_score == pytest.approx(-0.15)
    assert segments[1].max_score == pytest.approx(-0.15)
    assert segments[1].window_count == 1
    assert segments[2].max_score == pytest.approx(-0.3)


def test_lightgbm_segments_merge_when_gap_is_two_seconds():
    timestamps = np.array([[0.0, 4.0], [6.0, 10.0], [12.01, 16.0]])
    segments = select_ranked_segments(
        timestamps, np.array([-0.1, -0.2, -0.3]), top_k=3, max_gap_seconds=2.0
    )
    assert len(segments) == 2
    merged = next(segment for segment in segments if segment.window_count == 2)
    assert merged.start == pytest.approx(0.0)
    assert merged.end == pytest.approx(10.0)
    assert merged.max_score == pytest.approx(-0.1)
    assert merged.mean_score == pytest.approx(-0.15)


@pytest.mark.parametrize("top_k,gap", [(0, 2.0), (1, -1.0)])
def test_lightgbm_segment_config_validation(top_k, gap):
    with pytest.raises(ValueError):
        select_ranked_segments(np.array([[0.0, 1.0]]), np.array([0.1]), top_k=top_k, max_gap_seconds=gap)


def test_shared_candidate_builder_applies_min_duration_padding_and_video_bounds():
    timestamps = np.array([[0.0, 4.0], [20.0, 22.0], [96.0, 100.0]])
    scores = np.array([0.9, 0.8, 0.7])
    candidates = build_candidate_clips(
        timestamps, scores, video_duration=100.0, top_k=10,
        max_gap_seconds=2.0, min_clip_seconds=3.0, padding_seconds=2.0,
    )
    # The 2-second segment is filtered before padding. The first and last
    # candidates are clipped to the actual source boundaries.
    assert len(candidates) == 2
    assert candidates[0][1].start == pytest.approx(0.0)
    assert candidates[0][1].end == pytest.approx(6.0)
    assert candidates[1][1].start == pytest.approx(94.0)
    assert candidates[1][1].end == pytest.approx(100.0)


def test_shared_candidate_builder_retains_ranked_segment_and_window_evidence():
    timestamps = np.array([[10.0, 14.0], [16.0, 20.0]])
    candidates = build_candidate_clips(
        timestamps, np.array([-0.1, -0.2]), video_duration=30.0,
        top_k=10, max_gap_seconds=2.0, min_clip_seconds=3.0,
        padding_seconds=1.0,
    )
    assert len(candidates) == 1
    segment, candidate = candidates[0]
    assert segment.window_count == 2
    assert candidate.start == pytest.approx(9.0)
    assert candidate.end == pytest.approx(21.0)
    assert len(candidate.window_evidence) == 2
