"""Tests for event-based class-3 Segment Recall@5."""
import numpy as np
import pytest

from autocut.ml.ranking_metrics import segment_recall_at_k, make_segment_recall_at_5


def test_overlapping_and_touching_class3_windows_form_one_event():
    labels = np.array([3, 3, 3, 0, 3, 3])
    times = np.array([
        [0, 4], [2, 6], [6, 10], [8, 12], [20, 24], [22, 26],
    ], dtype=float)
    scores = np.array([0.9, 0.1, 0.2, 0.0, 0.8, 0.3])
    result = segment_recall_at_k(labels, scores, [6], times, k=2)
    # Top-2 contains one window from each of the two events.
    assert result["events"] == 2
    assert result["hit_events"] == 2
    assert result["segment_recall_at_k"] == pytest.approx(1.0)


def test_top_k_is_computed_independently_per_video_and_only_class3_counts():
    labels = np.array([3, 0, 0, 3, 2, 0, 3, 0])
    scores = np.array([0.1, 0.9, 0.8, 0.95, 1.0, 0.9, 0.2, 0.1])
    times = np.array([
        [0, 4], [2, 6], [4, 8], [10, 14],
        [0, 4], [2, 6], [10, 14], [12, 16],
    ], dtype=float)
    result = segment_recall_at_k(labels, scores, [4, 4], times, k=1)
    assert result["videos_with_events"] == 2
    assert result["events"] == 3
    assert result["hit_events"] == 1
    assert result["segment_recall_at_k"] == pytest.approx(0.25)


def test_class2_does_not_join_or_create_class3_event():
    labels = np.array([3, 2, 3, 0])
    times = np.array([[0, 4], [2, 6], [4, 8], [6, 10]], dtype=float)
    result = segment_recall_at_k(labels, np.array([0.9, 0.1, 0.8, 0.2]), [4], times, k=1)
    assert result["events"] == 1
    assert result["hit_events"] == 1


def test_videos_without_class3_are_excluded_and_reported():
    labels = np.array([0, 1, 2, 0, 2, 1])
    times = np.array([[0, 4], [2, 6], [4, 8], [0, 4], [2, 6], [4, 8]], dtype=float)
    result = segment_recall_at_k(labels, np.arange(6), [3, 3], times)
    assert result["segment_recall_at_5"] is None
    assert result["videos_with_events"] == 0
    assert result["evaluated_videos"] == 0


def test_metric_rejects_misaligned_timestamps_and_group_sizes():
    with pytest.raises(ValueError, match="timestamps"):
        segment_recall_at_k(np.array([3]), np.array([1.0]), [1], np.zeros((2, 2)))
    with pytest.raises(ValueError, match="sum"):
        segment_recall_at_k(
            np.array([3]), np.array([1.0]), [2], np.array([[0.0, 1.0]])
        )


def test_lightgbm_feval_returns_higher_is_better():
    class Dataset:
        def get_label(self):
            return np.array([3, 0, 3, 0])

    times = np.array([[0, 4], [2, 6], [10, 14], [12, 16]], dtype=float)
    callback = make_segment_recall_at_5(times, [4])
    name, value, higher_is_better = callback(np.array([0.9, 0.1, 0.8, 0.2]), Dataset())
    assert name == "segment_recall_at_5"
    assert value == pytest.approx(1.0)
    assert higher_is_better is True
