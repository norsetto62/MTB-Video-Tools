"""Tests for NMS-suppressed, pooled event-based Segment Recall@K."""
import numpy as np
import pytest

from autocut.ml.ranking_metrics import segment_recall_at_k, make_segment_recall_at_5


def test_overlapping_class3_windows_form_one_event():
    labels = np.array([3, 3, 3, 0, 3, 3])
    times = np.array([[0, 4], [2, 6], [6, 10], [8, 12], [20, 24], [22, 26]], dtype=float)
    scores = np.array([0.9, 0.1, 0.2, 0.0, 0.8, 0.3])
    result = segment_recall_at_k(labels, scores, [6], times, k=2)
    assert result["events"] == 2
    assert result["hit_events"] == 2
    assert result["segment_recall_at_k"] == pytest.approx(1.0)


def test_nms_suppresses_adjacent_overlapping_predictions_before_top_k():
    labels = np.array([3, 3, 0, 3, 0, 0])
    times = np.array([[0, 4], [2, 6], [4, 8], [10, 14], [12, 16], [20, 24]], dtype=float)
    scores = np.array([0.99, 0.98, 0.97, 0.96, 0.95, 0.94])
    result = segment_recall_at_k(labels, scores, [6], times, k=2)
    assert result["hit_events"] == 2
    assert result["segment_recall_at_k"] == pytest.approx(1.0)
    assert result["predictions_suppressed"] >= 1
    assert result["per_video"]["0"]["predictions_after_suppression"] == 2


def test_pooled_recall_weights_videos_by_event_count():
    labels = np.array([3, 0, 0, 3, 0, 0, 3, 0, 3, 0, 3, 0])
    times = np.array([
        [0, 4], [4, 8], [8, 12], [20, 24], [24, 28], [28, 32],
        [40, 44], [44, 48], [60, 64], [64, 68], [80, 84], [84, 88],
    ], dtype=float)
    scores = np.array([0.99, 0.1, 0.1, 0.98, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    result = segment_recall_at_k(labels, scores, [6, 6], times, k=2)
    assert result["per_video"]["0"]["recall"] == pytest.approx(1.0)
    assert result["per_video"]["1"]["recall"] == pytest.approx(0.0)
    assert result["mean_per_video_event_recall"] == pytest.approx(0.5)
    assert result["pooled_event_recall"] == pytest.approx(2 / 5)
    assert result["segment_recall_at_5"] == pytest.approx(2 / 5)


def test_class2_does_not_join_or_create_class3_event():
    labels = np.array([3, 2, 3, 0])
    times = np.array([[0, 4], [2, 6], [4, 8], [6, 10]], dtype=float)
    result = segment_recall_at_k(labels, np.array([0.9, 0.1, 0.8, 0.2]), [4], times, k=1)
    assert result["events"] == 1
    assert result["hit_events"] == 1


def test_videos_without_class3_are_excluded_from_pooled_denominator():
    labels = np.array([0, 1, 2, 0, 2, 1])
    times = np.array([[0, 4], [2, 6], [4, 8], [0, 4], [2, 6], [4, 8]], dtype=float)
    result = segment_recall_at_k(labels, np.arange(6), [3, 3], times)
    assert result["segment_recall_at_5"] is None
    assert result["videos_with_events"] == 0
    assert result["evaluated_videos"] == 0


def test_metric_rejects_invalid_suppression_threshold():
    with pytest.raises(ValueError, match="suppression_iou_threshold"):
        segment_recall_at_k(
            np.array([3]), np.array([1.0]), [1], np.array([[0.0, 1.0]]),
            suppression_iou_threshold=1.1,
        )


def test_metric_rejects_misaligned_timestamps_and_group_sizes():
    with pytest.raises(ValueError, match="timestamps"):
        segment_recall_at_k(np.array([3]), np.array([1.0]), [1], np.zeros((2, 2)))
    with pytest.raises(ValueError, match="sum"):
        segment_recall_at_k(np.array([3]), np.array([1.0]), [2], np.array([[0.0, 1.0]]))


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
