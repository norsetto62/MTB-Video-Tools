"""Tests for production-equivalent candidate-level event metrics."""
import numpy as np
import pytest

from autocut.ml.ranking_metrics import (
    METRIC_ID,
    candidate_event_metrics,
    make_candidate_event_recall_evaluator,
)


def test_candidate_event_recall_uses_padded_clips_and_one_second_overlap():
    timestamps = np.array([[10.0, 14.0], [30.0, 34.0], [50.0, 54.0]])
    scores = np.array([0.9, 0.8, 0.1])
    annotations = [[(13.0, 16.0, 3), (30.0, 34.0, 3)]]
    result = candidate_event_metrics(
        scores, [3], timestamps, [60.0], annotations,
        max_gap_seconds=2.0, min_clip_seconds=3.0, padding_seconds=2.0,
    )
    # First candidate starts at 8s and ends at 16s, covering the first event
    # for 3 seconds. Second candidate also covers the second event.
    assert result["events"] == 2
    assert result["hit_events"] == 2
    assert result["pooled_event_recall"] == pytest.approx(1.0)
    assert result["selected_segments"] == 2
    assert result["useful_segments"] == 2
    assert result["segment_precision"] == pytest.approx(1.0)
    assert result["metric"] == METRIC_ID


def test_event_requires_at_least_one_second_overlap():
    timestamps = np.array([[0.0, 4.0], [20.0, 24.0]])
    result = candidate_event_metrics(
        np.array([0.9, 0.1]), [2], timestamps, [30.0],
        [[(3.1, 5.0, 3)]],
        max_gap_seconds=0.0, min_clip_seconds=3.0, padding_seconds=0.0,
    )
    # Unpadded segment [0,4) overlaps the event by only 0.9 seconds.
    assert result["events"] == 1
    assert result["hit_events"] == 0
    assert result["pooled_event_recall"] == pytest.approx(0.0)


def test_adjacent_class3_annotations_are_one_event_and_only_counted_once():
    timestamps = np.array([[0.0, 4.0], [4.0, 8.0], [30.0, 34.0]])
    result = candidate_event_metrics(
        np.array([0.9, 0.8, 0.1]), [3], timestamps, [40.0],
        [[(1.0, 3.0, 3), (3.0, 5.0, 3)]],
        max_gap_seconds=2.0, min_clip_seconds=3.0, padding_seconds=2.0,
    )
    assert result["events"] == 1
    assert result["hit_events"] == 1


def test_class2_3_duration_fraction_is_reported_separately():
    timestamps = np.array([[0.0, 4.0], [30.0, 34.0]])
    result = candidate_event_metrics(
        np.array([0.9, 0.1]), [2], timestamps, [40.0],
        [[(0.0, 2.0, 2), (2.0, 4.0, 0)]],
        max_gap_seconds=0.0, min_clip_seconds=3.0, padding_seconds=0.0,
    )
    assert result["interesting_segments"] == 1
    assert result["interesting_segment_rate"] == pytest.approx(1.0)
    assert result["events"] == 0
    assert result["pooled_event_recall"] is None


def test_candidate_metric_reports_segment_and_duration_diagnostics():
    timestamps = np.array([[0.0, 4.0], [20.0, 24.0], [40.0, 44.0]])
    result = candidate_event_metrics(
        np.array([0.9, 0.8, 0.1]), [3], timestamps, [50.0],
        [[(0.0, 4.0, 3)]],
        max_gap_seconds=0.0, min_clip_seconds=3.0, padding_seconds=2.0,
    )
    assert result["average_segments_per_video"] == pytest.approx(2.0)
    assert result["total_selected_duration_seconds"] == pytest.approx(16.0)
    assert result["per_video"]["0"]["candidate_count"] == 2


def test_lightgbm_evaluator_returns_candidate_event_recall():
    class Dataset:
        def get_label(self):
            return np.array([3, 0, 3, 0])

    callback = make_candidate_event_recall_evaluator(
        np.array([[0, 4], [2, 6], [20, 24], [22, 26]], dtype=float),
        [4], [40.0], [[(0, 4, 3), (20, 24, 3)]],
        max_gap_seconds=2.0, min_clip_seconds=3.0, padding_seconds=2.0,
    )
    name, value, higher_is_better = callback(
        np.array([0.9, 0.1, 0.8, 0.2]), Dataset()
    )
    assert name == METRIC_ID
    assert value == pytest.approx(1.0)
    assert higher_is_better is True


def test_metric_rejects_misaligned_metadata():
    with pytest.raises(ValueError, match="align"):
        candidate_event_metrics(
            np.array([0.9]), [1], np.array([[0.0, 4.0]]),
            [10.0, 11.0], [[(0.0, 2.0, 3)]],
        )
