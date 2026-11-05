import numpy as np
import pytest

from autocut.data_models import Annotation, FeatureWindows
from autocut.dataset.targets import build_targets, validate_targets


def _windows(timestamps):
    n = len(timestamps)
    X = np.arange(n * 8 * 54, dtype=np.float32).reshape(n, 8, 54)
    return FeatureWindows(X=X, timestamps=np.asarray(timestamps, dtype=np.float64))


def _annotations(*segments):
    return [Annotation(start=s, end=e, score=c, remarks="") for s, e, c in segments]


@pytest.mark.parametrize("score", [0, 1, 2, 3])
def test_pure_window_preserves_original_annotation_score(score):
    windows = _windows([(0, 4)])
    X, y = build_targets(windows, _annotations((0, 4, score)))
    np.testing.assert_array_equal(X, windows.X)
    np.testing.assert_array_equal(y, np.array([score], dtype=np.float32))


def test_window_with_different_overlapping_scores_is_discarded_even_if_one_dominates():
    windows = _windows([(0, 4)])
    X, y = build_targets(windows, _annotations((0, 3, 3), (3, 4, 2)))
    assert X.shape == (0, 8, 54)
    assert y.shape == (0,)


def test_window_fully_covered_by_same_score_segments_is_kept():
    windows = _windows([(0, 4)])
    X, y = build_targets(windows, _annotations((0, 2, 3), (2, 4, 3)))
    np.testing.assert_array_equal(X, windows.X)
    np.testing.assert_array_equal(y, np.array([3], dtype=np.float32))


def test_uncovered_window_is_discarded():
    windows = _windows([(0, 4)])
    X, y = build_targets(windows, _annotations((4, 5, 3)))
    assert X.shape == (0, 8, 54)
    assert y.shape == (0,)


def test_pure_label_validator_accepts_scores_zero_through_three():
    X = np.zeros((4, 8, 54), dtype=np.float32)
    ts = np.array([[i, i + 4] for i in range(4)], dtype=np.float64)
    y = np.array([0, 1, 2, 3], dtype=np.float32)
    assert validate_targets(X, ts, y) == []


@pytest.mark.parametrize("bad", [-1, 0.5, 4, np.nan, np.inf])
def test_validator_rejects_non_pure_scores(bad):
    X = np.zeros((1, 8, 54), dtype=np.float32)
    ts = np.array([[0, 4]], dtype=np.float64)
    errors = validate_targets(X, ts, np.array([bad]))
    assert errors


def test_validator_reports_target_and_timestamp_mismatches():
    X = np.zeros((2, 8, 54), dtype=np.float32)
    ts = np.array([[0, 4]], dtype=np.float64)
    y = np.array([3, 1], dtype=np.float32)
    errors = validate_targets(X, ts, y)
    assert any("timestamp count" in error for error in errors)
