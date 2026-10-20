import numpy as np
import pytest

from autocut.data_models import Annotation, FeatureWindows
from autocut.dataset.targets import build_targets, validate_targets


def _make_windows(timestamps: list[tuple[float, float]]) -> FeatureWindows:
    n_windows = len(timestamps)
    X = np.arange(n_windows * 8 * 54, dtype=np.float32).reshape(n_windows, 8, 54)
    return FeatureWindows(X=X, timestamps=np.asarray(timestamps, dtype=np.float64))


def _annotations(*segments: tuple[float, float, int]) -> list[Annotation]:
    return [
        Annotation(start=start, end=end, score=score, remarks="")
        for start, end, score in segments
    ]


def _build_one(annotations: list[Annotation], *, positive: int = 4, negative: int = 4):
    windows = _make_windows([(0.0, 4.0)])
    X, y = build_targets(
        windows,
        annotations,
        positive_threshold_samples=positive,
        negative_threshold_samples=negative,
    )
    return windows, X, y


def test_positive_window_counts_mtb3_and_ignores_mtb2():
    windows, X, y = _build_one(_annotations((0, 2, 3), (2, 4, 2)))

    np.testing.assert_array_equal(X, windows.X)
    np.testing.assert_array_equal(y, np.array([1.0], dtype=np.float32))


def test_negative_window_counts_mtb0_and_mtb1_together():
    windows, X, y = _build_one(_annotations((0, 1, 0), (1, 2, 1), (2, 4, 2)))

    np.testing.assert_array_equal(X, windows.X)
    np.testing.assert_array_equal(y, np.array([0.0], dtype=np.float32))


def test_window_below_both_thresholds_is_discarded():
    windows, X, y = _build_one(_annotations((0, 1, 3), (1, 2, 0), (2, 4, 2)))

    assert X.shape == (0, 8, 54)
    assert y.shape == (0,)
    assert X.dtype == np.float32
    assert y.dtype == np.float32


def test_window_meeting_both_thresholds_is_ambiguous_and_discarded():
    windows, X, y = _build_one(_annotations((0, 2, 3), (2, 4, 1)))

    assert X.shape == (0, 8, 54)
    assert y.shape == (0,)


def test_threshold_boundary_is_inclusive():
    windows, X, y = _build_one(_annotations((0, 2, 3), (2, 4, 2)), positive=4)

    np.testing.assert_array_equal(X, windows.X)
    np.testing.assert_array_equal(y, np.array([1.0], dtype=np.float32))


def test_mtb2_only_window_is_discarded():
    _, X, y = _build_one(_annotations((0, 4, 2)))

    assert X.shape == (0, 8, 54)
    assert y.shape == (0,)


@pytest.mark.parametrize(
    ("positive", "negative"),
    [(0, 2), (-1, 2), (float("inf"), 2), (2, 0), (2, float("nan"))],
)
def test_build_targets_rejects_invalid_thresholds(positive: int, negative: int):
    windows = _make_windows([(0.0, 4.0)])
    with pytest.raises(ValueError):
        build_targets(
            windows,
            [],
            positive_threshold_samples=positive,
            negative_threshold_samples=negative,
        )


def _validation_inputs(y: list[float]):
    n = len(y)
    X = np.zeros((n, 8, 54), dtype=np.float32)
    timestamps = np.asarray([(2.0 * i, 2.0 * i + 4.0) for i in range(n)], dtype=np.float64)
    return X, timestamps, np.asarray(y, dtype=np.float32)


def test_validate_targets_accepts_binary_labels():
    X, timestamps, y = _validation_inputs([0.0, 1.0, 0.0, 1.0])
    assert validate_targets(X, timestamps, y) == []


@pytest.mark.parametrize("bad_target", [-0.01, 0.5, 2.0, 3.0])
def test_validate_targets_rejects_non_binary_values(bad_target: float):
    X, timestamps, y = _validation_inputs([0.0, bad_target])
    errors = validate_targets(X, timestamps, y)
    assert any("Binary targets must be 0 or 1" in error for error in errors)


@pytest.mark.parametrize("bad_target", [np.nan, np.inf, -np.inf])
def test_validate_targets_rejects_non_finite_values(bad_target: float):
    X, timestamps, y = _validation_inputs([1.0, bad_target])
    errors = validate_targets(X, timestamps, y)
    assert any("Non-finite values" in error for error in errors)


def test_validate_targets_rejects_non_one_dimensional_y():
    X = np.zeros((2, 8, 54), dtype=np.float32)
    timestamps = np.asarray([[0.0, 4.0], [2.0, 6.0]], dtype=np.float64)
    y = np.asarray([[0.0], [1.0]], dtype=np.float32)
    assert validate_targets(X, timestamps, y) == [
        "Target vector y must be 1D, got shape (2, 1)."
    ]


def test_validate_targets_reports_target_window_count_mismatch():
    X, timestamps, y = _validation_inputs([1.0])
    y = np.asarray([1.0, 0.0], dtype=np.float32)
    assert any("target count (2)" in error for error in validate_targets(X, timestamps, y))


def test_validate_targets_reports_timestamp_window_count_mismatch():
    X, timestamps, y = _validation_inputs([1.0, 0.0])
    assert any(
        "timestamp count (1)" in error
        for error in validate_targets(X, timestamps[:1], y)
    )


def test_validate_targets_accepts_empty_aligned_arrays():
    X = np.empty((0, 8, 54), dtype=np.float32)
    timestamps = np.empty((0, 2), dtype=np.float64)
    y = np.empty((0,), dtype=np.float32)
    assert validate_targets(X, timestamps, y) == []
