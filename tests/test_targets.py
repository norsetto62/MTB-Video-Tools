import numpy as np
import pytest

from autocut.data_models import Annotation, FeatureWindows
from autocut.dataset.targets import build_targets, validate_targets


def _make_windows(
    timestamps: list[tuple[float, float]],
) -> FeatureWindows:
    """Create identifiable test windows."""

    n_windows = len(timestamps)

    X = np.arange(
        n_windows * 8 * 54,
        dtype=np.float32,
    ).reshape(n_windows, 8, 54)

    return FeatureWindows(
        X=X,
        timestamps=np.asarray(timestamps, dtype=np.float64),
    )


def test_no_annotations_returns_empty_training_data():
    windows = _make_windows(
        [
            (0.0, 4.0),
            (2.0, 6.0),
        ]
    )

    X_training, y = build_targets(windows, [])

    assert X_training.shape == (0, 8, 54)
    assert y.shape == (0,)
    assert X_training.dtype == np.float32
    assert y.dtype == np.float32


def test_uncovered_windows_are_discarded():
    windows = _make_windows(
        [
            (0.0, 4.0),
            (2.0, 6.0),
            (4.0, 8.0),
        ]
    )

    annotations = [
        Annotation(
            start=4.0,
            end=8.0,
            score=3,
            remarks="interesting",
        )
    ]

    X_training, y = build_targets(windows, annotations)

    assert X_training.shape == (2, 8, 54)
    assert y.shape == (2,)

    np.testing.assert_array_equal(
        X_training[0],
        windows.X[1],
    )
    np.testing.assert_array_equal(
        X_training[1],
        windows.X[2],
    )

    np.testing.assert_array_equal(
        y,
        np.array([3.0, 3.0], dtype=np.float32),
    )


def test_first_uncovered_window_is_discarded_and_alignment_is_preserved():
    windows = _make_windows(
        [
            (0.0, 4.0),
            (2.0, 6.0),
            (4.0, 8.0),
        ]
    )

    annotations = [
        Annotation(
            start=1.0,
            end=3.0,
            score=1,
            remarks="ordinary",
        ),
        Annotation(
            start=5.0,
            end=7.0,
            score=3,
            remarks="interesting",
        ),
    ]

    X_training, y = build_targets(windows, annotations)

    assert X_training.shape == (3, 8, 54)
    assert y.shape == (3,)

    # Window 0: [0,4] -> 2 seconds score 1.
    # Window 1: [2,6] -> 1 second score 1 + 1 second score 3.
    # Window 2: [4,8] -> 2 seconds score 3.
    np.testing.assert_array_equal(
        X_training,
        windows.X[[0, 1, 2]],
    )

    np.testing.assert_allclose(
        y,
        np.array([1.0, 2.0, 3.0], dtype=np.float32),
    )


def test_fully_covered_window_gets_annotation_score():
    windows = _make_windows(
        [(2.0, 6.0)]
    )

    annotations = [
        Annotation(
            start=0.0,
            end=10.0,
            score=2,
            remarks="middle",
        )
    ]

    X_training, y = build_targets(windows, annotations)

    assert X_training.shape == (1, 8, 54)
    np.testing.assert_array_equal(X_training[0], windows.X[0])
    np.testing.assert_array_equal(
        y,
        np.array([2.0], dtype=np.float32),
    )


def test_partial_coverage_uses_covered_duration():
    windows = _make_windows(
        [(0.0, 4.0)]
    )

    annotations = [
        Annotation(
            start=2.0,
            end=4.0,
            score=3,
            remarks="interesting",
        )
    ]

    X_training, y = build_targets(windows, annotations)

    assert X_training.shape == (1, 8, 54)

    # Only [2,4] is covered, so the target remains exactly 3.
    np.testing.assert_array_equal(
        y,
        np.array([3.0], dtype=np.float32),
    )


def test_multiple_annotations_use_duration_weighted_mean():
    windows = _make_windows(
        [(0.0, 4.0)]
    )

    annotations = [
        Annotation(
            start=0.0,
            end=1.0,
            score=1,
            remarks="ordinary",
        ),
        Annotation(
            start=1.0,
            end=3.0,
            score=3,
            remarks="interesting",
        ),
    ]

    X_training, y = build_targets(windows, annotations)

    assert X_training.shape == (1, 8, 54)

    # (1*1 + 2*3) / 3 = 7/3
    np.testing.assert_allclose(
        y,
        np.array([7.0 / 3.0], dtype=np.float32),
    )


def test_annotations_touching_window_boundary_do_not_overlap():
    windows = _make_windows(
        [(0.0, 4.0)]
    )

    annotations = [
        Annotation(
            start=4.0,
            end=6.0,
            score=3,
            remarks="after",
        )
    ]

    X_training, y = build_targets(windows, annotations)

    assert X_training.shape == (0, 8, 54)
    assert y.shape == (0,)


def test_annotation_before_window_does_not_overlap():
    windows = _make_windows(
        [(4.0, 8.0)]
    )

    annotations = [
        Annotation(
            start=0.0,
            end=4.0,
            score=3,
            remarks="before",
        )
    ]

    X_training, y = build_targets(windows, annotations)

    assert X_training.shape == (0, 8, 54)
    assert y.shape == (0,)


def test_target_output_is_float32():
    windows = _make_windows(
        [(0.0, 4.0)]
    )

    annotations = [
        Annotation(
            start=0.0,
            end=4.0,
            score=2,
            remarks="test",
        )
    ]

    X_training, y = build_targets(windows, annotations)

    assert X_training.dtype == np.float32
    assert y.dtype == np.float32


def test_all_windows_are_retained_when_all_are_covered():
    windows = _make_windows(
        [
            (0.0, 4.0),
            (2.0, 6.0),
            (4.0, 8.0),
        ]
    )

    annotations = [
        Annotation(
            start=0.0,
            end=8.0,
            score=1,
            remarks="ordinary",
        )
    ]

    X_training, y = build_targets(windows, annotations)

    assert X_training.shape == windows.X.shape
    assert y.shape == (3,)

    np.testing.assert_array_equal(
        X_training,
        windows.X,
    )
    np.testing.assert_array_equal(
        y,
        np.ones(3, dtype=np.float32),
    )
    


def _make_target_validation_inputs(y: list[float]):
    """Create aligned feature windows, timestamps and interest scores."""
    n = len(y)
    X = np.zeros((n, 8, 54), dtype=np.float32)
    timestamps = np.asarray(
        [(2.0 * i, 2.0 * i + 4.0) for i in range(n)],
        dtype=np.float64,
    )
    return X, timestamps, np.asarray(y, dtype=np.float32)


def test_validate_targets_accepts_integer_and_fractional_interest_scores():
    X, timestamps, y = _make_target_validation_inputs([0.0, 1.0, 1.5, 2.25, 3.0])

    assert validate_targets(X, timestamps, y) == []


@pytest.mark.parametrize("bad_score", [-0.01, 3.01, -1.0, 4.0])
def test_validate_targets_rejects_scores_outside_zero_to_three(bad_score: float):
    X, timestamps, y = _make_target_validation_inputs([1.0, bad_score])

    errors = validate_targets(X, timestamps, y)

    assert any("within [0, 3]" in error for error in errors)


@pytest.mark.parametrize("bad_score", [np.nan, np.inf, -np.inf])
def test_validate_targets_rejects_non_finite_scores(bad_score: float):
    X, timestamps, y = _make_target_validation_inputs([1.0, bad_score])

    errors = validate_targets(X, timestamps, y)

    assert any("Non-finite values" in error for error in errors)


def test_validate_targets_rejects_non_one_dimensional_y():
    X = np.zeros((2, 8, 54), dtype=np.float32)
    timestamps = np.asarray([[0.0, 4.0], [2.0, 6.0]], dtype=np.float64)
    y = np.asarray([[1.0], [2.0]], dtype=np.float32)

    errors = validate_targets(X, timestamps, y)

    assert errors == ["Target vector y must be 1D, got shape (2, 1)."]


def test_validate_targets_reports_target_window_count_mismatch():
    X, timestamps, y = _make_target_validation_inputs([1.0])
    y = np.asarray([1.0, 2.0], dtype=np.float32)

    errors = validate_targets(X, timestamps, y)

    assert any("target count (2)" in error for error in errors)


def test_validate_targets_reports_timestamp_window_count_mismatch():
    X, timestamps, y = _make_target_validation_inputs([1.0, 2.0])
    timestamps = timestamps[:1]

    errors = validate_targets(X, timestamps, y)

    assert any("timestamp count (1)" in error for error in errors)


def test_validate_targets_does_not_reject_a_skewed_score_distribution():
    """Distribution reporting belongs to dataset statistics, not validation."""
    X, timestamps, y = _make_target_validation_inputs([0.0, 0.0, 0.0, 3.0])

    assert validate_targets(X, timestamps, y) == []


def test_validate_targets_accepts_empty_aligned_arrays():
    X = np.empty((0, 8, 54), dtype=np.float32)
    timestamps = np.empty((0, 2), dtype=np.float64)
    y = np.empty((0,), dtype=np.float32)

    assert validate_targets(X, timestamps, y) == []
