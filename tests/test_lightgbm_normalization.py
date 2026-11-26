"""Tests for robust per-video feature normalization."""
import numpy as np
import pytest

from autocut.dataset.normalization import (
    feature_normalization_diagnostics,
    normalize_per_video_features,
)


def test_median_iqr_scaling_is_robust_to_extreme_outlier():
    values = np.array([[0.0], [1.0], [2.0], [3.0], [1000.0]], dtype=np.float32)
    robust = normalize_per_video_features(
        values, method="per_video_median_iqr_all_windows"
    )
    assert robust[2, 0] == pytest.approx(0.0)
    assert robust[3, 0] == pytest.approx(0.5)
    assert robust[0, 0] == pytest.approx(-1.0)
    assert np.isfinite(robust).all()


def test_median_iqr_constant_columns_use_unit_scale():
    values = np.array([[2.0, 1.0], [2.0, 3.0], [2.0, 5.0]], dtype=np.float32)
    result = normalize_per_video_features(
        values, method="per_video_median_iqr_all_windows"
    )
    np.testing.assert_array_equal(result[:, 0], np.zeros(3, dtype=np.float32))
    np.testing.assert_allclose(result[:, 1], [-1.0, 0.0, 1.0])


def test_unknown_normalization_is_rejected():
    with pytest.raises(ValueError, match="Unsupported feature normalization"):
        normalize_per_video_features(np.ones((3, 2)), method="rank")



def test_iqr_fallback_uses_std_and_reports_per_video_column_counts():
    values = np.array([
        [0.0, 7.0],
        [0.0, 7.0],
        [0.0, 7.0],
        [0.0, 7.0],
        [10.0, 7.0],
    ], dtype=np.float32)
    diagnostics = feature_normalization_diagnostics(
        values, method="per_video_median_iqr_all_windows"
    )
    assert diagnostics == {
        "std_fallback_columns": 2,
        "near_constant_columns": 1,
    }
    normalized = normalize_per_video_features(
        values, method="per_video_median_iqr_all_windows"
    )
    assert np.isfinite(normalized).all()
    assert normalized[-1, 0] == pytest.approx(2.0)


def test_normalization_method_must_be_explicit():
    with pytest.raises(TypeError, match="method"):
        normalize_per_video_features(np.ones((3, 2)))
