"""Tests for robust per-video feature normalization."""
import numpy as np
import pytest

from autocut.dataset.normalization import normalize_per_video_features


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
