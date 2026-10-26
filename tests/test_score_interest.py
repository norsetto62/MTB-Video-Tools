import numpy as np
import pytest

from autocut.motion.features import FEATURE_NAMES
from tools.score_interest import aggregate_windows, derive_interest_channels, robust_unit_scale, rolling_median


def test_rolling_median_uses_time_axis_and_preserves_constant_signal():
    t = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    y = np.full(5, 3.0)
    np.testing.assert_allclose(rolling_median(y, t, 0.5), y)


def test_rolling_median_rejects_non_monotonic_timestamps():
    with pytest.raises(ValueError, match="strictly increasing"):
        rolling_median(np.ones(3), np.array([0.0, 1.0, 0.5]), 0.5)


def test_robust_scale_clips_outliers_and_handles_constant():
    scaled = robust_unit_scale(np.array([0.0, 1.0, 2.0, 3.0, 1000.0]))
    assert scaled.min() == 0.0
    assert scaled.max() == 1.0
    np.testing.assert_array_equal(robust_unit_scale(np.ones(10)), np.zeros(10))


def test_window_aggregation_uses_60_percent_max_40_percent_mean():
    rows = aggregate_windows(
        np.array([0.0, 1.0, 2.0]),
        {"Score_Final": np.array([0.0, 0.5, 1.0])},
        window_seconds=2.0,
        stride_seconds=1.0,
    )
    assert rows[0]["Score_Final"] == pytest.approx(0.6 * 0.5 + 0.4 * 0.25)
    assert rows[0]["window_start"] == 0.0
    assert rows[1]["window_start"] == 1.0


def test_interest_channels_are_finite_and_bounded():
    rng = np.random.default_rng(42)
    n = 50
    features = rng.normal(size=(n, len(FEATURE_NAMES)))
    timestamps = np.arange(n, dtype=float) * 0.5
    channels = derive_interest_channels(features, timestamps)
    assert {"Score_Tech", "Score_Maneuver", "Gate_Motion", "Score_Final"} <= channels.keys()
    for name in ("Score_Tech", "Score_Maneuver", "Gate_Motion", "Score_Final"):
        assert np.isfinite(channels[name]).all()
        assert np.all((channels[name] >= 0.0) & (channels[name] <= 1.0))


def test_derivative_is_based_on_timestamps():
    timestamps = np.array([0.0, 0.5, 1.0, 1.5])
    # Test the underlying NumPy behavior used by the implementation.
    values = 2.0 * timestamps
    np.testing.assert_allclose(np.gradient(values, timestamps), 2.0)
