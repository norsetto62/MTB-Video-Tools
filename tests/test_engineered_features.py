"""Tests for the four compact engineered motion signals."""
import numpy as np
import pytest

from autocut.motion.engineered import ENGINEERED_FEATURE_NAMES, engineer_motion_features
from autocut.motion.features import FEATURE_NAMES


def _canonical(grid_magnitudes):
    values = np.zeros((len(grid_magnitudes), len(FEATURE_NAMES)), dtype=np.float32)
    for row in range(3):
        for col in range(3):
            index = FEATURE_NAMES.index(f"grid_{row}_{col}_magnitude")
            values[:, index] = np.asarray(grid_magnitudes)[:, row * 3 + col]
    return values


def test_engineered_features_have_fixed_order_and_expected_grid_ratios():
    grid = np.asarray([[2, 2, 4, 2, 2, 6, 8, 8, 10]], dtype=np.float32)
    X = engineer_motion_features(_canonical(grid))

    assert ENGINEERED_FEATURE_NAMES == (
        "vertical_flow_gradient", "lateral_flow_asymmetry",
        "expansion_index", "roll_index",
    )
    assert X.shape == (1, 4)
    np.testing.assert_allclose(X[0], [-3.0, 4.0 / 3.0, 1.625, 0.0])


def test_engineered_features_support_window_batches_and_do_not_smooth():
    canonical = _canonical(np.asarray([
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 10000],
    ], dtype=np.float32))
    X = engineer_motion_features(canonical[None, ...])

    assert X.shape == (1, 2, 4)
    assert not np.array_equal(X[0, 0], X[0, 1])


@pytest.mark.parametrize("bad", [np.zeros((3, 53)), np.zeros((0, 54)), np.full((2, 54), np.nan)])
def test_engineered_features_reject_invalid_inputs(bad):
    with pytest.raises(ValueError):
        engineer_motion_features(bad)



def test_spike_compression_is_signed_continuous_and_linear_inside_threshold():
    from autocut.motion.engineered import compress_engineered_spikes

    values = np.asarray([-20.0, -5.0, -1.0, 0.0, 1.0, 5.0, 10.0, 20.0])
    result = compress_engineered_spikes(values)
    np.testing.assert_allclose(result, [-8.0, -5.0, -1.0, 0.0, 1.0, 5.0, 6.0, 8.0])
    assert result[0] < result[1] < result[2]
    assert result[-1] - result[-2] == pytest.approx(2.0)


def test_seven_signal_sequence_has_fixed_order_and_smooths_before_windowing():
    from autocut.motion.engineered import MODEL_FEATURE_NAMES, engineer_motion_sequence
    from autocut.dataset.windows import build_windows_from_features

    canonical = _canonical(np.ones((20, 9), dtype=np.float32))
    canonical[:, FEATURE_NAMES.index("flow_u")] = np.arange(20, dtype=np.float32)
    canonical[:, FEATURE_NAMES.index("flow_v")] = np.arange(20, dtype=np.float32) * 2
    canonical[:, FEATURE_NAMES.index("grid_1_1_u")] = canonical[:, FEATURE_NAMES.index("flow_u")]
    canonical[:, FEATURE_NAMES.index("grid_1_1_v")] = canonical[:, FEATURE_NAMES.index("flow_v")]
    timestamps = 0.5 + np.arange(20, dtype=np.float64) * 0.5

    transformed = engineer_motion_sequence(canonical)
    assert transformed.shape == (20, 7)
    assert MODEL_FEATURE_NAMES[-3:] == ("u", "v", "center_residual")
    windows = build_windows_from_features(transformed, timestamps, window=4.0, stride=2.0)
    assert windows.X.shape == (4, 8, 7)
    # Shared timestamps in overlapping windows must contain identical transformed values.
    np.testing.assert_array_equal(windows.X[0, 4:], windows.X[1, :4])


def test_center_residual_is_center_cell_flow_relative_to_global_flow():
    from autocut.motion.engineered import engineer_motion_sequence

    canonical = _canonical(np.ones((12, 9), dtype=np.float32))
    canonical[:, FEATURE_NAMES.index("flow_u")] = 2.0
    canonical[:, FEATURE_NAMES.index("flow_v")] = 3.0
    canonical[:, FEATURE_NAMES.index("grid_1_1_u")] = 5.0
    canonical[:, FEATURE_NAMES.index("grid_1_1_v")] = 7.0

    result = engineer_motion_sequence(canonical)
    np.testing.assert_allclose(result[:, 6], np.hypot(3.0, 4.0), atol=1e-5)
