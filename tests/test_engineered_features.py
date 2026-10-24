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
    np.testing.assert_allclose(X[0], [-7.0 / 2.0, 2.0 / 2.0, 5.0 / 2.0, (2 + 10 - 4 - 8) / 2.0])


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
