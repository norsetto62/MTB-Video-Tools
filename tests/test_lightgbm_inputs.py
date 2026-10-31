"""Tests for the LightGBM tabular feature and four-class target inputs."""
import numpy as np
import pytest

from autocut.data_models import Annotation, FeatureWindows
from autocut.dataset.targets import build_majority_vote_targets
from autocut.dataset.windows import (
    LIGHTGBM_TABULAR_FEATURE_NAMES,
    aggregate_lightgbm_windows,
)
from autocut.motion.engineered import LIGHTGBM_FEATURE_NAMES, engineer_lightgbm_sequence
from autocut.motion.features import FEATURE_NAMES


def _canonical(grid_vectors, *, flow_u=0.0, flow_v=0.0):
    values = np.zeros((len(grid_vectors), len(FEATURE_NAMES)), dtype=np.float32)
    for row in range(3):
        for col in range(3):
            values[:, FEATURE_NAMES.index(f"grid_{row}_{col}_u")] = np.asarray(
                grid_vectors
            )[:, row * 3 + col, 0]
            values[:, FEATURE_NAMES.index(f"grid_{row}_{col}_v")] = np.asarray(
                grid_vectors
            )[:, row * 3 + col, 1]
    values[:, FEATURE_NAMES.index("flow_u")] = flow_u
    values[:, FEATURE_NAMES.index("flow_v")] = flow_v
    return values


def _annotation(start, end, score):
    return Annotation(start=start, end=end, score=score, remarks="")


def test_lightgbm_signal_order_and_shape():
    vectors = np.ones((3, 9, 2), dtype=np.float32)
    result = engineer_lightgbm_sequence(_canonical(vectors))
    assert LIGHTGBM_FEATURE_NAMES == (
        "vertical_flow_gradient", "lateral_flow_asymmetry",
        "expansion_index", "roll_index", "directional_turbulence",
        "global_flow_magnitude", "center_residual", "tunnel_vision",
    )
    assert result.shape == (3, 8)
    assert result.dtype == np.float32
    assert np.isfinite(result).all()


def test_lightgbm_signals_use_global_net_flow_and_relative_center_residual():
    vectors = np.zeros((1, 9, 2), dtype=np.float32)
    vectors[0, :, :] = (4.0, 3.0)
    vectors[0, 4, :] = (7.0, 7.0)
    canonical = _canonical(vectors, flow_u=4.0, flow_v=3.0)
    result = engineer_lightgbm_sequence(canonical)
    assert result[0, 5] == pytest.approx(5.0)  # hypot(global flow_u, flow_v)
    assert result[0, 6] == pytest.approx(5.0)  # center vector residual: (3,4)



def test_tunnel_vision_uses_the_two_experimental_top_corner_residuals():
    vectors = np.zeros((1, 9, 2), dtype=np.float32)
    vectors[0, 0] = (3.0, 0.0)
    vectors[0, 2] = (5.0, 0.0)
    vectors[0, 4] = (1.0, 0.0)
    result = engineer_lightgbm_sequence(_canonical(vectors))
    # global flow is zero, so corner residuals are 3 and 5; center residual is 1.
    assert result[0, 6] == pytest.approx(1.0)
    assert result[0, 7] == pytest.approx(4.0 / (1.0 + 1e-5), rel=1e-6)


def test_lightgbm_window_aggregation_has_48_columns_in_stable_order():
    # Two windows, three timesteps, two signals repeated across signal slots.
    temporal = np.zeros((2, 3, 8), dtype=np.float32)
    temporal[0, :, 0] = [1.0, 2.0, 3.0]
    temporal[0, :, 1] = [-2.0, 0.0, 2.0]
    temporal[1, :, 0] = [4.0, 4.0, 4.0]
    timestamps = np.array([[0.0, 3.0], [1.0, 4.0]], dtype=np.float64)
    result = aggregate_lightgbm_windows(FeatureWindows(temporal, timestamps))
    assert result.X.shape == (2, 48)
    assert len(LIGHTGBM_TABULAR_FEATURE_NAMES) == 48
    assert LIGHTGBM_TABULAR_FEATURE_NAMES[:6] == (
        "vertical_flow_gradient_mean", "vertical_flow_gradient_max",
        "vertical_flow_gradient_std", "vertical_flow_gradient_median",
        "vertical_flow_gradient_q90", "vertical_flow_gradient_min",
    )
    assert result.X.dtype == np.float32
    # Signal 0: mean, max, population std, median, q90, min.
    np.testing.assert_allclose(
        result.X[0, :6],
        [2.0, 3.0, np.std([1.0, 2.0, 3.0]), 2.0, 2.8, 1.0],
    )
    # Signal 1 occupies the next six columns.
    np.testing.assert_allclose(
        result.X[0, 6:12],
        [0.0, 2.0, np.std([-2.0, 0.0, 2.0]), 0.0, 1.6, -2.0],
    )
    np.testing.assert_array_equal(result.timestamps, timestamps)


def test_majority_vote_uses_duration_and_higher_class_breaks_ties():
    windows = FeatureWindows(
        X=np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32),
        timestamps=np.array([[0.0, 4.0], [4.0, 8.0]], dtype=np.float64),
    )
    annotations = [
        _annotation(0.0, 1.0, 1),
        _annotation(1.0, 2.0, 2),
        _annotation(2.0, 4.0, 1),  # class 1 has 3 seconds -> wins
        _annotation(4.0, 6.0, 1),
        _annotation(6.0, 8.0, 2),  # tie -> higher class 2
    ]
    kept, labels = build_majority_vote_targets(windows, annotations)
    np.testing.assert_array_equal(kept.X, windows.X)
    np.testing.assert_array_equal(kept.timestamps, windows.timestamps)
    np.testing.assert_array_equal(labels, [1, 2])


def test_majority_vote_ignores_uncovered_time_but_drops_unannotated_windows():
    windows = FeatureWindows(
        X=np.arange(3 * 8, dtype=np.float32).reshape(3, 8),
        timestamps=np.array([[0.0, 4.0], [4.0, 8.0], [8.0, 12.0]], dtype=np.float64),
    )
    annotations = [_annotation(1.0, 2.0, 0), _annotation(6.0, 7.0, 3)]
    kept, labels = build_majority_vote_targets(windows, annotations)
    np.testing.assert_array_equal(labels, [0, 3])
    np.testing.assert_array_equal(kept.timestamps, windows.timestamps[:2])


def test_majority_vote_returns_correct_empty_shapes():
    windows = FeatureWindows(
        X=np.zeros((2, 48), dtype=np.float32),
        timestamps=np.array([[0.0, 4.0], [4.0, 8.0]], dtype=np.float64),
    )
    kept, labels = build_majority_vote_targets(windows, [])
    assert kept.X.shape == (0, 48)
    assert kept.timestamps.shape == (0, 2)
    assert labels.shape == (0,)
    assert labels.dtype == np.int64


@pytest.mark.parametrize("bad", [np.zeros((4, 7)), np.zeros((0, 8))])
def test_lightgbm_aggregation_rejects_wrong_or_empty_signal_windows(bad):
    timestamps = np.tile(np.array([[0.0, 4.0]]), (len(bad), 1)).astype(np.float64)
    with pytest.raises(ValueError):
        aggregate_lightgbm_windows(FeatureWindows(bad, timestamps))
