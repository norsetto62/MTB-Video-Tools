"""Tests for AutoCut's feature scaler."""

import numpy as np
import pytest

from autocut.ml.scaling import FeatureScaler


def test_fit_3d_uses_joint_batch_and_time_statistics():
    X = np.array(
        [
            [[1.0, 10.0], [3.0, 10.0]],
            [[5.0, 10.0], [7.0, 10.0]],
        ],
        dtype=np.float32,
    )
    scaler = FeatureScaler().fit(X)

    np.testing.assert_allclose(scaler.mean_, [4.0, 10.0])
    np.testing.assert_allclose(scaler.scale_, [np.std([1, 3, 5, 7]), 1.0])


def test_fit_2d_uses_time_axis_statistics():
    X = np.array([[1.0, 10.0], [3.0, 14.0], [5.0, 18.0]], dtype=np.float32)
    scaler = FeatureScaler().fit(X)

    np.testing.assert_allclose(scaler.mean_, [3.0, 14.0])
    np.testing.assert_allclose(scaler.scale_, [np.std([1, 3, 5]), np.std([10, 14, 18])])


def test_transform_standardizes_each_feature_and_returns_float32():
    X = np.array([[[1.0, 10.0], [3.0, 14.0]], [[5.0, 18.0], [7.0, 22.0]]])
    scaler = FeatureScaler()
    transformed = scaler.fit_transform(X)

    assert transformed.dtype == np.float32
    np.testing.assert_allclose(transformed.mean(axis=(0, 1)), [0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(transformed.std(axis=(0, 1)), [1.0, 1.0], atol=1e-6)


def test_constant_and_near_constant_features_use_scale_one():
    X = np.array([[2.0, 1.0], [2.0, 1.0 + 1e-10]], dtype=np.float32)
    scaler = FeatureScaler(eps=1e-8).fit(X)

    np.testing.assert_array_equal(scaler.scale_, [1.0, 1.0])
    assert np.isfinite(scaler.transform(X)).all()


def test_transform_requires_fitting_when_enabled():
    with pytest.raises(RuntimeError, match="must be fitted"):
        FeatureScaler().transform(np.ones((2, 3), dtype=np.float32))


def test_transform_rejects_feature_count_mismatch():
    scaler = FeatureScaler().fit(np.ones((3, 2), dtype=np.float32))

    with pytest.raises(ValueError, match="2 features"):
        scaler.transform(np.ones((4, 3), dtype=np.float32))


@pytest.mark.parametrize("shape", [(0, 3), (3, 0), (0, 2, 3), (2, 0, 3), (2, 3, 0)])
def test_fit_rejects_empty_dimensions(shape):
    with pytest.raises(ValueError, match="empty dimensions"):
        FeatureScaler().fit(np.empty(shape, dtype=np.float32))


@pytest.mark.parametrize("shape", [(3,), (1, 2, 3, 4)])
def test_fit_rejects_unsupported_dimensions(shape):
    with pytest.raises(ValueError, match="2D.*3D"):
        FeatureScaler().fit(np.ones(shape, dtype=np.float32))


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("operation", ["fit", "transform"])
def test_rejects_non_finite_input(bad_value, operation):
    X = np.ones((3, 2), dtype=np.float32)
    X[1, 0] = bad_value
    scaler = FeatureScaler().fit(np.ones((3, 2), dtype=np.float32))

    with pytest.raises(ValueError, match="finite values"):
        if operation == "fit":
            FeatureScaler().fit(X)
        else:
            scaler.transform(X)


@pytest.mark.parametrize("eps", [0, -1, np.nan, np.inf, -np.inf, True, "1e-8"])
def test_constructor_rejects_invalid_eps(eps):
    with pytest.raises(ValueError, match="eps"):
        FeatureScaler(eps=eps)


def test_constructor_rejects_non_boolean_enabled():
    with pytest.raises(ValueError, match="enabled"):
        FeatureScaler(enabled=1)


def test_disabled_scaler_skips_standardization_and_returns_float32():
    X = np.array([[1.25, 8.5], [2.75, 9.5]], dtype=np.float64)
    scaler = FeatureScaler(enabled=False)

    transformed = scaler.fit_transform(X)

    assert transformed.dtype == np.float32
    np.testing.assert_array_equal(transformed, X.astype(np.float32))
    assert scaler.mean_ is None
    assert scaler.scale_ is None
    np.testing.assert_array_equal(scaler.transform(X), X.astype(np.float32))


def test_state_dict_round_trip_preserves_transformation():
    X = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    scaler = FeatureScaler(eps=1e-7).fit(X)
    restored = FeatureScaler.from_state_dict(scaler.state_dict())

    assert restored.enabled is True
    assert restored.eps == pytest.approx(1e-7)
    np.testing.assert_array_equal(restored.mean_, scaler.mean_)
    np.testing.assert_array_equal(restored.scale_, scaler.scale_)
    np.testing.assert_array_equal(restored.transform(X), scaler.transform(X))


def test_unfitted_scaler_state_round_trip():
    scaler = FeatureScaler()
    restored = FeatureScaler.from_state_dict(scaler.state_dict())

    assert restored.mean_ is None
    assert restored.scale_ is None
    with pytest.raises(RuntimeError, match="must be fitted"):
        restored.transform(np.ones((2, 3), dtype=np.float32))


def test_disabled_scaler_state_round_trip():
    scaler = FeatureScaler(enabled=False)
    restored = FeatureScaler.from_state_dict(scaler.state_dict())

    assert restored.enabled is False
    assert restored.mean_ is None
    assert restored.scale_ is None


@pytest.mark.parametrize(
    "state",
    [
        None,
        [],
        {"enabled": True, "eps": 1e-8, "mean": [1.0]},
        {"enabled": True, "eps": 1e-8, "mean": [1.0], "scale": [1.0, 2.0]},
        {"enabled": True, "eps": 1e-8, "mean": [np.nan], "scale": [1.0]},
        {"enabled": True, "eps": 1e-8, "mean": [1.0], "scale": [np.inf]},
        {"enabled": True, "eps": 1e-8, "mean": [1.0], "scale": [0.0]},
        {"enabled": True, "eps": 1e-8, "mean": [1.0], "scale": [-1.0]},
        {"enabled": True, "eps": 1e-8, "mean": [[1.0]], "scale": [[1.0]]},
        {"enabled": True, "eps": 1e-8, "mean": [1.0], "scale": None},
        {"enabled": 1, "eps": 1e-8, "mean": None, "scale": None},
        {"enabled": True, "eps": -1.0, "mean": None, "scale": None},
    ],
)
def test_from_state_dict_rejects_invalid_state(state):
    with pytest.raises(ValueError):
        FeatureScaler.from_state_dict(state)
