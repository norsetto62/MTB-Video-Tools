"""Tests for the four-class LightGBM model, checkpoint, training and inference."""
import numpy as np
import pytest

lgb = pytest.importorskip("lightgbm")

from autocut.data_models import FeatureWindows
from autocut.dataset.windows import LIGHTGBM_TABULAR_FEATURE_NAMES
from autocut.ml.checkpoint import load_lightgbm_checkpoint
from autocut.ml.inference import infer_lightgbm_windows
from autocut.ml.model import LightGBMConfig
from autocut.ml.training import train_lightgbm


def _data(seed=9, rows=100):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(rows, len(LIGHTGBM_TABULAR_FEATURE_NAMES))).astype(np.float32)
    y = np.arange(rows, dtype=np.int32) % 4
    # Add a learnable class-specific signal while retaining varied features.
    X[:, 0] += y * 1.5
    return X, y


def _config(**kwargs):
    params = dict(
        num_leaves=7, learning_rate=0.1, max_depth=4,
        min_data_in_leaf=2, num_iterations=15,
        feature_fraction=1.0, bagging_fraction=1.0,
        bagging_frequency=1, lambda_l1=0.1, lambda_l2=0.1,
        min_split_gain=0.0, early_stopping_rounds=3, random_seed=11,
    )
    params.update(kwargs)
    return LightGBMConfig(**params)


def test_lightgbm_config_defaults_and_validation():
    config = LightGBMConfig()
    assert config.objective == "multiclass"
    assert config.num_class == 4
    assert config.num_leaves == 31
    assert config.learning_rate == pytest.approx(0.03)
    assert config.to_params()["metric"] == "multi_logloss"
    assert "task" not in config.to_params()
    with pytest.raises(ValueError, match="num_class"):
        LightGBMConfig(num_class=2)
    with pytest.raises(ValueError, match="feature_fraction"):
        LightGBMConfig(feature_fraction=0)


def test_train_save_load_and_infer_lightgbm_multiclass(tmp_path):
    train_X, train_y = _data(9, 120)
    val_X, val_y = _data(10, 48)
    path = tmp_path / "models" / "lightgbm.json"
    model, summary = train_lightgbm(
        train_X, train_y, val_X=val_X, val_y=val_y,
        config=_config(), checkpoint_path=path,
        data_config={"window": 4.0, "stride": 2.0, "flow_fps": 2.0},
        metadata={"fold": "heldout-group"},
    )
    assert model.feature_dim == 48
    assert summary["training_rows"] == 120
    assert summary["validation_rows"] == 48
    assert path.is_file()

    payload = load_lightgbm_checkpoint(path)
    assert payload["class_mapping"] == {
        "0": "MTB_0", "1": "MTB_1", "2": "MTB_2", "3": "MTB_3"
    }
    windows = FeatureWindows(
        X=val_X[:7],
        timestamps=np.column_stack((np.arange(7) * 2.0, np.arange(7) * 2.0 + 4.0)),
    )
    result = infer_lightgbm_windows(path, windows)
    assert result.probabilities.shape == (7, 4)
    assert result.predictions.shape == (7,)
    assert np.all((result.predictions >= 0) & (result.predictions <= 3))
    np.testing.assert_allclose(result.probabilities.sum(axis=1), 1.0, atol=1e-5)
    np.testing.assert_array_equal(result.timestamps, windows.timestamps)


def test_lightgbm_training_rejects_wrong_feature_count_and_labels():
    X, y = _data(rows=40)
    with pytest.raises(ValueError, match="48"):
        train_lightgbm(X[:, :47], y, config=_config())
    with pytest.raises(ValueError, match="classes 0, 1, 2, or 3"):
        train_lightgbm(X, np.full(len(y), 5), config=_config())
