"""Tests for LambdaRank LightGBM training, checkpoints and inference."""
import numpy as np
import pytest

lgb = pytest.importorskip("lightgbm")

from autocut.data_models import FeatureWindows
from autocut.dataset.windows import LIGHTGBM_TABULAR_FEATURE_NAMES
from autocut.dataset.normalization import normalize_per_video_features
from autocut.ml.checkpoint import load_lightgbm_checkpoint
from autocut.ml.inference import infer_lightgbm_ranking_windows
from autocut.ml.model import LightGBMConfig
from autocut.ml.training import train_lightgbm


def _data(seed=9, rows=120):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(rows, len(LIGHTGBM_TABULAR_FEATURE_NAMES))).astype(np.float32)
    y = np.arange(rows, dtype=np.int32) % 4
    X[:, 0] += y * 1.5
    return X, y


def _config(**kwargs):
    params = dict(
        num_leaves=7, learning_rate=0.1, max_depth=4,
        min_data_in_leaf=2, num_iterations=15,
        feature_fraction=1.0, bagging_fraction=1.0,
        bagging_frequency=1, lambda_l1=0.1, lambda_l2=0.1,
        min_split_gain=0.0, early_stopping_rounds=3, random_seed=11,
        ndcg_eval_at=(5, 10),
    )
    params.update(kwargs)
    return LightGBMConfig(**params)


def test_lightgbm_ranker_config_defaults_and_validation():
    config = LightGBMConfig()
    assert config.objective == "lambdarank"
    assert config.num_leaves == 31
    assert config.learning_rate == pytest.approx(0.03)
    assert config.to_params()["metric"] == "ndcg"
    assert config.to_params()["ndcg_eval_at"] == [10]
    assert "task" not in config.to_params()
    with pytest.raises(ValueError, match="objective"):
        LightGBMConfig(objective="multiclass")
    with pytest.raises(ValueError, match="feature_fraction"):
        LightGBMConfig(feature_fraction=0)
    with pytest.raises(ValueError, match="ndcg_eval_at"):
        LightGBMConfig(ndcg_eval_at=(0,))


def test_train_save_load_and_infer_lightgbm_ranker(tmp_path):
    train_X, train_y = _data(9, 120)
    val_X, val_y = _data(10, 48)
    path = tmp_path / "models" / "lightgbm-ranker.json"
    model, summary = train_lightgbm(
        train_X, train_y, train_group_sizes=[30, 30, 30, 30],
        val_X=val_X, val_y=val_y, val_group_sizes=[12, 12, 12, 12],
        config=_config(), checkpoint_path=path,
        data_config={"window": 4.0, "stride": 2.0, "sampling_fps": 2.0},
        metadata={"fold": "heldout-group"},
    )
    assert model.feature_dim == 48
    assert summary["training_rows"] == 120
    assert summary["validation_rows"] == 48
    assert summary["ranking_metric"] == "ndcg"
    assert summary["training_group_sizes"] == [30, 30, 30, 30]
    assert path.is_file()

    payload = load_lightgbm_checkpoint(path)
    assert payload["model_class"] == "LightGBMRanker"
    assert payload["class_mapping"] is None
    windows = FeatureWindows(
        X=val_X[:7],
        timestamps=np.column_stack((np.arange(7) * 2.0, np.arange(7) * 2.0 + 4.0)),
    )
    result = infer_lightgbm_ranking_windows(path, windows)
    assert result.scores.shape == (7,)
    assert np.isfinite(result.scores).all()
    np.testing.assert_array_equal(result.timestamps, windows.timestamps)


def test_lightgbm_training_rejects_wrong_feature_count_and_labels_and_missing_groups():
    X, y = _data(rows=40)
    with pytest.raises(ValueError, match="48"):
        train_lightgbm(X[:, :47], y, train_group_sizes=[40], config=_config())
    with pytest.raises(ValueError, match="relevance labels"):
        train_lightgbm(X, np.full(len(y), 5), train_group_sizes=[40], config=_config())
    with pytest.raises(ValueError, match="group sizes are required"):
        train_lightgbm(X, y, config=_config())



def test_per_video_zscore_normalization_handles_constant_columns():
    X = np.asarray([
        [10.0, 4.0, 1.0],
        [12.0, 4.0, 3.0],
        [14.0, 4.0, 5.0],
    ], dtype=np.float32)
    normalized = normalize_per_video_features(X)
    np.testing.assert_allclose(normalized.mean(axis=0), [0.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(normalized.std(axis=0), [1.0, 0.0, 1.0], atol=1e-6)
    assert np.isfinite(normalized).all()
    np.testing.assert_array_equal(normalized[:, 1], np.zeros(3, dtype=np.float32))


def test_per_video_zscore_rejects_invalid_feature_matrices():
    with pytest.raises(ValueError, match="non-empty shape"):
        normalize_per_video_features(np.empty((0, 48), dtype=np.float32))
    with pytest.raises(ValueError, match="finite"):
        normalize_per_video_features(np.asarray([[np.nan]], dtype=np.float32))


def test_ranking_inference_normalizes_the_complete_video_when_checkpoint_requests_it():
    class CaptureBooster:
        def __init__(self):
            self.seen = None

        def predict(self, X):
            self.seen = np.asarray(X).copy()
            return np.zeros(len(X), dtype=np.float32)

    booster = CaptureBooster()
    X = np.zeros((3, len(LIGHTGBM_TABULAR_FEATURE_NAMES)), dtype=np.float32)
    X[:, 0] = [10.0, 12.0, 14.0]
    timestamps = np.column_stack((np.arange(3) * 2.0, np.arange(3) * 2.0 + 4.0))
    payload = {
        "checkpoint_version": "lightgbm-1.0",
        "model_class": "LightGBMRanker",
        "feature_names": list(LIGHTGBM_TABULAR_FEATURE_NAMES),
        "_booster": booster,
        "data_config": {
            "feature_normalization": "per_video_zscore_all_windows_ddof0"
        },
    }
    result = infer_lightgbm_ranking_windows(
        payload, FeatureWindows(X=X, timestamps=timestamps)
    )
    np.testing.assert_allclose(booster.seen[:, 0], [-1.2247449, 0.0, 1.2247449], atol=1e-6)
    np.testing.assert_array_equal(result.scores, np.zeros(3, dtype=np.float32))
