"""Tests for group-isolated four-class LightGBM training and validation."""
import numpy as np
import pytest

pytest.importorskip("lightgbm")

from autocut.data_models import FeatureWindows
from autocut.dataset.windows import LIGHTGBM_TABULAR_FEATURE_NAMES
from autocut.ml.lightgbm_training import (
    LightGBMVideoData,
    train_lightgbm_group_fold,
    train_lightgbm_lovo,
)
from autocut.ml.model import LightGBMConfig


def _video_data(seed, labels):
    rng = np.random.default_rng(seed)
    y = np.asarray(labels, dtype=np.int64)
    X = rng.normal(size=(len(y), len(LIGHTGBM_TABULAR_FEATURE_NAMES))).astype(np.float32)
    X[:, 0] += y * 2.0
    timestamps = np.column_stack((np.arange(len(y)) * 2.0, np.arange(len(y)) * 2.0 + 4.0))
    return LightGBMVideoData(
        windows=FeatureWindows(X=X, timestamps=timestamps.astype(np.float64)),
        targets=y,
    )


def _config():
    return LightGBMConfig(
        num_leaves=7, learning_rate=0.1, max_depth=4,
        min_data_in_leaf=2, num_iterations=12,
        feature_fraction=1.0, bagging_fraction=1.0,
        bagging_frequency=1, lambda_l1=0.1, lambda_l2=0.1,
        min_split_gain=0.0, early_stopping_rounds=3, random_seed=11,
    )


def _dataset():
    videos = {
        "video_a": _video_data(1, [0, 1] * 8),
        "video_b": _video_data(2, [2, 3] * 8),
        "video_c": _video_data(3, [0, 1, 2, 3] * 4),
    }
    groups = {"video_a": "group_a", "video_b": "group_b", "video_c": "group_c"}
    return videos, groups


def test_group_fold_keeps_validation_group_out_of_training_and_reports_metrics(tmp_path):
    videos, groups = _dataset()
    model, result = train_lightgbm_group_fold(
        videos, groups, held_out_group="group_c",
        config=_config(), checkpoint_path=tmp_path / "fold.json",
        data_config={"window": 4.0, "stride": 2.0, "flow_fps": 2.0},
    )
    assert result.held_out_group == "group_c"
    assert result.train_video_ids == ("video_a", "video_b")
    assert result.validation_video_ids == ("video_c",)
    assert set(result.train_video_ids).isdisjoint(result.validation_video_ids)
    metrics = result.summary["validation_metrics"]
    assert metrics["validation_rows"] == 16
    assert len(metrics["confusion_matrix"]) == 4
    assert set(metrics["per_class_recall"]) == {"0", "1", "2", "3"}
    assert model.feature_dim == 48
    assert (tmp_path / "fold.json").is_file()


def test_lovo_runs_one_fold_per_group():
    videos, groups = _dataset()
    results = train_lightgbm_lovo(videos, groups, config=_config())
    assert [result.held_out_group for result in results] == ["group_a", "group_b", "group_c"]
    for result in results:
        assert set(result.train_video_ids).isdisjoint(result.validation_video_ids)
        assert result.validation_video_ids


def test_group_fold_rejects_training_fold_missing_a_class():
    videos = {
        "only_0": _video_data(1, [0, 0, 0, 0]),
        "only_1": _video_data(2, [1, 1, 1, 1]),
        "only_2": _video_data(3, [2, 2, 2, 2]),
        "only_3": _video_data(4, [3, 3, 3, 3]),
    }
    groups = {video_id: f"group_{index}" for index, video_id in enumerate(videos)}
    with pytest.raises(ValueError, match="missing MTB class"):
        train_lightgbm_group_fold(
            videos, groups, held_out_group="group_0", config=_config()
        )


def test_video_data_rejects_misaligned_labels():
    X = np.zeros((3, 48), dtype=np.float32)
    ts = np.zeros((3, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="aligned"):
        LightGBMVideoData(FeatureWindows(X, ts), np.array([0, 1]))



def test_group_fold_allows_a_single_class_in_the_held_out_group():
    videos = {
        "training_video": _video_data(10, [0, 1, 2, 3] * 4),
        "validation_video": _video_data(11, [2] * 8),
    }
    groups = {"training_video": "training", "validation_video": "validation"}
    _, result = train_lightgbm_group_fold(
        videos, groups, held_out_group="validation", config=_config()
    )
    metrics = result.summary["validation_metrics"]
    assert metrics["validation_rows"] == 8
    assert metrics["per_class_support"]["2"] == 8
    assert metrics["per_class_support"]["0"] == 0
