from pathlib import Path

import numpy as np
import pytest

from autocut.ml.model import LightGBMConfig
from autocut.pipeline import lightgbm_training as pipeline


def test_single_model_training_uses_all_manifest_data_and_saves_checkpoint(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text("videos: []\n", encoding="utf-8")
    checkpoint = tmp_path / "models" / "single.json"
    prepared = {"video-a": object(), "video-b": object()}
    group_by_video = {"video-a": "ascoli", "video-b": "castelli"}
    data_config = {"sampling_fps": 2.0, "window": 4.0, "stride": 2.0}
    config = LightGBMConfig(num_iterations=10, early_stopping_rounds=2)
    captured = {}

    def fake_prepare(path, settings, *, require_multiple_groups=True):
        captured["prepare"] = (path, settings, require_multiple_groups)
        return prepared, group_by_video, data_config

    train_X = np.zeros((8, 48), dtype=np.float32)
    train_y = np.repeat(np.arange(4, dtype=np.int32), 2)

    def fake_combine(data, video_ids):
        captured["combine"] = (data, video_ids)
        return train_X, train_y

    def fake_train(X, y, **kwargs):
        captured["train"] = (X, y, kwargs)
        return object(), {"training_rows": len(X), "checkpoint_path": str(kwargs["checkpoint_path"])}

    monkeypatch.setattr(pipeline, "prepare_lightgbm_manifest_data", fake_prepare)
    monkeypatch.setattr(pipeline, "_combine", fake_combine)
    monkeypatch.setattr(pipeline, "train_lightgbm", fake_train)

    result = pipeline.run_lightgbm_train_from_manifest(
        manifest, {"experiment": {}}, checkpoint_path=checkpoint, config=config
    )

    assert captured["prepare"] == (manifest, {"experiment": {}}, False)
    assert captured["combine"] == (prepared, ["video-a", "video-b"])
    X, y, kwargs = captured["train"]
    assert X is train_X
    assert y is train_y
    assert kwargs["config"] is config
    assert kwargs["checkpoint_path"] == checkpoint
    assert kwargs["data_config"]["manifest"] == str(manifest.resolve())
    assert kwargs["data_config"]["group_by_video"] == group_by_video
    assert kwargs["metadata"]["training_strategy"] == "single_model_all_annotated_videos"
    assert result["training_rows"] == 8


def test_single_model_training_rejects_missing_annotation_class(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text("videos: []\n", encoding="utf-8")
    monkeypatch.setattr(
        pipeline,
        "prepare_lightgbm_manifest_data",
        lambda *args, **kwargs: (
            {"video-a": object()},
            {"video-a": "ascoli"},
            {"sampling_fps": 2.0},
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "_combine",
        lambda data, ids: (
            np.zeros((3, 48), dtype=np.float32),
            np.array([0, 1, 3], dtype=np.int32),
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "train_lightgbm",
        lambda *args, **kwargs: pytest.fail("trainer must not run when a class is missing"),
    )

    with pytest.raises(ValueError, match=r"missing class\(es\): \[2\]"):
        pipeline.run_lightgbm_train_from_manifest(
            manifest, {}, checkpoint_path=tmp_path / "model.json",
            config=LightGBMConfig(num_iterations=10, early_stopping_rounds=2),
        )


def test_single_model_training_rejects_checkpoint_directory(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text("videos: []\n", encoding="utf-8")
    destination = tmp_path / "existing-directory"
    destination.mkdir()
    monkeypatch.setattr(
        pipeline,
        "prepare_lightgbm_manifest_data",
        lambda *args, **kwargs: (
            {"video-a": object()},
            {"video-a": "ascoli"},
            {"sampling_fps": 2.0},
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "_combine",
        lambda data, ids: (
            np.zeros((4, 48), dtype=np.float32),
            np.arange(4, dtype=np.int32),
        ),
    )
    with pytest.raises(ValueError, match="checkpoint path is a directory"):
        pipeline.run_lightgbm_train_from_manifest(
            manifest, {}, checkpoint_path=destination,
            config=LightGBMConfig(num_iterations=10, early_stopping_rounds=2),
        )
