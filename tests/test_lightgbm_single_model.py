from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from autocut.ml.model import LightGBMConfig
from autocut.pipeline import lightgbm_training as pipeline


def _fake_dataset(monkeypatch, tmp_path, *, train_labels=None):
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text("videos: []\n", encoding="utf-8")
    records = (
        SimpleNamespace(id="video-a", group_id="train"),
        SimpleNamespace(id="video-b", group_id="validation"),
        SimpleNamespace(id="video-c", group_id="other"),
    )
    prepared = {
        record.id: SimpleNamespace(
            windows=SimpleNamespace(
                timestamps=np.column_stack((
                    np.arange(len(labels_by_video[record.id]), dtype=float) * 2.0,
                    np.arange(len(labels_by_video[record.id]), dtype=float) * 2.0 + 4.0,
                ))
            )
        )
        for record in records
    }
    group_by_video = {record.id: record.group_id for record in records}
    data_config = {"sampling_fps": 2.0, "window": 4.0, "stride": 2.0}
    captured = {"train_calls": []}
    labels_by_video = {
        "video-a": np.asarray(train_labels if train_labels is not None else [0, 1, 2, 3], dtype=np.int32),
        "video-b": np.asarray([0, 1], dtype=np.int32),
        "video-c": np.asarray([0, 1, 2, 3], dtype=np.int32),
    }

    monkeypatch.setattr(pipeline, "load_manifest", lambda *args, **kwargs: records)

    def fake_prepare(path, settings, *, records=None, require_multiple_groups=True):
        captured["prepare"] = (tuple(records), require_multiple_groups)
        return prepared, group_by_video, data_config

    def fake_combine(data, video_ids):
        ids = tuple(video_ids)
        captured.setdefault("combine", []).append(ids)
        labels = np.concatenate([labels_by_video[video_id] for video_id in ids])
        return np.zeros((len(labels), 48), dtype=np.float32), labels

    def fake_combine_with_group_sizes(data, video_ids):
        ids = tuple(video_ids)
        X, y = fake_combine(data, ids)
        return X, y, [len(labels_by_video[video_id]) for video_id in ids]

    def fake_train(X, y, **kwargs):
        captured["train_calls"].append((X, y, kwargs))
        if kwargs.get("val_X") is not None:
            return object(), {
                "best_iteration": 7,
                "evaluation_history": {"train": {"ndcg@10": [0.7]}, "validation": {"ndcg@10": [0.8]}},
            }
        return object(), {"best_iteration": 7, "training_rows": len(X), "checkpoint_path": str(kwargs["checkpoint_path"])}

    monkeypatch.setattr(pipeline, "prepare_lightgbm_manifest_data", fake_prepare)
    monkeypatch.setattr(pipeline, "_combine", fake_combine)
    monkeypatch.setattr(pipeline, "_combine_with_group_sizes", fake_combine_with_group_sizes)
    monkeypatch.setattr(pipeline, "train_lightgbm", fake_train)
    return manifest, records, captured


def test_legacy_multiclass_yaml_is_migrated_to_lambdarank():
    config = pipeline._lightgbm_config({
        "lightgbm": {
            "objective": "multiclass",
            "num_class": 4,
            "num_iterations": 17,
        }
    })
    assert config.objective == "lambdarank"
    assert config.num_iterations == 17
    assert config.to_params()["metric"] == "ndcg"


def test_single_model_uses_yaml_groups_for_validation_then_refits_all_data(tmp_path, monkeypatch):
    manifest, records, captured = _fake_dataset(monkeypatch, tmp_path)
    checkpoint = tmp_path / "models" / "single.json"
    config = LightGBMConfig(num_iterations=10, early_stopping_rounds=2)
    settings = {
        "data": {"train_groups": ["train"], "validation_groups": ["validation"]},
        "experiment": {},
    }

    result = pipeline.run_lightgbm_train_from_manifest(
        manifest, settings, checkpoint_path=checkpoint, config=config
    )

    assert captured["prepare"] == (records, False)
    assert captured["combine"] == [
        ("video-a",), ("video-b",), ("video-a", "video-b", "video-c")
    ]
    assert len(captured["train_calls"]) == 2
    _, _, validation_call = captured["train_calls"][0]
    assert validation_call["val_X"] is not None
    assert validation_call["val_y"] is not None
    assert validation_call["config"] is config
    _, _, refit_call = captured["train_calls"][1]
    assert refit_call.get("val_X") is None
    assert refit_call["config"].num_iterations == 7
    assert refit_call["checkpoint_path"] == checkpoint
    assert result["validation_selection"]["validation_groups"] == ["validation"]
    assert result["validation_selection"]["best_iteration"] == 7
    assert result["validation_selection"]["evaluation_history"]["validation"]["ndcg@10"] == [0.8]
    assert result["training_video_ids"] == ["video-a", "video-b", "video-c"]


def test_single_model_rejects_overlapping_configured_groups(tmp_path, monkeypatch):
    manifest, _, captured = _fake_dataset(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="must be disjoint"):
        pipeline.run_lightgbm_train_from_manifest(
            manifest,
            {"data": {"train_groups": ["train"], "validation_groups": ["train"]}},
            checkpoint_path=tmp_path / "model.json",
            config=LightGBMConfig(num_iterations=10, early_stopping_rounds=2),
        )
    assert "prepare" not in captured


def test_single_model_rejects_training_video_without_relevance_pairs(tmp_path, monkeypatch):
    manifest, _, captured = _fake_dataset(
        monkeypatch, tmp_path, train_labels=[1, 1, 1, 1]
    )
    with pytest.raises(ValueError, match="multiple relevance grades"):
        pipeline.run_lightgbm_train_from_manifest(
            manifest,
            {"data": {"train_groups": ["train"], "validation_groups": ["validation"]}},
            checkpoint_path=tmp_path / "model.json",
            config=LightGBMConfig(num_iterations=10, early_stopping_rounds=2),
        )
    assert captured["train_calls"] == []


def test_single_model_rejects_checkpoint_directory(tmp_path, monkeypatch):
    manifest, _, _ = _fake_dataset(monkeypatch, tmp_path)
    destination = tmp_path / "existing-directory"
    destination.mkdir()
    with pytest.raises(ValueError, match="checkpoint path is a directory"):
        pipeline.run_lightgbm_train_from_manifest(
            manifest,
            {"data": {"train_groups": ["train"], "validation_groups": ["validation"]}},
            checkpoint_path=destination,
            config=LightGBMConfig(num_iterations=10, early_stopping_rounds=2),
        )
