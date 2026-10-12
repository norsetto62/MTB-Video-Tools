"""Tests for the fixed-group training orchestration and CLI dispatch."""

import json
from pathlib import Path

import numpy as np
import pytest

from autocut.config_loader import load_training_config
from autocut.dataset.manifest import load_manifest
from autocut import pipeline_training


def _settings(tmp_path: Path, *, train_groups=("train",), validation_groups=("val",)):
    config = tmp_path / "train.yaml"
    manifest = tmp_path / "manifest.yaml"
    config.write_text(
        f"""
paths:
  cache_dir: {tmp_path.as_posix()}/cache
data:
  manifest: {manifest.as_posix()}
  sampling_fps: 2
  processing_width: 480
  window_duration: 4
  window_stride: 2
  train_groups: {list(train_groups)}
  validation_groups: {list(validation_groups)}
training:
  epochs: 1
  batch_size: 4
  hidden_dim: 4
  classifier_mid_dim: 4
  dropout: 0
  patience: 1
experiment:
  name: train-test
  checkpoint_dir: {tmp_path.as_posix()}/models
  results_dir: {tmp_path.as_posix()}/results
""",
        encoding="utf-8",
    )
    return load_training_config(config), manifest


def _manifest(path: Path, tmp_path: Path):
    entries = []
    for video_id, group in (("a", "train"), ("b", "val")):
        video = tmp_path / f"{video_id}.mp4"
        annotation = tmp_path / f"{video_id}.txt"
        video.touch()
        annotation.touch()
        entries.append(
            f"  - id: {video_id}\n    group_id: {group}\n"
            f"    video: {video.name}\n    annotation: {annotation.name}"
        )
    path.write_text("videos:\n" + "\n".join(entries) + "\n", encoding="utf-8")


def test_run_training_combines_groups_and_writes_checkpoint_and_json(tmp_path, monkeypatch):
    settings, manifest = _settings(tmp_path)
    _manifest(manifest, tmp_path)
    captured = {}

    def fake_combine(records, settings):
        # Make it explicit that train and validation receive separate records.
        captured.setdefault("record_ids", []).append([r.id for r in records])
        count = 9 if records[0].group_id == "train" else 6
        X = np.zeros((count, 8, 54), dtype=np.float32)
        y = np.resize(np.array([0.0, 1.0, 3.0], dtype=np.float32), count)
        return X, y

    def fake_train(train_X, train_y, val_X, val_y, config, **kwargs):
        captured["shapes"] = (train_X.shape, val_X.shape)
        captured["kwargs"] = kwargs
        return object(), object(), {
            "best_epoch": 1,
            "best_val_loss": 0.5,
            "best_val_f1": 0.4,
        }

    monkeypatch.setattr(pipeline_training, "_combine_records", fake_combine)
    monkeypatch.setattr(pipeline_training, "train_single_fold", fake_train)
    result = pipeline_training.run_training(manifest, settings)

    assert captured["record_ids"] == [["a"], ["b"]]
    assert captured["shapes"] == ((9, 8, 54), (6, 8, 54))
    assert captured["kwargs"]["checkpoint_path"] == tmp_path / "models" / "train-test_best.pt"
    assert result["train_groups"] == ["train"]
    assert result["validation_groups"] == ["val"]
    assert result["train_windows"] == 9
    assert result["validation_windows"] == 6
    saved = json.loads((tmp_path / "results" / "train-test_train.json").read_text())
    assert saved["summary"]["best_epoch"] == 1
    assert saved["checkpoint"].endswith("train-test_best.pt")


def test_run_training_rejects_unknown_configured_groups(tmp_path):
    settings, manifest = _settings(tmp_path, train_groups=("missing",), validation_groups=("val",))
    _manifest(manifest, tmp_path)
    with pytest.raises(ValueError, match="not found in the manifest"):
        pipeline_training.run_training(manifest, settings)


def test_training_annotation_video_must_match_manifest_record(tmp_path):
    settings, manifest = _settings(tmp_path)
    _manifest(manifest, tmp_path)
    records = load_manifest(manifest, verify_files=True)
    from dataclasses import replace
    wrong = replace(records[0], video_path=tmp_path / "other.mp4")
    with pytest.raises(ValueError, match="annotation file declares"):
        pipeline_training._load_record_windows(wrong, settings)


def test_cli_train_dispatch_calls_pipeline(tmp_path, monkeypatch, capsys):
    from autocut import cli

    settings, manifest = _settings(tmp_path)
    _manifest(manifest, tmp_path)
    config_path = tmp_path / "train.yaml"
    captured = {}
    monkeypatch.setattr("autocut.config_loader.load_training_config", lambda path: settings)

    def fake_run(path, loaded_settings):
        captured["manifest"] = path
        return {"experiment_name": "train-test", "summary": {"best_epoch": 1}}

    monkeypatch.setattr("autocut.pipeline_training.run_training", fake_run)
    assert cli.main(["train", str(manifest), "--config", str(config_path)]) == 0
    assert captured["manifest"] == manifest
    assert '"best_epoch": 1' in capsys.readouterr().out
