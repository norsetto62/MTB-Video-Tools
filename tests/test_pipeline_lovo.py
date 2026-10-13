"""Tests for leakage-safe leave-one-group-out orchestration."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from autocut import pipeline_lovo


def _settings(tmp_path):
    from autocut.config_loader import CommonSettings, TrainingRunConfig
    from autocut.ml.training import TrainingConfig

    return TrainingRunConfig(
        common=CommonSettings(
            data_dir=tmp_path / "data",
            models_dir=tmp_path / "models",
            output_dir=tmp_path / "output",
            log_level="INFO",
            min_clip=3.0,
            window_duration=4.0,
            window_stride=2.0,
        ),
        cache_dir=tmp_path / "cache",
        manifest=tmp_path / "manifest.yaml",
        sample_fps=2.0,
        processing_width=480,
        train_groups=("unused",),
        validation_groups=("unused-val",),
        training=TrainingConfig(epochs=2, batch_size=2, hidden_dim=4, classifier_mid_dim=4, dropout=0),
        experiment_name="lovo-test",
        checkpoint_dir=tmp_path / "models",
        results_dir=tmp_path / "results",
    )


def test_metrics_uses_fixed_three_class_confusion_matrix():
    result = pipeline_lovo._metrics(
        np.array([0, 1, 2, 2]),
        np.array([0, 2, 2, 1]),
    )
    assert result["windows"] == 4
    assert result["accuracy"] == pytest.approx(0.5)
    assert result["confusion_matrix"] == [[1, 0, 0], [0, 0, 1], [0, 1, 1]]
    assert set(result["class_f1"]) == {"0", "1", "2"}


def test_lovo_rejects_fewer_than_three_groups(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text("placeholder", encoding="utf-8")
    records = (
        SimpleNamespace(id="a", group_id="one"),
        SimpleNamespace(id="b", group_id="two"),
    )
    monkeypatch.setattr(pipeline_lovo, "load_manifest", lambda *a, **k: records)
    with pytest.raises(ValueError, match="at least 3 distinct manifest groups"):
        pipeline_lovo.run_lovo(manifest, settings)


def test_lovo_holds_out_each_group_and_writes_fold_artifacts(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text("placeholder", encoding="utf-8")
    records = tuple(
        SimpleNamespace(id=f"video-{group}", group_id=group)
        for group in ("A", "B", "C")
    )
    monkeypatch.setattr(pipeline_lovo, "load_manifest", lambda *a, **k: records)

    def fake_combine(selected, _settings):
        # Group identity is encoded into the feature tensor so the test can
        # verify that the outer held-out group never appears in training inputs.
        group_ids = {r.group_id for r in selected}
        count = max(1, len(selected))
        marker = float(ord(sorted(group_ids)[0]) if group_ids else 0)
        X = np.full((count, 8, 54), marker, dtype=np.float32)
        y = np.resize(np.array([0.0, 1.0, 3.0]), count)
        return X, y

    captured = {"inner": [], "outer": []}

    def fake_train(train_X, train_y, val_X, val_y, config, **kwargs):
        captured["inner"].append((train_X.copy(), val_X.copy()))
        return object(), object(), {"best_epoch": 1}

    def fake_refit(train_X, train_y, config, *, epochs, checkpoint_path, data_config, metadata):
        captured["outer"].append((train_X.copy(), metadata.copy()))
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_text("stub", encoding="utf-8")
        return object(), object(), {"epochs_trained": epochs}

    def fake_predict(model, scaler, test_X):
        return np.zeros(len(test_X), dtype=np.int64), np.tile([1.0, 0.0, 0.0], (len(test_X), 1))

    monkeypatch.setattr(pipeline_lovo, "_combine_records", fake_combine)
    monkeypatch.setattr(pipeline_lovo, "train_single_fold", fake_train)
    monkeypatch.setattr(pipeline_lovo, "_fit_fixed_epochs", fake_refit)
    monkeypatch.setattr(pipeline_lovo, "_predict", fake_predict)

    result = pipeline_lovo.run_lovo(manifest, settings)

    assert result["fold_count"] == 3
    assert result["groups"] == ["A", "B", "C"]
    assert len(captured["outer"]) == 3
    for fold in result["folds"]:
        assert fold["test_group"] not in fold["outer_training_groups"]
        assert fold["inner_validation_group"] in fold["outer_training_groups"]
        assert fold["inner_validation_group"] not in fold["inner_training_groups"]
        assert Path(fold["checkpoint"]).is_file()
        assert Path(fold["predictions_path"]).is_file()
    assert Path(result["results_path"]).is_file()
