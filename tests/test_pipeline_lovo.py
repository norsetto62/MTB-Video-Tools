"""Tests for leakage-safe leave-one-group-out orchestration."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from autocut.pipeline import lovo as pipeline_lovo


def _settings(tmp_path):
    from autocut.config_loader import Config

    return Config({
        "paths": {
            "cache_dir": str(tmp_path / "cache"),
            "models_dir": str(tmp_path / "models"),
        },
        "data": {
            "manifest": str(tmp_path / "manifest.yaml"),
            "sampling_fps": 2.0,
            "processing_width": 480,
            "window_duration": 4.0,
            "window_stride": 2.0,
            "train_groups": ["unused"],
            "validation_groups": ["unused-val"],
        },
        "training": {
            "epochs": 2,
            "batch_size": 2,
            "hidden_dim": 4,
            "classifier_mid_dim": 4,
            "dropout": 0,
        },
        "experiment": {
            "name": "lovo-test",
            "checkpoint_dir": str(tmp_path / "models"),
            "results_dir": str(tmp_path / "results"),
        },
        "logging": {"level": "INFO"},
    })


def test_metrics_does_not_treat_unlabeled_as_negative():
    result = pipeline_lovo._metrics(
        np.array([1, 0, 0, 1]),
        np.array([1, 1, 0, 0]),
    )
    assert result["windows"] == 4
    assert result["positive_windows"] == 2
    assert result["unlabeled_windows"] == 2
    assert result["positive_recall_on_P"] == pytest.approx(0.5)
    assert result["selected_fraction_of_U"] == pytest.approx(0.5)


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
        captured["prepared"].append(tuple(r.id for r in selected))
        # Group identity is encoded into the feature tensor so the test can
        # verify that the outer held-out group never appears in training inputs.
        group_ids = {r.group_id for r in selected}
        count = max(1, len(selected))
        marker = float(ord(sorted(group_ids)[0]) if group_ids else 0)
        X = np.full((count, 8, 54), marker, dtype=np.float32)
        y = np.resize(np.array([3.0, 0.0, 1.0, 2.0]), count)
        return X, y

    captured = {"inner": [], "outer": [], "prepared": []}

    def fake_train(train_X, train_y, val_X, val_y, config, **kwargs):
        captured["inner"].append((train_X.copy(), val_X.copy()))
        return object(), object(), {"best_epoch": 1}

    def fake_refit(train_X, train_y, config, *, epochs, checkpoint_path, data_config, metadata):
        captured["outer"].append((train_X.copy(), metadata.copy()))
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_text("stub", encoding="utf-8")
        return object(), object(), {"epochs_trained": epochs}

    def fake_predict(model, scaler, test_X, **kwargs):
        return np.zeros(len(test_X), dtype=np.int64), np.full(len(test_X), 0.3, dtype=np.float32)

    monkeypatch.setattr(pipeline_lovo, "_combine_records", fake_combine)
    monkeypatch.setattr(pipeline_lovo, "train_single_fold", fake_train)
    monkeypatch.setattr(pipeline_lovo, "_fit_fixed_epochs", fake_refit)
    monkeypatch.setattr(pipeline_lovo, "_predict", fake_predict)

    result = pipeline_lovo.run_lovo(manifest, settings)

    assert result["fold_count"] == 3
    assert result["groups"] == ["A", "B", "C"]
    # Each manifest record is prepared once, rather than once per fold/split.
    assert sorted(captured["prepared"]) == [("video-A",), ("video-B",), ("video-C",)]
    # The LOVO pipeline fits one model per held-out group; production refit is a CLI step.
    assert len(captured["outer"]) == 3
    for fold, (outer_X, metadata) in zip(result["folds"], captured["outer"], strict=True):
        assert fold["test_group"] not in fold["outer_training_groups"]
        # Verify your multi-group consensus inner validation target is active
        assert fold["inner_validation_group"] == "consensus_multi_group"
        assert fold["inner_validation_group"] not in fold["inner_training_groups"]
        assert metadata["outer_test_group"] == fold["test_group"]
        assert not np.any(outer_X[:, 0, 0] == float(ord(fold["test_group"])))
        assert Path(fold["checkpoint"]).is_file()
        assert Path(fold["predictions_path"]).is_file()
    assert Path(result["results_path"]).is_file()


def test_fixed_epoch_refit_saves_scaler_and_training_metadata(tmp_path):
    from autocut.ml.checkpoint import load_checkpoint
    from autocut.ml.training import TrainingConfig, train_fixed_epochs

    rng = np.random.default_rng(7)
    windows = rng.normal(size=(6, 8, 54)).astype(np.float32)
    targets = np.array([3.0, 0.0, 1.0, 2.0, 3.0, 0.0], dtype=np.float32)
    checkpoint = tmp_path / "fold.pt"
    model, scaler, summary = train_fixed_epochs(
        windows,
        targets,
        TrainingConfig(
            epochs=1,
            batch_size=3,
            hidden_dim=4,
            classifier_mid_dim=4,
            dropout=0,
            seed=3,
        ),
        epochs=2,
        checkpoint_path=checkpoint,
        data_config={"flow_fps": 2.0, "window": 4.0, "stride": 2.0},
        metadata={"outer_test_group": "held-out"},
    )

    payload = load_checkpoint(checkpoint)
    assert summary["epochs_trained"] == 2
    assert payload["epoch"] == 2
    assert payload["metadata"]["outer_test_group"] == "held-out"
    assert payload["metadata"]["refit_epochs_selected_by_inner_group"] == 2
    assert payload["scaler_state"] is not None
    assert model.training is False
    assert scaler.mean_ is not None



def test_lovo_prediction_thresholds_positive_probability():
    import torch

    class FixedModel:
        def eval(self):
            return self

        def __call__(self, batch):
            return torch.tensor([[2.0], [-2.0]], dtype=torch.float32)

    class IdentityScaler:
        def transform(self, values):
            return values

    windows = np.zeros((2, 8, 54), dtype=np.float32)
    predicted, probabilities = pipeline_lovo._predict(
        FixedModel(), IdentityScaler(), windows, positive_threshold=0.60
    )
    np.testing.assert_array_equal(predicted, [1, 0])
    assert probabilities.shape == (2,)
    assert probabilities[0] > probabilities[1]

def test_lovo_prediction_rejects_invalid_positive_threshold():
    import torch

    class FixedModel:
        def eval(self):
            return self

    class IdentityScaler:
        def transform(self, values):
            return values

    for threshold in (-0.1, 1.1, float("nan"), True):
        with pytest.raises(ValueError, match="positive_threshold"):
            pipeline_lovo._predict(
                FixedModel(), IdentityScaler(),
                np.zeros((1, 8, 54), dtype=np.float32),
                positive_threshold=threshold,
            )
