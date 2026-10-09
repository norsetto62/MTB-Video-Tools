"""Focused tests for the modular three-class ML trainer."""

import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from autocut.ml.checkpoint import load_checkpoint, restore_model
from autocut.ml.model import MTBClassifier
from autocut.ml.scaling import FeatureScaler
from autocut.ml.training import TrainingConfig, targets_to_classes, train_single_fold
import autocut.ml.training as training


def _fold(seed=7, train_n=12, val_n=6):
    rng = np.random.default_rng(seed)
    train_X = rng.normal(size=(train_n, 8, 54)).astype(np.float32)
    val_X = rng.normal(loc=5.0, size=(val_n, 8, 54)).astype(np.float32)
    # Ensure all three classes appear in both splits.
    train_y = np.resize(np.array([0.0, 1.0, 3.0]), train_n).astype(np.float32)
    val_y = np.resize(np.array([0.0, 1.5, 2.5]), val_n).astype(np.float32)
    return train_X, train_y, val_X, val_y


def _config(**kwargs):
    defaults = dict(
        epochs=3,
        batch_size=4,
        hidden_dim=4,
        classifier_mid_dim=5,
        dropout=0.0,
        patience=3,
        seed=17,
    )
    defaults.update(kwargs)
    return TrainingConfig(**defaults)


def test_targets_to_classes_uses_exact_boundaries():
    targets = np.array([0.0, 0.4999, 0.5, 2.4999, 2.5, 3.0])
    np.testing.assert_array_equal(targets_to_classes(targets), [0, 0, 1, 1, 2, 2])


@pytest.mark.parametrize(
    "targets",
    [
        np.array([]),
        np.array([[0.0, 1.0]]),
        np.array([np.nan]),
        np.array([np.inf]),
        np.array([-0.01]),
        np.array([3.01]),
        np.array(["bad"]),
    ],
)
def test_target_conversion_rejects_invalid_targets(targets):
    with pytest.raises(ValueError):
        targets_to_classes(targets)


def test_config_rejects_invalid_hyperparameters():
    for kwargs in (
        {"epochs": 0},
        {"batch_size": True},
        {"lr": 0},
        {"lr": float("nan")},
        {"dropout": 1.0},
        {"patience": -1},
        {"min_delta": -0.1},
        {"use_scaling": 1},
        {"class_weights": (1.0, 2.0)},
        {"class_weights": (1.0, 0.0, 2.0)},
        {"num_classes": 2},
    ):
        with pytest.raises(ValueError):
            TrainingConfig(**kwargs)


@pytest.mark.parametrize(
    "which, bad",
    [
        ("train_windows", np.zeros((4, 7, 54), dtype=np.float32)),
        ("train_windows", np.full((4, 8, 54), np.nan, dtype=np.float32)),
        ("val_windows", np.zeros((4, 8, 53), dtype=np.float32)),
    ],
)
def test_train_rejects_invalid_windows(which, bad):
    train_X, train_y, val_X, val_y = _fold()
    args = {
        "train_windows": train_X,
        "train_targets": train_y,
        "val_windows": val_X,
        "val_targets": val_y,
        "config": _config(epochs=1),
    }
    args[which] = bad
    with pytest.raises(ValueError):
        train_single_fold(**args)


def test_train_rejects_target_count_mismatch_and_single_class_training():
    train_X, train_y, val_X, val_y = _fold()
    with pytest.raises(ValueError, match="same number"):
        train_single_fold(train_X, train_y[:-1], val_X, val_y, _config(epochs=1))
    with pytest.raises(ValueError, match="at least two distinct classes"):
        train_single_fold(
            train_X,
            np.zeros(len(train_X)),
            val_X,
            val_y,
            _config(epochs=1),
        )


def test_scaler_is_fitted_only_on_training_data():
    train_X, train_y, val_X, val_y = _fold()
    _, scaler, _ = train_single_fold(
        train_X, train_y, val_X, val_y, _config(epochs=1)
    )

    expected_mean = train_X.mean(axis=(0, 1))
    np.testing.assert_allclose(scaler.mean_, expected_mean, rtol=1e-5, atol=1e-5)
    assert not np.allclose(scaler.mean_, val_X.mean(axis=(0, 1)))


def test_disabled_scaling_preserves_values():
    train_X, train_y, val_X, val_y = _fold()
    _, scaler, _ = train_single_fold(
        train_X, train_y, val_X, val_y, _config(epochs=1, use_scaling=False)
    )

    assert scaler.enabled is False
    np.testing.assert_array_equal(scaler.transform(val_X), val_X)


def test_class_weights_are_passed_to_cross_entropy(monkeypatch):
    original = torch.nn.CrossEntropyLoss
    captured = []

    def recording_loss(*args, **kwargs):
        captured.append(kwargs.get("weight"))
        return original(*args, **kwargs)

    monkeypatch.setattr(training.nn, "CrossEntropyLoss", recording_loss)
    train_X, train_y, val_X, val_y = _fold()
    weights = (0.4, 1.2, 4.5)
    train_single_fold(
        train_X, train_y, val_X, val_y, _config(epochs=1, class_weights=weights)
    )

    assert len(captured) == 1
    torch.testing.assert_close(captured[0], torch.tensor(weights, dtype=torch.float32))


def test_validation_forward_runs_without_gradients(monkeypatch):
    original_forward = MTBClassifier.forward
    observations = []

    def checked_forward(self, x):
        if not self.training:
            observations.append(torch.is_grad_enabled())
        return original_forward(self, x)

    monkeypatch.setattr(MTBClassifier, "forward", checked_forward)
    train_X, train_y, val_X, val_y = _fold()
    train_single_fold(train_X, train_y, val_X, val_y, _config(epochs=2))

    assert observations
    assert observations == [False] * len(observations)


def test_best_weights_are_restored_and_summary_matches_best_epoch(monkeypatch):
    original_evaluate = training._evaluate
    snapshots = []
    losses = iter([0.5, 0.8, 0.9])

    def controlled_evaluate(model, *args, **kwargs):
        snapshots.append(copy.deepcopy(model.state_dict()))
        loss = next(losses)
        return loss, 0.25 + len(snapshots) / 100

    monkeypatch.setattr(training, "_evaluate", controlled_evaluate)
    train_X, train_y, val_X, val_y = _fold()
    model, _, summary = train_single_fold(
        train_X, train_y, val_X, val_y, _config(epochs=3, patience=3)
    )

    assert summary["best_epoch"] == 1
    assert summary["best_val_loss"] == pytest.approx(0.5)
    assert summary["best_val_f1"] == pytest.approx(0.26)
    for key, expected in snapshots[0].items():
        torch.testing.assert_close(model.state_dict()[key], expected)
    assert model.training is False


def test_early_stopping_obeys_patience(monkeypatch):
    calls = []
    losses = iter([0.5, 0.6, 0.7, 0.8])

    def controlled_evaluate(model, *args, **kwargs):
        calls.append(len(calls) + 1)
        return next(losses), 0.2

    monkeypatch.setattr(training, "_evaluate", controlled_evaluate)
    train_X, train_y, val_X, val_y = _fold()
    _, _, summary = train_single_fold(
        train_X, train_y, val_X, val_y, _config(epochs=10, patience=2)
    )

    assert len(calls) == 3
    assert summary["epochs_completed"] == 3
    assert summary["stopped_early"] is True
    assert summary["best_epoch"] == 1


def test_min_delta_controls_model_selection(monkeypatch):
    losses = iter([0.5, 0.49995, 0.4998])
    monkeypatch.setattr(
        training,
        "_evaluate",
        lambda *args, **kwargs: (next(losses), 0.5),
    )
    train_X, train_y, val_X, val_y = _fold()
    _, _, summary = train_single_fold(
        train_X,
        train_y,
        val_X,
        val_y,
        _config(epochs=3, patience=5, min_delta=1e-4),
    )

    assert summary["best_epoch"] == 3
    assert summary["best_val_loss"] == pytest.approx(0.4998)


def test_checkpoint_contains_restored_model_scaler_and_training_metadata(tmp_path):
    train_X, train_y, val_X, val_y = _fold()
    path = tmp_path / "fold" / "best.pt"
    model, scaler, summary = train_single_fold(
        train_X,
        train_y,
        val_X,
        val_y,
        _config(epochs=2),
        checkpoint_path=path,
        fold_video_id="validation-ride",
    )

    payload = load_checkpoint(path)
    restored = restore_model(payload)
    restored_scaler = FeatureScaler.from_state_dict(payload["scaler_state"])

    assert payload["epoch"] == summary["best_epoch"]
    assert payload["best_metric"] == pytest.approx(summary["best_val_loss"])
    assert payload["metadata"]["best_val_f1"] == pytest.approx(summary["best_val_f1"])
    assert payload["metadata"]["fold_video_id"] == "validation-ride"
    assert payload["training_config"]["class_weights"] == [0.2, 1.0, 6.5]
    assert restored.feature_dim == model.feature_dim
    assert restored.hidden_dim == model.hidden_dim
    np.testing.assert_array_equal(restored_scaler.mean_, scaler.mean_)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value.cpu())


def test_no_checkpoint_is_written_without_explicit_path(tmp_path):
    train_X, train_y, val_X, val_y = _fold()
    train_single_fold(train_X, train_y, val_X, val_y, _config(epochs=1))
    assert list(tmp_path.iterdir()) == []
