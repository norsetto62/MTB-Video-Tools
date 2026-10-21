"""Tests for the three-output positive-unlabeled trainer."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from autocut.ml.training import TrainingConfig, targets_to_classes, train_single_fold, _pu_risk
from autocut.ml.model import MTBClassifier


def _fold(seed=7, train_n=18, val_n=12):
    rng = np.random.default_rng(seed)
    train_X = rng.normal(size=(train_n, 8, 54)).astype(np.float32)
    val_X = rng.normal(loc=0.5, size=(val_n, 8, 54)).astype(np.float32)
    train_y = np.resize(np.array([3, 0, 1, 2], dtype=np.float32), train_n)
    val_y = np.resize(np.array([3, 0, 1, 2], dtype=np.float32), val_n)
    return train_X, train_y, val_X, val_y


def _config(**kwargs):
    values = dict(epochs=2, batch_size=4, hidden_dim=4, classifier_mid_dim=5,
                  dropout=0.0, patience=2, seed=17, positive_prior=0.25)
    values.update(kwargs)
    return TrainingConfig(**values)


def test_scores_map_to_existing_three_output_classes():
    np.testing.assert_array_equal(targets_to_classes(np.array([0, 1, 2, 3])), [0, 1, 1, 2])


@pytest.mark.parametrize("kwargs", [
    {"epochs": 0}, {"batch_size": 1}, {"num_classes": 2},
    {"positive_prior": 0}, {"positive_prior": 1}, {"dropout": 1.0},
])
def test_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        TrainingConfig(**kwargs)


def test_pu_risk_requires_positive_and_unlabeled_examples():
    logits = torch.zeros((3, 3))
    labels = torch.tensor([2, 2, 2])
    with pytest.raises(ValueError, match="both pure score-3 positives"):
        _pu_risk(logits, labels, 0.2)


def test_training_uses_three_output_model_and_pu_risk():
    train_X, train_y, val_X, val_y = _fold()
    model, scaler, summary = train_single_fold(train_X, train_y, val_X, val_y, _config())
    assert model.num_classes == 3
    assert scaler.enabled
    assert summary["positive_windows"] > 0
    assert summary["unlabeled_windows"] > 0
    assert summary["best_val_pu_risk"] >= 0
    assert "class_weights" not in summary


def test_training_rejects_missing_positive_or_unlabeled_set():
    train_X, train_y, val_X, val_y = _fold()
    with pytest.raises(ValueError, match="pure score-3 positive"):
        train_single_fold(train_X, np.zeros(len(train_X)), val_X, val_y, _config())
    with pytest.raises(ValueError, match="unlabeled"):
        train_single_fold(train_X, np.full(len(train_X), 3.0), val_X, val_y, _config())
