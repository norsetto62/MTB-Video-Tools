"""Tests for single-logit positive-unlabeled training."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from autocut.ml.training import TrainingConfig, targets_to_classes, train_single_fold, _pu_risk, _training_loss
from autocut.ml.model import MTBClassifier


def _fold(seed=7, train_n=18, val_n=12, steps=8):
    rng = np.random.default_rng(seed)
    train_X = rng.normal(size=(train_n, steps, 54)).astype(np.float32)
    val_X = rng.normal(loc=0.5, size=(val_n, steps, 54)).astype(np.float32)
    train_y = np.resize(np.array([3, 0, 1, 2], dtype=np.float32), train_n)
    val_y = np.resize(np.array([3, 0, 1, 2], dtype=np.float32), val_n)
    return train_X, train_y, val_X, val_y


def _config(**kwargs):
    values = dict(epochs=2, batch_size=4, hidden_dim=4, classifier_mid_dim=5,
                  dropout=0.0, patience=2, seed=17, positive_prior=0.25)
    values.update(kwargs)
    return TrainingConfig(**values)


def test_scores_map_to_binary_pu_labels():
    np.testing.assert_array_equal(targets_to_classes(np.array([0, 1, 2, 3])), [0, 0, 0, 1])


@pytest.mark.parametrize("kwargs", [
    {"epochs": 0}, {"batch_size": 1},
    {"positive_prior": 0}, {"positive_prior": 1}, {"dropout": 1.0},
])
def test_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        TrainingConfig(**kwargs)


def test_pu_risk_requires_positive_and_unlabeled_examples():
    logits = torch.zeros((3, 1))
    labels = torch.tensor([1, 1, 1])
    with pytest.raises(ValueError, match="both pure score-3 positives"):
        _pu_risk(logits, labels, 0.2)


def test_training_uses_single_logit_and_nnpu_risk_only():
    train_X, train_y, val_X, val_y = _fold()
    model, scaler, summary = train_single_fold(train_X, train_y, val_X, val_y, _config())
    assert model.num_classes == 1
    assert model(torch.from_numpy(train_X[:2])).shape == (2, 1)
    assert scaler.enabled
    assert summary["positive_windows"] > 0
    assert summary["unlabeled_windows"] > 0
    assert summary["best_val_pu_risk"] >= 0
    assert "unlabeled_aux_weight" not in summary


def test_training_loss_is_exactly_nnpu_risk():
    logits = torch.tensor([[1.0], [-0.5], [0.25], [-1.0]], requires_grad=True)
    labels = torch.tensor([1, 0, 0, 1])
    config = _config(positive_prior=0.2)
    expected = _pu_risk(logits, labels, config.positive_prior)
    actual = _training_loss(logits, labels, config)
    torch.testing.assert_close(actual, expected)


def test_training_rejects_missing_positive_or_unlabeled_set():
    train_X, train_y, val_X, val_y = _fold()
    with pytest.raises(ValueError, match="pure score-3 positive"):
        train_single_fold(train_X, np.zeros(len(train_X)), val_X, val_y, _config())
    with pytest.raises(ValueError, match="unlabeled"):
        train_single_fold(train_X, np.full(len(train_X), 3.0), val_X, val_y, _config())



def test_training_accepts_window_length_derived_from_fps_and_duration():
    train_X, train_y, val_X, val_y = _fold(steps=5)
    data_config = {"flow_fps": 1.0, "window": 5.0, "stride": 1.0}
    model, _scaler, summary = train_single_fold(
        train_X, train_y, val_X, val_y, _config(), data_config=data_config
    )
    assert model.num_classes == 1
    assert summary["best_val_pu_risk"] >= 0


def test_training_rejects_window_length_inconsistent_with_fps_and_duration():
    train_X, train_y, val_X, val_y = _fold(steps=5)
    with pytest.raises(ValueError, match="must contain 8 time steps"):
        train_single_fold(
            train_X, train_y, val_X, val_y, _config(),
            data_config={"flow_fps": 2.0, "window": 4.0, "stride": 2.0},
        )
