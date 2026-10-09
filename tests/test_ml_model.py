"""Tests for the legacy-compatible ML classifier."""

import pytest

torch = pytest.importorskip("torch")

from autocut.ml.model import MTBClassifier


def test_default_model_matches_legacy_architecture():
    model = MTBClassifier()

    assert model.feature_dim == 54
    assert model.hidden_dim == 128
    assert model.num_classes == 3
    assert model.dropout_probability == pytest.approx(0.4)
    assert model.classifier_mid_dim == 64
    assert model.backbone.batch_first is True
    assert model.backbone.bidirectional is True
    assert model.backbone.input_size == 54
    assert model.backbone.hidden_size == 128
    assert model.classifier[0].p == pytest.approx(0.4)
    assert model.classifier[1].in_features == 256
    assert model.classifier[1].out_features == 64
    assert model.classifier[2].__class__.__name__ == "ReLU"
    assert model.classifier[3].in_features == 64
    assert model.classifier[3].out_features == 3


@pytest.mark.parametrize("batch_size", [1, 4])
def test_forward_returns_one_logit_vector_per_window(batch_size):
    model = MTBClassifier()
    x = torch.randn(batch_size, 8, 54)

    logits = model(x)

    assert logits.shape == (batch_size, 3)
    assert logits.dtype == torch.float32
    assert torch.isfinite(logits).all()


def test_forward_supports_configurable_dimensions():
    model = MTBClassifier(
        feature_dim=12,
        hidden_dim=10,
        num_classes=4,
        dropout=0.2,
        classifier_mid_dim=7,
    )
    x = torch.randn(2, 5, 12)

    assert model(x).shape == (2, 4)


def test_forward_backpropagates_gradients():
    model = MTBClassifier()
    x = torch.randn(2, 8, 54)

    loss = model(x).sum()
    loss.backward()

    assert model.backbone.weight_ih_l0.grad is not None
    assert model.classifier[3].weight.grad is not None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"feature_dim": 0},
        {"feature_dim": -1},
        {"hidden_dim": 0},
        {"num_classes": 0},
        {"classifier_mid_dim": 0},
        {"feature_dim": 54.0},
        {"hidden_dim": True},
        {"dropout": -0.01},
        {"dropout": 1.0},
        {"dropout": float("nan")},
    ],
)
def test_constructor_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        MTBClassifier(**kwargs)


@pytest.mark.parametrize(
    "x, error, message",
    [
        (torch.randn(8, 54), ValueError, "3D"),
        (torch.empty(0, 8, 54), ValueError, "at least one sequence"),
        (torch.empty(2, 0, 54), ValueError, "at least one time step"),
        (torch.randn(2, 8, 53), ValueError, "54 features"),
        (torch.ones(2, 8, 54, dtype=torch.int64), TypeError, "floating-point"),
        (
            torch.full((2, 8, 54), float("nan")),
            ValueError,
            "finite values",
        ),
        (
            torch.full((2, 8, 54), float("inf")),
            ValueError,
            "finite values",
        ),
    ],
)
def test_forward_rejects_invalid_inputs(x, error, message):
    model = MTBClassifier()

    with pytest.raises(error, match=message):
        model(x)


def test_forward_rejects_input_with_wrong_floating_dtype():
    model = MTBClassifier()
    x = torch.randn(2, 8, 54, dtype=torch.float64)

    with pytest.raises(TypeError, match="dtype torch.float32"):
        model(x)


def test_forward_rejects_non_tensor_input():
    model = MTBClassifier()

    with pytest.raises(TypeError, match="torch.Tensor"):
        model([[0.0] * 54])
