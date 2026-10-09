"""Tests for versioned ML checkpoint persistence."""

import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from autocut.ml.checkpoint import (
    CHECKPOINT_VERSION,
    build_checkpoint,
    load_checkpoint,
    restore_model,
    save_checkpoint,
    validate_checkpoint,
)
from autocut.ml.model import MTBClassifier
from autocut.ml.scaling import FeatureScaler
from autocut.motion.features import FEATURE_NAMES


def _same_parameters(left, right):
    for key, value in left.state_dict().items():
        torch.testing.assert_close(value, right.state_dict()[key])


def test_build_checkpoint_captures_model_and_canonical_feature_order():
    model = MTBClassifier()
    payload = build_checkpoint(model)

    assert payload["checkpoint_version"] == CHECKPOINT_VERSION
    assert payload["model_class"] == "MTBClassifier"
    assert payload["feature_dim"] == 54
    assert payload["feature_names"] == list(FEATURE_NAMES)
    assert payload["model_config"] == {
        "hidden_dim": 128,
        "classifier_mid_dim": 64,
        "num_classes": 3,
        "dropout": pytest.approx(0.4),
    }
    assert payload["class_mapping"] == {
        "0": "negative_or_non_riding",
        "1": "ordinary_or_intermediate",
        "2": "interesting",
    }
    assert payload["optimizer_state_dict"] is None
    assert payload["epoch"] is None


def test_save_load_and_restore_model_round_trip(tmp_path):
    torch.manual_seed(17)
    model = MTBClassifier(hidden_dim=12, classifier_mid_dim=7, dropout=0.2)
    scaler = FeatureScaler().fit(np.arange(540, dtype=np.float32).reshape(10, 54))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # Initialize optimizer state so the checkpoint contains more than defaults.
    model(torch.randn(2, 4, 54)).sum().backward()
    optimizer.step()

    path = tmp_path / "nested" / "model.pt"
    save_checkpoint(
        path,
        model,
        scaler=scaler,
        optimizer=optimizer,
        epoch=4,
        best_metric=0.73,
        training_config={"epochs": 15, "seed": 42},
        data_config={"window": 4.0, "stride": 2.0, "flow_fps": 2.0},
        metadata={"source_revision": "abc123"},
    )

    payload = load_checkpoint(path)
    restored = restore_model(payload)
    _same_parameters(model, restored)

    assert payload["epoch"] == 4
    assert payload["best_metric"] == pytest.approx(0.73)
    assert payload["optimizer_state_dict"] is not None
    assert payload["training_config"]["seed"] == 42
    assert payload["metadata"]["source_revision"] == "abc123"
    restored_scaler = FeatureScaler.from_state_dict(payload["scaler_state"])
    np.testing.assert_array_equal(
        restored_scaler.transform(np.ones((2, 54), dtype=np.float32)),
        scaler.transform(np.ones((2, 54), dtype=np.float32)),
    )


def test_restored_model_produces_same_logits():
    torch.manual_seed(5)
    model = MTBClassifier(hidden_dim=8, classifier_mid_dim=5)
    model.eval()
    x = torch.randn(3, 8, 54)
    expected = model(x)

    restored = restore_model(build_checkpoint(model))
    restored.eval()

    torch.testing.assert_close(restored(x), expected)


def test_feature_mask_can_describe_full_schema_and_reduced_model_input():
    model = MTBClassifier(feature_dim=2)
    mask = [True, False, True, False]
    names = ["a", "b", "c", "d"]

    payload = build_checkpoint(model, feature_names=names, feature_mask=mask)

    assert payload["feature_names"] == names
    assert payload["feature_mask"] == mask
    validate_checkpoint(payload)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"feature_names": ["a", "a"]},
        {"feature_names": ["a"]},
        {"feature_mask": [False, False]},
        {"feature_mask": [True, "false"]},
        {"feature_mask": [True, False, True]},
    ],
)
def test_build_rejects_invalid_feature_metadata(kwargs):
    model = MTBClassifier(feature_dim=2)
    with pytest.raises(ValueError):
        build_checkpoint(model, **kwargs)


def test_custom_class_count_requires_explicit_class_mapping():
    model = MTBClassifier(num_classes=4)
    with pytest.raises(ValueError, match="class_mapping"):
        build_checkpoint(model)

    payload = build_checkpoint(
        model,
        class_mapping={"0": "a", "1": "b", "2": "c", "3": "d"},
    )
    assert payload["model_config"]["num_classes"] == 4


@pytest.mark.parametrize("epoch", [-1, 1.5, True])
def test_build_rejects_invalid_epoch(epoch):
    with pytest.raises(ValueError, match="epoch"):
        build_checkpoint(MTBClassifier(), epoch=epoch)


@pytest.mark.parametrize("metric", [float("nan"), float("inf"), -float("inf")])
def test_build_rejects_nonfinite_metric(metric):
    with pytest.raises(ValueError, match="best_metric"):
        build_checkpoint(MTBClassifier(), best_metric=metric)


def test_build_rejects_unfitted_enabled_scaler():
    with pytest.raises(ValueError, match="must be fitted"):
        build_checkpoint(MTBClassifier(), scaler=FeatureScaler())


def test_build_rejects_incompatible_scaler_feature_count():
    scaler = FeatureScaler().fit(np.ones((4, 3), dtype=np.float32))
    with pytest.raises(ValueError):
        build_checkpoint(MTBClassifier(feature_dim=2), scaler=scaler)


def test_save_refuses_directory_as_destination(tmp_path):
    with pytest.raises(IsADirectoryError):
        save_checkpoint(tmp_path, MTBClassifier())


def test_load_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Checkpoint file"):
        load_checkpoint(tmp_path / "missing.pt")


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda p: p.update(checkpoint_version="999"), "Unsupported checkpoint_version"),
        (lambda p: p.pop("feature_dim"), "missing required fields"),
        (lambda p: p.update(feature_dim=0), "feature_dim"),
        (lambda p: p.update(model_class="OtherModel"), "model_class"),
        (lambda p: p.update(feature_mask=[False] * 54), "feature_mask"),
        (lambda p: p.update(class_mapping={"0": "a"}), "class_mapping"),
        (lambda p: p.update(scaler_state={"enabled": True, "mean": [0.0], "scale": [1.0]}), "scaler_state"),
        (lambda p: p.update(epoch=-1), "epoch"),
    ],
)
def test_validation_rejects_corrupted_payload(mutate, message):
    payload = copy.deepcopy(build_checkpoint(MTBClassifier()))
    mutate(payload)

    with pytest.raises(ValueError, match=message):
        validate_checkpoint(payload)


def test_load_rejects_non_checkpoint_dictionary(tmp_path):
    path = tmp_path / "bad.pt"
    torch.save(["not", "a", "checkpoint"], path)

    with pytest.raises(ValueError, match="dictionary"):
        load_checkpoint(path)


def test_restore_model_rejects_incompatible_weights():
    payload = build_checkpoint(MTBClassifier())
    payload["model_state_dict"]["classifier.3.weight"] = torch.zeros(9, 9)

    with pytest.raises(ValueError, match="incompatible"):
        restore_model(payload)
