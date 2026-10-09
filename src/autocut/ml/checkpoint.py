"""Versioned checkpoint persistence for AutoCut ML models.

This module owns checkpoint serialization and schema validation. It deliberately
does not implement training loops or inference-time feature processing.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import torch

from autocut.ml.model import MTBClassifier
from autocut.ml.scaling import FeatureScaler
from autocut.motion.features import FEATURE_NAMES

CHECKPOINT_VERSION = "1.1"
DEFAULT_CLASS_MAPPING = {
    "0": "negative_or_non_riding",
    "1": "ordinary_or_intermediate",
    "2": "interesting",
}
DEFAULT_DATA_CONFIG = {
    "flow_fps": 2.0,
    "window": 4.0,
    "stride": 2.0,
}


def _plain_mapping(value: Mapping[str, Any] | None, *, name: str) -> dict[str, Any]:
    """Copy a string-keyed mapping so the checkpoint contains plain metadata."""
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping.")
    if not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} keys must be strings.")
    return dict(value)


def _validate_feature_metadata(
    model: MTBClassifier,
    feature_names: Sequence[str] | None,
    feature_mask: Sequence[bool] | None,
) -> tuple[list[str] | None, list[bool] | None]:
    mask = None if feature_mask is None else list(feature_mask)
    if feature_names is None and model.feature_dim == len(FEATURE_NAMES):
        names = list(FEATURE_NAMES)
    else:
        names = None if feature_names is None else list(feature_names)

    if names is not None:
        if not names or any(not isinstance(name, str) or not name for name in names):
            raise ValueError("feature_names must contain non-empty strings.")
        if len(set(names)) != len(names):
            raise ValueError("feature_names must not contain duplicates.")

    if mask is not None:
        if not mask or any(not isinstance(item, bool) for item in mask):
            raise ValueError("feature_mask must be a non-empty sequence of bools.")
        if not any(mask):
            raise ValueError("feature_mask must select at least one feature.")
        if names is not None and len(names) != len(mask):
            raise ValueError("feature_names and feature_mask must have matching lengths.")
        if sum(mask) != model.feature_dim:
            raise ValueError(
                "The number of selected features in feature_mask must match "
                f"model.feature_dim ({model.feature_dim})."
            )
    elif names is not None and len(names) != model.feature_dim:
        raise ValueError(
            "Without feature_mask, feature_names length must match "
            f"model.feature_dim ({model.feature_dim})."
        )

    return names, mask


def build_checkpoint(
    model: MTBClassifier,
    *,
    scaler: FeatureScaler | None = None,
    feature_names: Sequence[str] | None = None,
    feature_mask: Sequence[bool] | None = None,
    class_mapping: Mapping[str, str] | None = None,
    training_config: Mapping[str, Any] | None = None,
    data_config: Mapping[str, Any] | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    epoch: int | None = None,
    best_metric: float | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a checkpoint payload without writing it to disk.

    The feature names and mask describe the full ordered input schema. When a
    mask is supplied, its true entries must equal the model's input dimension.
    Training-only resume fields (optimizer state and epoch) are optional.
    """
    if not isinstance(model, MTBClassifier):
        raise TypeError("model must be an MTBClassifier.")
    if scaler is not None and not isinstance(scaler, FeatureScaler):
        raise TypeError("scaler must be a FeatureScaler or None.")
    if optimizer is not None and not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer must be a torch optimizer or None.")
    if epoch is not None and (
        isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0
    ):
        raise ValueError("epoch must be a non-negative integer or None.")
    if best_metric is not None:
        if isinstance(best_metric, bool) or not isinstance(best_metric, (int, float)):
            raise ValueError("best_metric must be a finite number or None.")
        if not math.isfinite(float(best_metric)):
            raise ValueError("best_metric must be a finite number or None.")

    names, mask = _validate_feature_metadata(model, feature_names, feature_mask)
    mapping = dict(DEFAULT_CLASS_MAPPING if class_mapping is None else class_mapping)
    if not mapping or not all(
        isinstance(key, str) and isinstance(value, str) and value
        for key, value in mapping.items()
    ):
        raise ValueError("class_mapping must map string class IDs to non-empty names.")
    if set(mapping) != {str(index) for index in range(model.num_classes)}:
        raise ValueError("class_mapping keys must cover exactly the model's output classes.")

    model_config = {
        "hidden_dim": model.hidden_dim,
        "classifier_mid_dim": model.classifier_mid_dim,
        "num_classes": model.num_classes,
        "dropout": model.dropout_probability,
    }
    training = _plain_mapping(training_config, name="training_config")
    data = dict(DEFAULT_DATA_CONFIG)
    data.update(_plain_mapping(data_config, name="data_config"))
    for key in ("flow_fps", "window", "stride"):
        value = data.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise ValueError(f"data_config[{key!r}] must be a finite positive number.")
    extra = _plain_mapping(metadata, name="metadata")

    payload: dict[str, Any] = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "model_state_dict": {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        },
        "model_class": "MTBClassifier",
        "feature_dim": model.feature_dim,
        "feature_names": names,
        "feature_mask": mask,
        "model_config": model_config,
        "class_mapping": mapping,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "training_config": training,
        "data_config": data,
        "optimizer_state_dict": (
            optimizer.state_dict() if optimizer is not None else None
        ),
        "epoch": epoch,
        "best_metric": float(best_metric) if best_metric is not None else None,
        "metadata": extra,
    }
    return payload


def save_checkpoint(
    path: str | Path,
    model: MTBClassifier,
    **kwargs: Any,
) -> dict[str, Any]:
    """Build and atomically save a checkpoint, returning its payload.

    Parent directories are created when needed. A temporary file in the target
    directory is replaced atomically after serialization succeeds.
    """
    destination = Path(path)
    if destination.exists() and destination.is_dir():
        raise IsADirectoryError(f"Checkpoint path is a directory: {destination}")

    payload = build_checkpoint(model, **kwargs)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = temporary_file.name
            torch.save(payload, temporary_file)
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass
    return payload


def validate_checkpoint(payload: Any) -> dict[str, Any]:
    """Validate the structural contract of a loaded checkpoint payload."""
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint must contain a dictionary.")
    version = payload.get("checkpoint_version")
    if version != CHECKPOINT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint_version {version!r}; "
            f"expected {CHECKPOINT_VERSION!r}."
        )

    required = ("model_state_dict", "feature_dim", "model_config")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"Checkpoint is missing required fields: {missing}.")

    feature_dim = payload["feature_dim"]
    if isinstance(feature_dim, bool) or not isinstance(feature_dim, int) or feature_dim <= 0:
        raise ValueError("Checkpoint feature_dim must be a positive integer.")

    config = payload["model_config"]
    if not isinstance(config, dict):
        raise ValueError("Checkpoint model_config must be a dictionary.")
    for key in ("hidden_dim", "classifier_mid_dim", "num_classes", "dropout"):
        if key not in config:
            raise ValueError(f"Checkpoint model_config is missing {key!r}.")

    if not isinstance(payload["model_state_dict"], dict):
        raise ValueError("Checkpoint model_state_dict must be a dictionary.")

    names = payload.get("feature_names")
    mask = payload.get("feature_mask")
    if names is not None:
        if not isinstance(names, list) or not names or any(
            not isinstance(name, str) or not name for name in names
        ):
            raise ValueError("Checkpoint feature_names must be a non-empty list of strings.")
        if len(set(names)) != len(names):
            raise ValueError("Checkpoint feature_names must not contain duplicates.")
    if mask is not None:
        if not isinstance(mask, list) or not mask or any(
            not isinstance(item, bool) for item in mask
        ):
            raise ValueError("Checkpoint feature_mask must be a non-empty list of bools.")
        if not any(mask) or sum(mask) != feature_dim:
            raise ValueError("Checkpoint feature_mask does not match feature_dim.")
        if names is not None and len(names) != len(mask):
            raise ValueError("Checkpoint feature_names and feature_mask lengths differ.")
    elif names is not None and len(names) != feature_dim:
        raise ValueError("Checkpoint feature_names length does not match feature_dim.")

    mapping = payload.get("class_mapping")
    if mapping is not None:
        if not isinstance(mapping, dict) or set(mapping) != {
            str(index) for index in range(config["num_classes"])
        }:
            raise ValueError("Checkpoint class_mapping does not match num_classes.")

    scaler_state = payload.get("scaler_state")
    if scaler_state is not None:
        try:
            FeatureScaler.from_state_dict(scaler_state)
        except (TypeError, ValueError) as exc:
            raise ValueError("Checkpoint scaler_state is invalid.") from exc

    for key in ("training_config", "data_config", "metadata"):
        if key in payload and not isinstance(payload[key], dict):
            raise ValueError(f"Checkpoint {key} must be a dictionary.")
    if payload.get("epoch") is not None and (
        isinstance(payload["epoch"], bool)
        or not isinstance(payload["epoch"], int)
        or payload["epoch"] < 0
    ):
        raise ValueError("Checkpoint epoch must be a non-negative integer or None.")

    return payload


def load_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load and validate a checkpoint.

    Uses PyTorch's restricted weights-only loader where available. Checkpoints
    are intended to contain tensors and plain dictionaries/lists/scalars only.
    """
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Checkpoint file does not exist: {source}")

    try:
        payload = torch.load(source, map_location=map_location, weights_only=True)
    except TypeError:
        # Compatibility with older PyTorch releases without weights_only.
        payload = torch.load(source, map_location=map_location)
    except (OSError, RuntimeError, EOFError) as exc:
        raise ValueError(f"Could not read checkpoint file {source}: {exc}") from exc

    return validate_checkpoint(payload)
