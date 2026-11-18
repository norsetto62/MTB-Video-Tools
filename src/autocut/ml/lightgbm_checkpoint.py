"""Serialization and validation for LightGBM checkpoints."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence


LIGHTGBM_CHECKPOINT_VERSION = "lightgbm-1.0"


def save_lightgbm_checkpoint(
    path: str | Path,
    model: Any,
    *,
    feature_names: Sequence[str],
    training_config: Mapping[str, Any],
    data_config: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Atomically save a LightGBM model plus its input schema and provenance."""
    booster = getattr(model, "booster", None)
    if not callable(getattr(booster, "model_to_string", None)):
        raise TypeError("model must wrap a trained LightGBM Booster.")
    names = list(feature_names)
    if not names or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("feature_names must contain non-empty strings.")
    if len(set(names)) != len(names) or len(names) != int(booster.num_feature()):
        raise ValueError("feature_names must be unique and match the booster feature count.")
    config = dict(training_config)
    data = dict(data_config or {})
    extra = dict(metadata or {})
    is_ranker = model.__class__.__name__ == "LightGBMRanker"
    payload = {
        "checkpoint_version": LIGHTGBM_CHECKPOINT_VERSION,
        "model_class": "LightGBMRanker" if is_ranker else "LightGBMClassifier",
        "model_string": booster.model_to_string(),
        "feature_names": names,
        "feature_dim": len(names),
        "class_mapping": None if is_ranker else {"0": "MTB_0", "1": "MTB_1", "2": "MTB_2", "3": "MTB_3"},
        "training_config": config,
        "data_config": data,
        "metadata": extra,
    }
    destination = Path(path)
    if destination.exists() and destination.is_dir():
        raise IsADirectoryError(f"Checkpoint path is a directory: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", prefix=f".{destination.name}.",
            suffix=".tmp", dir=destination.parent, delete=False,
        ) as stream:
            temporary_path = stream.name
            json.dump(payload, stream, ensure_ascii=False)
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass
    return payload


def load_lightgbm_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load and validate a LightGBM JSON checkpoint."""
    import lightgbm as lgb

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"LightGBM checkpoint file does not exist: {source}")
    try:
        with source.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read LightGBM checkpoint {source}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("checkpoint_version") != LIGHTGBM_CHECKPOINT_VERSION:
        raise ValueError("Unsupported or invalid LightGBM checkpoint.")
    names = payload.get("feature_names")
    if (
        not isinstance(names, list) or not names
        or any(not isinstance(name, str) or not name for name in names)
        or len(set(names)) != len(names)
        or payload.get("feature_dim") != len(names)
    ):
        raise ValueError("LightGBM checkpoint feature schema is invalid.")
    model_string = payload.get("model_string")
    if not isinstance(model_string, str) or not model_string:
        raise ValueError("LightGBM checkpoint model_string is missing.")
    booster = lgb.Booster(model_str=model_string)
    if int(booster.num_feature()) != len(names):
        raise ValueError("LightGBM checkpoint feature schema does not match its model.")
    model_class = payload.get("model_class")
    if model_class == "LightGBMRanker":
        if int(booster.num_model_per_iteration()) != 1:
            raise ValueError("LightGBM ranking checkpoint must contain a single-score ranker.")
        if payload.get("class_mapping") is not None:
            raise ValueError("LightGBM ranker checkpoint must not define a class mapping.")
    elif model_class == "LightGBMClassifier":
        if int(booster.num_model_per_iteration()) != 4:
            raise ValueError("LightGBM classifier checkpoint must contain four-class multiclass model.")
        if payload.get("class_mapping") != {"0": "MTB_0", "1": "MTB_1", "2": "MTB_2", "3": "MTB_3"}:
            raise ValueError("LightGBM checkpoint class mapping is invalid.")
    else:
        raise ValueError(f"Unsupported LightGBM model class {model_class!r}.")
    for key in ("training_config", "data_config", "metadata"):
        if not isinstance(payload.get(key), dict):
            raise ValueError(f"LightGBM checkpoint {key} must be a dictionary.")
    payload["_booster"] = booster
    return payload
