"""Inference helpers for LightGBM tabular window models."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from autocut.data_models import FeatureWindows

from autocut.ml.lightgbm.checkpoint import load_lightgbm_checkpoint
from autocut.ml.lightgbm.model import LightGBMClassifier, LightGBMRanker
from autocut.dataset.windows import LIGHTGBM_TABULAR_FEATURE_NAMES
from autocut.dataset.normalization import (
    LIGHTGBM_FEATURE_NORMALIZATION,
    normalize_per_video_features,
)


@dataclass(frozen=True)
class LightGBMInferenceResult:
    """Four-class probabilities and predictions aligned to window timestamps."""

    timestamps: np.ndarray
    probabilities: np.ndarray  # (N, 4), columns are class 0, 1, 2, 3
    predictions: np.ndarray    # (N,), argmax class IDs 0..3


@dataclass(frozen=True)
class LightGBMRankingInferenceResult:
    """Raw ranking scores aligned with aggregated-window timestamps."""

    timestamps: np.ndarray
    scores: np.ndarray


def infer_lightgbm_ranking_windows(
    checkpoint: str | Path | Mapping[str, Any],
    windows: FeatureWindows,
) -> LightGBMRankingInferenceResult:
    """Score aggregated windows with a saved LambdaRank checkpoint.

    Scores are relative within a query/video. Do not interpret them as class
    probabilities or compare absolute scores across unrelated videos.
    """
    if not isinstance(windows, FeatureWindows):
        raise TypeError("windows must be a FeatureWindows instance.")
    if isinstance(checkpoint, (str, Path)):
        payload = load_lightgbm_checkpoint(checkpoint)
    elif isinstance(checkpoint, Mapping):
        payload = dict(checkpoint)
        if payload.get("checkpoint_version") != "lightgbm-1.0":
            raise ValueError("Unsupported or invalid LightGBM checkpoint.")
        names = payload.get("feature_names")
        if names != list(LIGHTGBM_TABULAR_FEATURE_NAMES):
            raise ValueError("Checkpoint feature names do not match the canonical 74-column schema.")
        booster = payload.get("_booster")
        if booster is None:
            import lightgbm as lgb
            model_string = payload.get("model_string")
            if not isinstance(model_string, str) or not model_string:
                raise ValueError("LightGBM checkpoint model_string is missing.")
            booster = lgb.Booster(model_str=model_string)
            payload["_booster"] = booster
    else:
        raise TypeError("checkpoint must be a path or a checkpoint mapping.")
    if payload.get("model_class") != "LightGBMRanker":
        raise ValueError("Checkpoint is not a LightGBMRanker; use infer_lightgbm_windows for legacy classifiers.")
    if payload.get("feature_names") != list(LIGHTGBM_TABULAR_FEATURE_NAMES):
        raise ValueError("Checkpoint feature names do not match the canonical 74-column schema.")
    X = np.asarray(windows.X, dtype=np.float32)
    timestamps = np.asarray(windows.timestamps, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] != len(LIGHTGBM_TABULAR_FEATURE_NAMES):
        raise ValueError(f"windows.X must have shape (N, {len(LIGHTGBM_TABULAR_FEATURE_NAMES)}).")
    if not np.isfinite(X).all():
        raise ValueError("windows.X must contain only finite values.")
    if timestamps.shape != (len(X), 2) or not np.isfinite(timestamps).all():
        raise ValueError("windows.timestamps must be a finite (N,2) array aligned with X.")
    if np.any(timestamps[:, 1] <= timestamps[:, 0]):
        raise ValueError("Every window end timestamp must be after its start.")
    if len(timestamps) > 1 and np.any(np.diff(timestamps[:, 0]) <= 0):
        raise ValueError("Window start timestamps must be strictly increasing.")
    # New ranker checkpoints explicitly opt into per-video normalization.
    # Older checkpoints without this metadata retain their original behavior.
    data_config = payload.get("data_config") or {}
    normalization_method = data_config.get("feature_normalization")
    if normalization_method is not None:
        X = normalize_per_video_features(X, method=normalization_method)
    model = LightGBMRanker(payload["_booster"], feature_names=LIGHTGBM_TABULAR_FEATURE_NAMES)
    scores = model.predict(X)
    return LightGBMRankingInferenceResult(timestamps=timestamps.copy(), scores=scores)


def infer_lightgbm_windows(
    checkpoint: str | Path | Mapping[str, Any],
    windows: FeatureWindows,
) -> LightGBMInferenceResult:
    """Predict annotation classes from tabular 48-column LightGBM windows.

    This API deliberately accepts aggregated rows only. Continuous sequence
    engineering, temporal windowing and aggregation must use the same canonical
    functions as training before calling it.
    """
    if not isinstance(windows, FeatureWindows):
        raise TypeError("windows must be a FeatureWindows instance.")
    if isinstance(checkpoint, (str, Path)):
        payload = load_lightgbm_checkpoint(checkpoint)
    elif isinstance(checkpoint, Mapping):
        payload = dict(checkpoint)
        if payload.get("checkpoint_version") != "lightgbm-1.0":
            raise ValueError("Unsupported or invalid LightGBM checkpoint.")
        names = payload.get("feature_names")
        if names != list(LIGHTGBM_TABULAR_FEATURE_NAMES):
            raise ValueError("Checkpoint feature names do not match the canonical 74-column schema.")
        booster = payload.get("_booster")
        if booster is None:
            import lightgbm as lgb
            model_string = payload.get("model_string")
            if not isinstance(model_string, str) or not model_string:
                raise ValueError("LightGBM checkpoint model_string is missing.")
            booster = lgb.Booster(model_str=model_string)
            payload["_booster"] = booster
    else:
        raise TypeError("checkpoint must be a path or checkpoint mapping.")
    if payload.get("feature_names") != list(LIGHTGBM_TABULAR_FEATURE_NAMES):
        raise ValueError("Checkpoint feature names do not match the canonical 74-column schema.")
    X = np.asarray(windows.X, dtype=np.float32)
    timestamps = np.asarray(windows.timestamps, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] != len(LIGHTGBM_TABULAR_FEATURE_NAMES):
        raise ValueError(f"windows.X must have shape (N, {len(LIGHTGBM_TABULAR_FEATURE_NAMES)}).")
    if not np.isfinite(X).all():
        raise ValueError("windows.X must contain only finite values.")
    if timestamps.shape != (len(X), 2) or not np.isfinite(timestamps).all():
        raise ValueError("windows.timestamps must be a finite (N,2) array aligned with X.")
    if np.any(timestamps[:, 1] <= timestamps[:, 0]):
        raise ValueError("Every window end timestamp must be after its start.")
    if len(timestamps) > 1 and np.any(np.diff(timestamps[:, 0]) <= 0):
        raise ValueError("Window start timestamps must be strictly increasing.")
    if payload.get("model_class") == "LightGBMRanker":
        raise ValueError("Checkpoint is a LightGBMRanker; use infer_lightgbm_ranking_windows.")
    model = LightGBMClassifier(payload["_booster"], feature_names=LIGHTGBM_TABULAR_FEATURE_NAMES)
    probabilities = model.predict_proba(X)
    predictions = np.argmax(probabilities, axis=1).astype(np.int64)
    return LightGBMInferenceResult(
        timestamps=timestamps.copy(),
        probabilities=probabilities,
        predictions=predictions,
    )
