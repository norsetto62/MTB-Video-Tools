"""Window-level inference for trained AutoCut MTB-interest models.

Inference returns one P probability for each input window, along with timestamps. It does not classify windows with argmax,
apply candidate-selection thresholds, or merge/select segments.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from autocut.data_models import FeatureSequence, FeatureWindows
from autocut.ml.checkpoint import load_checkpoint, restore_model, validate_checkpoint
from autocut.ml.scaling import FeatureScaler
from autocut.motion.features import FEATURE_NAMES
from autocut.motion.engineered import ENGINEERED_FEATURE_NAMES, MODEL_FEATURE_NAMES, engineer_motion_features, engineer_motion_sequence
from autocut.dataset.windows import build_windows_from_features


@dataclass(frozen=True)
class InferenceResult:
    """Predictions aligned with the input windows.

    Attributes:
        timestamps: Window start/end times in seconds, shape (N, 2).
        probabilities: Probability of membership in P, shape (N,).
    """

    timestamps: np.ndarray
    probabilities: np.ndarray


def infer_windows(
    checkpoint: str | Path | Mapping[str, Any],
    windows: FeatureWindows,
    *,
    batch_size: int = 256,
    device: str | torch.device = "cpu",
) -> InferenceResult:
    """Predict P probabilities for precomputed feature windows.

    Args:
        checkpoint: Path to a saved AutoCut checkpoint, or an already-loaded
            checkpoint payload.
        windows: FeatureWindows produced by AutoCut's window builder. Input
            features must use the canonical FEATURE_NAMES order; compact-feature
            checkpoints derive their four engineered signals from these vectors.
        batch_size: Maximum number of windows evaluated in one model batch.
        device: PyTorch device on which to run inference.

    Returns:
        InferenceResult with timestamps and float32 P probabilities, shape (N,).

    Notes:
        If the checkpoint contains a feature mask, it is applied to the full
        canonical feature vectors before scaling. The scaler saved with the
        checkpoint is reused; inference never fits new scaling statistics.
        Checkpoints without scaler state are evaluated on the supplied feature
        values unchanged.

        This function deliberately does not compute argmax classes or perform
        candidate selection, thresholding, ranking, or temporal merging.
    """
    if not isinstance(windows, FeatureWindows):
        raise TypeError("windows must be a FeatureWindows instance.")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")

    if isinstance(checkpoint, (str, Path)):
        payload = load_checkpoint(checkpoint, map_location="cpu")
    elif isinstance(checkpoint, Mapping):
        payload = validate_checkpoint(dict(checkpoint))
    else:
        raise TypeError("checkpoint must be a path or a checkpoint mapping.")

    try:
        X = np.asarray(windows.X, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("windows.X must be convertible to a float32 array.") from exc

    if X.ndim != 3:
        raise ValueError("windows.X must have shape (N, T, F).")
    if any(size == 0 for size in X.shape):
        raise ValueError("windows.X must not have empty dimensions.")
    if not np.isfinite(X).all():
        raise ValueError("windows.X must contain only finite values.")
    feature_names = payload.get("feature_names")
    expected_input_features = (
        len(MODEL_FEATURE_NAMES)
        if feature_names is not None and list(feature_names) == list(MODEL_FEATURE_NAMES)
        else len(FEATURE_NAMES)
    )
    if X.shape[2] != expected_input_features:
        raise ValueError(
            f"windows.X must contain {expected_input_features} features for this checkpoint "
            f"(canonical features for canonical checkpoints); got {X.shape[2]}."
        )

    try:
        timestamps = np.asarray(windows.timestamps, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("windows.timestamps must be numeric.") from exc

    if timestamps.shape != (X.shape[0], 2):
        raise ValueError(
            "windows.timestamps must have shape (N, 2), matching windows.X."
        )
    if not np.isfinite(timestamps).all():
        raise ValueError("windows.timestamps must contain only finite values.")
    if np.any(timestamps[:, 1] <= timestamps[:, 0]):
        raise ValueError("Every window end timestamp must be after its start.")
    if len(timestamps) > 1 and np.any(np.diff(timestamps[:, 0]) <= 0):
        raise ValueError("Window start timestamps must be strictly increasing.")

    data_config = payload.get("data_config", {})
    flow_fps = float(data_config["flow_fps"])
    window_duration = float(data_config["window"])
    expected_steps_float = flow_fps * window_duration
    expected_steps = int(round(expected_steps_float))
    if not np.isclose(expected_steps_float, expected_steps, rtol=0.0, atol=1e-6):
        raise ValueError(
            "Checkpoint window duration multiplied by flow_fps must be "
            "an integer number of feature steps."
        )
    if X.shape[1] != expected_steps:
        raise ValueError(
            f"windows.X must contain {expected_steps} time steps per window "
            f"for this checkpoint; got {X.shape[1]}."
        )
    if not np.allclose(
        timestamps[:, 1] - timestamps[:, 0],
        window_duration,
        rtol=0.0,
        atol=1e-5,
    ):
        raise ValueError(
            "Window timestamp durations must match the checkpoint's "
            f"{window_duration:g}-second window."
        )

    compact_features = feature_names is not None and list(feature_names) == list(ENGINEERED_FEATURE_NAMES)
    seven_features = feature_names is not None and list(feature_names) == list(MODEL_FEATURE_NAMES)
    if compact_features:
        X = engineer_motion_features(X)
    elif not seven_features and feature_names is not None and list(feature_names) != list(FEATURE_NAMES):
        raise ValueError(
            "Checkpoint feature_names must match canonical features, the legacy "
            "four engineered signals, or the seven-signal sequence transform."
        )

    feature_mask = payload.get("feature_mask")
    if feature_mask is not None:
        mask = np.asarray(feature_mask, dtype=bool)
        if mask.shape != (X.shape[2],):
            raise ValueError(
                "Checkpoint feature_mask must match the supplied feature count."
            )
        X = X[..., mask]

    model = restore_model(payload, device=device)
    if X.shape[2] != model.feature_dim:
        raise ValueError(
            f"Selected input has {X.shape[2]} features, but the checkpoint "
            f"model expects {model.feature_dim}."
        )

    scaler_state = payload.get("scaler_state")
    scaler = (
        FeatureScaler.from_state_dict(scaler_state)
        if scaler_state is not None
        else None
    )
    if scaler is not None:
        if scaler.enabled and (
            scaler.mean_ is None
            or scaler.scale_ is None
            or scaler.mean_.size != model.feature_dim
        ):
            raise ValueError(
                "Checkpoint scaler statistics do not match the model's "
                "input feature count."
            )
        X = scaler.transform(X)

    model.eval()
    prediction_batches: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, X.shape[0], batch_size):
            batch = torch.from_numpy(X[start:start + batch_size]).to(device)
            logits = model(batch)
            if logits.ndim != 2 or logits.shape[1] != 1:
                raise RuntimeError("P/U model must return exactly one logit per window.")
            probabilities = torch.sigmoid(logits[:, 0])
            prediction_batches.append(probabilities.cpu().numpy())

    probabilities = np.concatenate(prediction_batches, axis=0).astype(
        np.float32,
        copy=False,
    )
    if probabilities.shape != (X.shape[0],):
        raise RuntimeError("Model output must contain exactly one P probability per window.")
    if not np.isfinite(probabilities).all():
        raise RuntimeError("Inference produced non-finite P probabilities.")

    return InferenceResult(
        timestamps=timestamps.copy(),
        probabilities=probabilities,
    )



def infer_sequence(
    checkpoint: str | Path | Mapping[str, Any],
    sequence: FeatureSequence,
    *,
    window: float | None = None,
    stride: float | None = None,
    batch_size: int = 256,
    device: str | torch.device = "cpu",
) -> InferenceResult:
    """Transform a full canonical sequence, window it, and infer consistently.

    Use this entry point for seven-signal checkpoints because Savitzky-Golay
    filtering must run over the continuous sequence before overlapping windows
    are created. Legacy canonical/four-signal checkpoints remain supported.
    """
    if not isinstance(sequence, FeatureSequence):
        raise TypeError("sequence must be a FeatureSequence instance.")
    if isinstance(checkpoint, (str, Path)):
        payload = load_checkpoint(checkpoint, map_location="cpu")
    elif isinstance(checkpoint, Mapping):
        payload = validate_checkpoint(dict(checkpoint))
    else:
        raise TypeError("checkpoint must be a path or a checkpoint mapping.")
    data_config = payload.get("data_config", {})
    duration = float(data_config["window"] if window is None else window)
    step = float(data_config.get("stride", duration) if stride is None else stride)
    feature_names = payload.get("feature_names")
    if feature_names is not None and list(feature_names) == list(MODEL_FEATURE_NAMES):
        transformed = engineer_motion_sequence(sequence.features)
        windows = build_windows_from_features(
            transformed, sequence.timestamps, window=duration, stride=step
        )
    else:
        from autocut.dataset.windows import build_windows
        windows = build_windows(sequence, window=duration, stride=step)
    return infer_windows(
        payload, windows, batch_size=batch_size, device=device
    )
