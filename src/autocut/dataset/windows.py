"""Generate fixed-size temporal windows from feature sequences."""
from __future__ import annotations

import numpy as np

from ..data_models import FeatureSequence, FeatureWindows


def build_windows(
    sequence: FeatureSequence, *, window: float, stride: float,
) -> FeatureWindows:
    """Window a canonical feature sequence using the standard temporal grid."""
    if len(sequence.timestamps) < 2:
        raise ValueError(
            "at least two feature timestamps are required "
            "to determine the flow sampling interval."
        )
    return build_windows_from_features(
        sequence.features, sequence.timestamps, window=window, stride=stride
    )


def build_windows_from_features(
    features: np.ndarray,
    timestamps: np.ndarray,
    *,
    window: float,
    stride: float,
) -> FeatureWindows:
    """Window any continuous (T,F) feature array before/after model transforms.

    Filtering and other sequence-level transforms should be applied to the
    entire sequence before calling this function, not independently per window.
    """
    if not np.isfinite(window) or window <= 0:
        raise ValueError("window must be greater than zero.")
    if not np.isfinite(stride) or stride <= 0:
        raise ValueError("stride must be greater than zero.")
    try:
        features = np.asarray(features, dtype=np.float32)
        timestamps = np.asarray(timestamps, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("features and timestamps must be numeric arrays.") from exc
    if features.ndim != 2 or features.shape[0] < 2 or features.shape[1] < 1:
        raise ValueError("features must have shape (T,F) with T >= 2 and F >= 1.")
    if timestamps.ndim != 1 or len(timestamps) != len(features):
        raise ValueError("timestamps must be 1D and match the feature sequence length.")
    if not np.isfinite(features).all() or not np.isfinite(timestamps).all():
        raise ValueError("features and timestamps must contain only finite values.")
    if not np.all(np.diff(timestamps) > 0):
        raise ValueError("timestamps must be strictly increasing.")

    dt = float(np.median(np.diff(timestamps)))
    if dt <= 0:
        raise ValueError("invalid feature sampling interval.")
    window_samples_float, stride_samples_float = window / dt, stride / dt
    samples_per_window, samples_per_stride = int(round(window_samples_float)), int(round(stride_samples_float))
    if not np.isclose(window_samples_float, samples_per_window, rtol=0.0, atol=1e-6):
        raise ValueError(f"window={window:.6f}s is not an integer multiple of feature dt={dt:.6f}s.")
    if not np.isclose(stride_samples_float, samples_per_stride, rtol=0.0, atol=1e-6):
        raise ValueError(f"stride={stride:.6f}s is not an integer multiple of feature dt={dt:.6f}s.")
    if samples_per_window < 2:
        raise ValueError(f"window={window:.3f}s is too short for feature dt={dt:.6f}s.")

    flow_start = float(timestamps[0] - dt)
    represented_duration = float(timestamps[-1] - flow_start)
    max_start_offset = represented_duration - window
    if max_start_offset < -1e-9:
        raise ValueError(f"feature duration={represented_duration:.6f}s is shorter than window={window:.6f}s.")
    n_windows = int(np.floor((max_start_offset + 1e-9) / stride)) + 1
    windows, window_timestamps = [], []
    for window_number in range(n_windows):
        index = window_number * samples_per_stride
        end_index = index + samples_per_window
        if end_index > len(features):
            break
        windows.append(features[index:end_index])
        start = flow_start + index * dt
        window_timestamps.append((start, start + window))
    if not windows:
        raise ValueError("feature sequence is too short to produce a complete window.")
    return FeatureWindows(
        X=np.stack(windows).astype(np.float32, copy=False),
        timestamps=np.asarray(window_timestamps, dtype=np.float64),
    )
