"""Generate fixed-size temporal windows from feature sequences."""
from __future__ import annotations

import numpy as np

from ..motion.engineered import LIGHTGBM_FEATURE_NAMES
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


# ---------------------------------------------------------------------------
# LightGBM tabular representation
# ---------------------------------------------------------------------------

LIGHTGBM_WINDOW_STATISTICS = ("mean", "max", "std", "median", "q90", "min")
LIGHTGBM_TABULAR_FEATURE_NAMES = tuple(
    f"{signal}_{statistic}"
    for signal in LIGHTGBM_FEATURE_NAMES
    for statistic in LIGHTGBM_WINDOW_STATISTICS
)


def aggregate_lightgbm_windows(windows: FeatureWindows) -> FeatureWindows:
    """Aggregate (N,T,8) signal windows into an (N,48) tabular matrix.

    Columns are grouped by signal, with statistics in the fixed order
    mean, max, population standard deviation (ddof=0), median, 90th percentile,
    min. Window timestamps are preserved unchanged.
    """
    values = np.asarray(windows.X, dtype=np.float32)
    timestamps = np.asarray(windows.timestamps, dtype=np.float64)
    if values.ndim != 3 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("windows.X must have shape (N,T,F) with N,T >= 1.")
    if values.shape[2] != 8:
        raise ValueError(f"LightGBM aggregation requires 8 signals; got {values.shape[2]}.")
    if timestamps.shape != (values.shape[0], 2):
        raise ValueError("window timestamps must have shape (N,2).")
    if not np.isfinite(values).all() or not np.isfinite(timestamps).all():
        raise ValueError("windows and timestamps must contain only finite values.")

    stats = (
        np.mean(values, axis=1),
        np.max(values, axis=1),
        np.std(values, axis=1, ddof=0),
        np.median(values, axis=1),
        np.percentile(values, 90, axis=1),
        np.min(values, axis=1),
    )
    # (N,F,stats) -> (N,F*stats), keeping each signal's statistics adjacent.
    matrix = np.stack(stats, axis=2).reshape(values.shape[0], 8 * 6)
    return FeatureWindows(X=matrix.astype(np.float32, copy=False), timestamps=timestamps.copy())


def build_lightgbm_windows_from_features(
    features: np.ndarray,
    timestamps: np.ndarray,
    *,
    window: float,
    stride: float,
) -> FeatureWindows:
    """Build 48-column LightGBM rows from a continuous (T,8) signal sequence."""
    temporal = build_windows_from_features(
        features, timestamps, window=window, stride=stride
    )
    return aggregate_lightgbm_windows(temporal)
