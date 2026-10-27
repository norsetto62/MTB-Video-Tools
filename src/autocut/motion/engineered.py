"""Seven-input motion representation used by the experimental MTB P/U model.

The four ratio signals retain their original formulas and are compressed before
scaling. Global horizontal/vertical flow and center-cell residual are smoothed
on the continuous sequence, before overlapping windows are built.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

from autocut.motion.features import FEATURE_NAMES

ENGINEERED_FEATURE_NAMES = (
    "vertical_flow_gradient",
    "lateral_flow_asymmetry",
    "expansion_index",
    "roll_index",
)
MODEL_FEATURE_NAMES = ENGINEERED_FEATURE_NAMES + ("u", "v", "center_residual")

SPIKE_COMPRESSION_THRESHOLD = 5.0
SPIKE_COMPRESSION_SLOPE = 0.2
SAVGOL_WINDOW_LENGTH = 9
SAVGOL_POLYORDER = 2

_EPSILON = 1e-6
_GRID_MAGNITUDE_INDICES = np.asarray(
    [FEATURE_NAMES.index(f"grid_{row}_{col}_magnitude")
     for row in range(3) for col in range(3)],
    dtype=np.int64,
)
_FLOW_U = FEATURE_NAMES.index("flow_u")
_FLOW_V = FEATURE_NAMES.index("flow_v")
_CENTER_U = FEATURE_NAMES.index("grid_1_1_u")
_CENTER_V = FEATURE_NAMES.index("grid_1_1_v")


def _validate_canonical_sequence(X: np.ndarray) -> np.ndarray:
    try:
        values = np.asarray(X, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("X must be convertible to a float32 array.") from exc
    if values.ndim != 2 or values.shape[1] != len(FEATURE_NAMES):
        raise ValueError(
            f"X must have shape (T, {len(FEATURE_NAMES)}); got {values.shape}."
        )
    if values.shape[0] == 0 or not np.isfinite(values).all():
        raise ValueError("X must be non-empty and contain only finite values.")
    return values


def compress_engineered_spikes(
    values: np.ndarray,
    *,
    threshold: float = SPIKE_COMPRESSION_THRESHOLD,
    slope: float = SPIKE_COMPRESSION_SLOPE,
) -> np.ndarray:
    """Continuously compress signed values outside +/- threshold."""
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("values must contain only finite values.")
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be finite and positive.")
    if not np.isfinite(slope) or not 0 < slope <= 1:
        raise ValueError("slope must be finite and in (0, 1].")
    magnitude = np.abs(array)
    compressed = np.where(
        magnitude <= threshold,
        magnitude,
        threshold + slope * (magnitude - threshold),
    )
    return (np.sign(array) * compressed).astype(np.float32)


def engineer_motion_sequence(
    X: np.ndarray,
    *,
    smooth_window_length: int = SAVGOL_WINDOW_LENGTH,
    smooth_polyorder: int = SAVGOL_POLYORDER,
) -> np.ndarray:
    """Transform a continuous canonical sequence into seven model signals.

    Filtering is deliberately performed here, before windowing, so adjacent
    overlapping windows see the same filtered value at shared timestamps.
    """
    values = _validate_canonical_sequence(X)
    if isinstance(smooth_window_length, bool) or not isinstance(smooth_window_length, int):
        raise ValueError("smooth_window_length must be an integer.")
    if isinstance(smooth_polyorder, bool) or not isinstance(smooth_polyorder, int):
        raise ValueError("smooth_polyorder must be an integer.")
    if smooth_window_length < 3 or smooth_window_length % 2 == 0:
        raise ValueError("smooth_window_length must be an odd integer >= 3.")
    if smooth_polyorder < 0 or smooth_polyorder >= smooth_window_length:
        raise ValueError("smooth_polyorder must be >= 0 and less than the window length.")

    grid = values[:, _GRID_MAGNITUDE_INDICES].astype(np.float64, copy=False)
    top, bottom = grid[:, 0:3], grid[:, 6:9]
    center = grid[:, 4]
    center_safe = np.maximum(center, _EPSILON)

    # Keep these four established formulas unchanged.
    vertical = (np.mean(top, axis=-1) - np.mean(bottom, axis=-1)) / center_safe
    lateral = (np.mean(grid[:, 2::3], axis=-1) - np.mean(grid[:, 0::3], axis=-1)) / center_safe
    periphery = np.mean(
        grid[:, [0, 1, 2, 3, 5, 6, 7, 8]], axis=-1
    )
    expansion = (periphery - center) / center_safe
    roll = ((grid[:, 0] + grid[:, 8]) - (grid[:, 2] + grid[:, 6])) / center_safe
    engineered = compress_engineered_spikes(
        np.stack((vertical, lateral, expansion, roll), axis=-1)
    )

    center_residual = np.hypot(
        values[:, _CENTER_U] - values[:, _FLOW_U],
        values[:, _CENTER_V] - values[:, _FLOW_V],
    )
    raw_signals = np.stack(
        (values[:, _FLOW_U], values[:, _FLOW_V], center_residual), axis=-1
    )

    # Savitzky-Golay needs an odd window no longer than the sequence. For short
    # sequences, use the largest valid odd window; leave <3 samples unchanged.
    effective_window = min(smooth_window_length, len(values) if len(values) % 2 else len(values) - 1)
    if effective_window >= 3:
        effective_polyorder = min(smooth_polyorder, effective_window - 1)
        smoothed = savgol_filter(
            raw_signals,
            window_length=effective_window,
            polyorder=effective_polyorder,
            axis=0,
            mode="interp",
        )
    else:
        smoothed = raw_signals

    result = np.concatenate((engineered, smoothed.astype(np.float32)), axis=-1).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Transformed motion sequence contains non-finite values.")
    return result


def engineer_motion_features(X: np.ndarray) -> np.ndarray:
    """Backward-compatible four-signal transform for legacy checkpoints."""
    try:
        values = np.asarray(X, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("X must be convertible to a float32 array.") from exc
    if values.ndim not in (2, 3) or values.shape[-1] != len(FEATURE_NAMES):
        raise ValueError(
            f"X must have shape (T, {len(FEATURE_NAMES)}) or "
            f"(N, T, {len(FEATURE_NAMES)}); got {values.shape}."
        )
    if any(size == 0 for size in values.shape) or not np.isfinite(values).all():
        raise ValueError("X must be non-empty and contain only finite values.")
    grid = values[..., _GRID_MAGNITUDE_INDICES].astype(np.float64, copy=False)
    top, bottom = grid[..., 0:3], grid[..., 6:9]
    center = grid[..., 4]
    center_safe = np.maximum(center, _EPSILON)
    vertical = (np.mean(top, axis=-1) - np.mean(bottom, axis=-1)) / center_safe
    lateral = (np.mean(grid[..., 2::3], axis=-1) - np.mean(grid[..., 0::3], axis=-1)) / center_safe
    periphery = np.mean(grid[..., [0, 1, 2, 3, 5, 6, 7, 8]], axis=-1)
    expansion = (periphery - center) / center_safe
    roll = ((grid[..., 0] + grid[..., 8]) - (grid[..., 2] + grid[..., 6])) / center_safe
    return np.stack((vertical, lateral, expansion, roll), axis=-1).astype(np.float32)


# ---------------------------------------------------------------------------
# LightGBM representation: eight experimental signals, without smoothing.
# ---------------------------------------------------------------------------

LIGHTGBM_FEATURE_NAMES = (
    "vertical_flow_gradient",
    "lateral_flow_asymmetry",
    "expansion_index",
    "roll_index",
    "directional_turbulence",
    "global_flow_magnitude",
    "center_residual",
    "tunnel_vision",
)

_GRID_U_INDICES = np.asarray(
    [FEATURE_NAMES.index(f"grid_{row}_{col}_u")
     for row in range(3) for col in range(3)],
    dtype=np.int64,
)
_GRID_V_INDICES = np.asarray(
    [FEATURE_NAMES.index(f"grid_{row}_{col}_v")
     for row in range(3) for col in range(3)],
    dtype=np.int64,
)


def engineer_lightgbm_sequence(X: np.ndarray) -> np.ndarray:
    """Compute the eight experimental signals on a continuous canonical sequence.

    Unlike the legacy neural-network representation above, this transform
    uses the experiment formulas directly: no Savitzky-Golay smoothing and no
    spike compression. Apply it to the full sequence before creating windows.
    """
    values = _validate_canonical_sequence(X).astype(np.float64, copy=False)
    grid_u = values[:, _GRID_U_INDICES].reshape(-1, 3, 3)
    grid_v = values[:, _GRID_V_INDICES].reshape(-1, 3, 3)
    magnitudes = np.hypot(grid_u, grid_v)
    epsilon = 1e-5

    center_magnitude = magnitudes[:, 1, 1]
    center_safe = center_magnitude + epsilon
    top = np.mean(magnitudes[:, 0, :], axis=1)
    bottom = np.mean(magnitudes[:, 2, :], axis=1)
    left = np.mean(magnitudes[:, :, 0], axis=1)
    right = np.mean(magnitudes[:, :, 2], axis=1)
    peripheral = np.mean(
        magnitudes[:, [0, 0, 0, 1, 1, 2, 2, 2],
                   [0, 1, 2, 0, 2, 0, 1, 2]],
        axis=1,
    )

    flow_u = values[:, _FLOW_U]
    flow_v = values[:, _FLOW_V]
    residuals = np.hypot(
        grid_u - flow_u[:, None, None],
        grid_v - flow_v[:, None, None],
    )
    center_residual = residuals[:, 1, 1]
    edge_residual = np.mean(
        residuals[:, [0, 0, 0, 1, 1, 2, 2, 2],
                  [0, 1, 2, 0, 2, 0, 1, 2]],
        axis=1,
    )
    angles = np.arctan2(grid_v, grid_u)
    mean_cos = np.mean(np.cos(angles), axis=(1, 2))
    mean_sin = np.mean(np.sin(angles), axis=(1, 2))
    directional_turbulence = 1.0 - np.hypot(mean_cos, mean_sin)

    result = np.stack(
        (
            (top - bottom) / center_safe,
            (right - left) / center_safe,
            (peripheral - center_magnitude) / center_safe,
            (
                magnitudes[:, 0, 0] + magnitudes[:, 2, 2]
                - magnitudes[:, 0, 2] - magnitudes[:, 2, 0]
            ) / center_safe,
            directional_turbulence,
            np.hypot(flow_u, flow_v),
            center_residual,
            edge_residual / (center_residual + epsilon),
        ),
        axis=1,
    )
    if not np.isfinite(result).all():
        raise ValueError("LightGBM engineered signals contain non-finite values.")
    return result.astype(np.float32, copy=False)
