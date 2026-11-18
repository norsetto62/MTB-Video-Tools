"""Engineered optical-flow signals used by the LightGBM ranking model."""

from __future__ import annotations

import numpy as np

from autocut.motion.features import FEATURE_NAMES


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
_FLOW_U = FEATURE_NAMES.index("flow_u")
_FLOW_V = FEATURE_NAMES.index("flow_v")


def engineer_lightgbm_sequence(X: np.ndarray) -> np.ndarray:
    """Compute eight signals on a continuous canonical sequence before windowing.

    The representation uses directional flow, global flow magnitude, center
    residual, and peripheral residual relative to center residual. It does not
    apply Savitzky-Golay smoothing or spike compression.
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
    edge_residual = np.mean(residuals[:, [0, 0], [0, 2]], axis=1)
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
