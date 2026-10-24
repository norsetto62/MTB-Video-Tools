"""Compact engineered motion signals used by the experimental MTB baseline.

The four signals are derived from the canonical 3x3 grid-magnitude features.
They are computed from the original, unsmoothed feature sequence.
"""
from __future__ import annotations

import numpy as np

from autocut.motion.features import FEATURE_NAMES

ENGINEERED_FEATURE_NAMES = (
    "vertical_flow_gradient",
    "lateral_flow_asymmetry",
    "expansion_index",
    "roll_index",
)
_EPSILON = 1e-6
_GRID_MAGNITUDE_INDICES = np.asarray(
    [FEATURE_NAMES.index(f"grid_{row}_{col}_magnitude")
     for row in range(3) for col in range(3)],
    dtype=np.int64,
)


def engineer_motion_features(X: np.ndarray) -> np.ndarray:
    """Map canonical (..., 54) feature arrays to (..., 4) engineered signals."""
    try:
        values = np.asarray(X, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("X must be convertible to a float32 array.") from exc
    if values.ndim not in (2, 3) or values.shape[-1] != len(FEATURE_NAMES):
        raise ValueError(
            f"X must have shape (T, {len(FEATURE_NAMES)}) or "
            f"(N, T, {len(FEATURE_NAMES)}); got {values.shape}."
        )
    if any(size == 0 for size in values.shape):
        raise ValueError("X must not have empty dimensions.")
    if not np.isfinite(values).all():
        raise ValueError("X must contain only finite values.")

    grid = values[..., _GRID_MAGNITUDE_INDICES].astype(np.float64, copy=False)
    top, middle, bottom = grid[..., 0:3], grid[..., 3:6], grid[..., 6:9]
    center = grid[..., 4]
    center_safe = np.maximum(center, _EPSILON)

    vertical = (np.mean(top, axis=-1) - np.mean(bottom, axis=-1)) / center_safe
    lateral = (np.mean(grid[..., 2::3], axis=-1) - np.mean(grid[..., 0::3], axis=-1)) / center_safe
    periphery = np.mean(np.stack(
        (grid[..., 0], grid[..., 1], grid[..., 2], grid[..., 3],
         grid[..., 5], grid[..., 6], grid[..., 7], grid[..., 8]), axis=-1
    ), axis=-1)
    expansion = (periphery - center) / center_safe
    roll = ((grid[..., 0] + grid[..., 8]) - (grid[..., 2] + grid[..., 6])) / center_safe

    result = np.stack((vertical, lateral, expansion, roll), axis=-1).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Engineered motion features contain non-finite values.")
    return result
