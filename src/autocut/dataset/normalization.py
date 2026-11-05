"""Per-video feature normalization used by the LightGBM ranking path."""
from __future__ import annotations

import numpy as np


LIGHTGBM_FEATURE_NORMALIZATION = "per_video_zscore_all_windows_ddof0"


def normalize_per_video_features(features: np.ndarray) -> np.ndarray:
    """Z-score each feature across all windows from one video.

    Statistics are calculated over the complete video query before annotation
    filtering. Constant or numerically near-constant columns are centered but
    not divided by a tiny scale.
    """
    try:
        values = np.asarray(features, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("features must be convertible to a float32 matrix.") from exc
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("features must have non-empty shape (N, F).")
    if not np.isfinite(values).all():
        raise ValueError("features must contain only finite values.")

    values64 = values.astype(np.float64, copy=False)
    mean = np.mean(values64, axis=0)
    scale = np.std(values64, axis=0, ddof=0)
    safe_scale = np.where(scale > 1e-8, scale, 1.0)
    normalized = (values64 - mean) / safe_scale
    if not np.isfinite(normalized).all():
        raise ValueError("Per-video feature normalization produced non-finite values.")
    return normalized.astype(np.float32, copy=False)
