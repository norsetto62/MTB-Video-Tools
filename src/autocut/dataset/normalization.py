"""Per-video feature normalization used by the LightGBM ranking path."""
from __future__ import annotations

import numpy as np


DEFAULT_FEATURE_NORMALIZATION = "per_video_zscore_all_windows_ddof0"
SUPPORTED_FEATURE_NORMALIZATIONS = (
    DEFAULT_FEATURE_NORMALIZATION,
    "per_video_median_iqr_all_windows",
)


def normalize_per_video_features(
    features: np.ndarray,
    *,
    method: str = DEFAULT_FEATURE_NORMALIZATION,
) -> np.ndarray:
    """Normalize each feature over all windows from one video.

    per_video_median_iqr_all_windows centers by the median and scales by
    IQR (Q75-Q25), making the transformation less sensitive to heavy tails.
    Constant or numerically near-constant columns use scale 1. Statistics are
    computed before annotation filtering, identically for training/inference.
    """
    try:
        values = np.asarray(features, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("features must be convertible to a float32 matrix.") from exc
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("features must have non-empty shape (N, F).")
    if not np.isfinite(values).all():
        raise ValueError("features must contain only finite values.")
    if method not in SUPPORTED_FEATURE_NORMALIZATIONS:
        raise ValueError(
            f"Unsupported feature normalization {method!r}; "
            f"choose one of {SUPPORTED_FEATURE_NORMALIZATIONS}."
        )

    values64 = values.astype(np.float64, copy=False)
    if method == "per_video_median_iqr_all_windows":
        center = np.median(values64, axis=0)
        q25, q75 = np.percentile(values64, [25, 75], axis=0)
        scale = q75 - q25
    else:
        center = np.mean(values64, axis=0)
        scale = np.std(values64, axis=0, ddof=0)
    safe_scale = np.where(scale > 1e-8, scale, 1.0)
    normalized = (values64 - center) / safe_scale
    if not np.isfinite(normalized).all():
        raise ValueError("Per-video feature normalization produced non-finite values.")
    return normalized.astype(np.float32, copy=False)
