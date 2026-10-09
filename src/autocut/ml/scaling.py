"""Feature standardization for AutoCut time-series feature arrays.

Statistics are fitted independently for each feature across all available
training examples and time steps. Callers are responsible for fitting only on
the training split, then reusing the fitted scaler for validation and inference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np


@dataclass
class FeatureScaler:
    """Standardize 2D (T, F) or 3D (N, T, F) arrays feature-wise.

    Args:
        enabled: If false, preserve input values (converted to float32) without
            fitting or applying standardization.
        eps: Positive threshold below which a feature's standard deviation is
            treated as zero and its scale is set to 1.0.
    """

    enabled: bool = True
    eps: float = 1e-8
    mean_: np.ndarray | None = field(default=None, init=False, repr=False)
    scale_: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be a bool.")
        if isinstance(self.eps, bool) or not isinstance(self.eps, (int, float)):
            raise ValueError("eps must be a finite positive number.")
        if not math.isfinite(float(self.eps)) or float(self.eps) <= 0.0:
            raise ValueError("eps must be a finite positive number.")
        self.eps = float(self.eps)

    @staticmethod
    def _as_feature_array(X: np.ndarray, *, operation: str) -> np.ndarray:
        """Convert and validate an input feature array."""
        try:
            X_arr = np.asarray(X, dtype=np.float32)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("X must be convertible to a float32 array.") from exc

        if X_arr.ndim not in (2, 3):
            raise ValueError(
                f"Expected a 2D (T, F) or 3D (N, T, F) array for {operation}; "
                f"got shape {X_arr.shape}."
            )
        if any(size == 0 for size in X_arr.shape):
            raise ValueError(
                f"X must not have empty dimensions for {operation}; "
                f"got shape {X_arr.shape}."
            )
        if not np.isfinite(X_arr).all():
            raise ValueError(f"X must contain only finite values for {operation}.")
        return X_arr

    def fit(self, X: np.ndarray) -> FeatureScaler:
        """Fit per-feature statistics using training data only.

        For 3D input, statistics are reduced over batch and temporal axes
        (0, 1). For 2D input, they are reduced over the temporal axis (0).
        """
        X_arr = self._as_feature_array(X, operation="fitting")

        if not self.enabled:
            self.mean_ = None
            self.scale_ = None
            return self

        reduce_axes = (0, 1) if X_arr.ndim == 3 else (0,)
        mean64 = np.mean(X_arr, axis=reduce_axes, dtype=np.float64)
        variance64 = np.var(X_arr, axis=reduce_axes, dtype=np.float64)
        std64 = np.sqrt(variance64)

        mean = mean64.astype(np.float32)
        std = std64.astype(np.float32)
        scale = np.where(std <= self.eps, 1.0, std).astype(np.float32)

        if not np.isfinite(mean).all() or not np.isfinite(scale).all():
            raise ValueError("Fitted feature statistics must be finite.")
        if np.any(scale <= 0.0):
            raise ValueError("Fitted feature scales must be positive.")

        self.mean_ = mean
        self.scale_ = scale
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply fitted standardization, returning a float32 array."""
        X_arr = self._as_feature_array(X, operation="transformation")

        if not self.enabled:
            return X_arr

        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("FeatureScaler must be fitted before calling transform().")
        if X_arr.shape[-1] != self.mean_.shape[0]:
            raise ValueError(
                f"X must have {self.mean_.shape[0]} features; "
                f"got {X_arr.shape[-1]}."
            )

        # Compute in float32 to match model inputs and stored statistics.
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            transformed = (X_arr - self.mean_) / self.scale_
        if not np.isfinite(transformed).all():
            raise ValueError("Standardized output contains non-finite values.")
        return transformed.astype(np.float32, copy=False)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit on X and return its standardized values."""
        return self.fit(X).transform(X)

    def state_dict(self) -> dict[str, bool | float | list[float] | None]:
        """Return a JSON-serializable representation of scaler state."""
        return {
            "enabled": self.enabled,
            "eps": self.eps,
            "mean": self.mean_.tolist() if self.mean_ is not None else None,
            "scale": self.scale_.tolist() if self.scale_ is not None else None,
        }

    @classmethod
    def from_state_dict(cls, state_dict: dict[str, Any]) -> FeatureScaler:
        """Restore a scaler from a state dictionary, validating its contents."""
        if not isinstance(state_dict, dict):
            raise ValueError("state_dict must be a dictionary.")

        enabled = state_dict.get("enabled", True)
        eps = state_dict.get("eps", 1e-8)
        scaler = cls(enabled=enabled, eps=eps)

        mean_value = state_dict.get("mean")
        scale_value = state_dict.get("scale")
        if (mean_value is None) != (scale_value is None):
            raise ValueError("Serialized mean and scale must either both be present or both be absent.")

        if mean_value is None:
            return scaler

        try:
            mean = np.asarray(mean_value, dtype=np.float32)
            scale = np.asarray(scale_value, dtype=np.float32)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Serialized mean and scale must be numeric arrays.") from exc

        if mean.ndim != 1 or scale.ndim != 1 or mean.size == 0 or scale.size == 0:
            raise ValueError("Serialized mean and scale must be non-empty 1D arrays.")
        if mean.shape != scale.shape:
            raise ValueError("Serialized mean and scale must have matching shapes.")
        if not np.isfinite(mean).all() or not np.isfinite(scale).all():
            raise ValueError("Serialized mean and scale must contain only finite values.")
        if np.any(scale <= 0.0):
            raise ValueError("Serialized scales must be positive.")

        scaler.mean_ = mean.copy()
        scaler.scale_ = scale.copy()
        return scaler
