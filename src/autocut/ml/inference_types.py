"""Framework-independent result types shared by inference consumers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class InferenceResult:
    """Predictions aligned with input windows.

    Attributes:
        timestamps: Window start/end times in seconds, shape (N, 2).
        probabilities: Probability of membership in P, shape (N,).
    """

    timestamps: np.ndarray
    probabilities: np.ndarray
