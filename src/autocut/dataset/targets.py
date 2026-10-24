"""Build pure annotation targets for positive-unlabeled training.

A window is retained only when every annotation interval overlapping it has
the same MTB score. Conflicting windows are discarded. The returned target
is the original annotation score (0, 1, 2, or 3): only score 3 is a confirmed
positive; scores 0, 1, and 2 are unlabeled, not confirmed negatives.
"""

import numpy as np

from ..data_models import FeatureWindows, Annotation


def _annotation_overlap(
    timestamp: tuple[float, float],
    annotation: Annotation,
) -> float:
    """Return overlap duration in seconds."""
    return max(
        0.0,
        min(timestamp[1], annotation.end)
        - max(timestamp[0], annotation.start),
    )


def validate_targets(
    X: np.ndarray,
    timestamps: np.ndarray,
    y: np.ndarray,
) -> list[str]:
    """Validate alignment and pure annotation labels in the range 0..3."""
    errors: list[str] = []
    X_arr = np.asarray(X)
    ts_arr = np.asarray(timestamps)
    y_arr = np.asarray(y)

    if y_arr.ndim != 1:
        errors.append(f"Target vector y must be 1D, got shape {y_arr.shape}.")
        return errors

    n_windows = len(X_arr)
    if len(y_arr) != n_windows:
        errors.append(
            f"Dimension mismatch: target count ({len(y_arr)}) does not match "
            f"window count ({n_windows})."
        )
    if len(ts_arr) != n_windows:
        errors.append(
            f"Dimension mismatch: timestamp count ({len(ts_arr)}) does not match "
            f"window count ({n_windows})."
        )
    if not np.isfinite(y_arr).all():
        errors.append("Non-finite values (NaN/Inf) detected in target labels y.")

    finite_values = y_arr[np.isfinite(y_arr)]
    invalid_values = np.unique(
        finite_values[~np.isin(finite_values, [0.0, 1.0, 2.0, 3.0])]
    )
    if invalid_values.size:
        errors.append(
            "Pure annotation targets must be integer scores 0, 1, 2, or 3; "
            f"invalid values: {invalid_values.tolist()}."
        )
    return errors


def build_targets(
    windows: FeatureWindows,
    annotations: list[Annotation],
) -> tuple[np.ndarray, np.ndarray]:
    """Keep windows whose overlapping annotations all agree on one score.

    A window crossing a class boundary is discarded even when one class
    covers more of the window. No-overlap windows are discarded as well.
    Returned labels preserve annotation scores 0..3 for the PU trainer.
    """
    X_training: list[np.ndarray] = []
    targets: list[float] = []
    timestamps = windows.timestamps

    for window_number in range(windows.X.shape[0]):
        timestamp = timestamps[window_number]
        overlapping_scores = {
            annotation.score
            for annotation in annotations
            if _annotation_overlap(timestamp, annotation) > 0.0
        }
        if len(overlapping_scores) != 1:
            continue

        X_training.append(windows.X[window_number])
        targets.append(float(next(iter(overlapping_scores))))

    if not X_training:
        return (
            np.empty((0, windows.X.shape[1], windows.X.shape[2]), dtype=np.float32),
            np.empty(0, dtype=np.float32),
        )
    return (
        np.stack(X_training).astype(np.float32, copy=False),
        np.asarray(targets, dtype=np.float32),
    )
