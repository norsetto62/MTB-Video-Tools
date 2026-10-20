"""Build binary training targets from MTB annotation durations.

For each window, overlapping MTB 3 intervals contribute to positive duration;
MTB 0 and 1 intervals contribute together to negative duration; MTB 2 is
ignored. A window is retained only when exactly one configured threshold is
met. Uncovered, insufficient-evidence, and mixed-evidence windows are dropped.

Window timestamps follow flow.py semantics and represent the interval covered
by the selected feature rows.
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
    """Validate interestingness targets and their alignment with feature windows.

    Targets are binary: 1.0 for positive and 0.0 for negative. Target-
    distribution summaries belong in the dataset audit rather than in this
    structural/value validator.

    Args:
        X: Windowed feature matrix of shape (N, T, F).
        timestamps: Window start/end times of shape (N, 2).
        y: Binary targets of shape (N,), containing only 0 or 1.

    Returns:
        List of validation error strings (empty if valid).
    """
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
        finite_values[~np.isin(finite_values, [0.0, 1.0])]
    )
    if invalid_values.size:
        errors.append(
            f"Binary targets must be 0 or 1; invalid values: "
            f"{invalid_values.tolist()}."
        )

    return errors


def _validate_threshold(value: float, *, name: str) -> float:
    """Validate a positive duration threshold in seconds."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite positive number of seconds.")
    result = float(value)
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a finite positive number of seconds.")
    return result


def build_targets(
    windows: FeatureWindows,
    annotations: list[Annotation],
    *,
    positive_threshold_seconds: float = 2.0,
    negative_threshold_seconds: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build binary targets from annotation durations within each window.

    MTB 3 duration counts toward the positive threshold. MTB 0 and 1
    durations count together toward the negative threshold. MTB 2 is ignored.
    A window is retained only when exactly one threshold is met; windows
    meeting both thresholds or neither threshold are discarded.

    Threshold comparisons are inclusive. Returned labels are 1.0 for positive
    and 0.0 for negative windows.
    """
    positive_threshold_seconds = _validate_threshold(
        positive_threshold_seconds, name="positive_threshold_seconds"
    )
    negative_threshold_seconds = _validate_threshold(
        negative_threshold_seconds, name="negative_threshold_seconds"
    )

    X_training: list[np.ndarray] = []
    y: list[float] = []
    timestamps = windows.timestamps

    for window_number in range(windows.X.shape[0]):
        positive_duration = 0.0
        negative_duration = 0.0

        for annotation in annotations:
            overlap = _annotation_overlap(
                timestamps[window_number],
                annotation,
            )
            if overlap <= 0:
                continue

            if annotation.score == 3:
                positive_duration += overlap
            elif annotation.score in (0, 1):
                negative_duration += overlap
            # MTB 2 is intentionally excluded from both counts.

        is_positive = positive_duration >= positive_threshold_seconds
        is_negative = negative_duration >= negative_threshold_seconds

        # Discard ambiguous windows and windows with insufficient evidence.
        if is_positive == is_negative:
            continue

        X_training.append(windows.X[window_number])
        y.append(1.0 if is_positive else 0.0)

    if not X_training:
        return (
            np.empty((0, windows.X.shape[1], windows.X.shape[2]), dtype=np.float32),
            np.empty(0, dtype=np.float32),
        )

    return (
        np.stack(X_training).astype(np.float32, copy=False),
        np.asarray(y, dtype=np.float32),
    )
