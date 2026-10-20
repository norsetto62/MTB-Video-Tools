"""Build binary training targets by counting sampled MTB labels.

Each feature row represents one flow-sampling interval. MTB 3 samples count
toward the positive threshold; MTB 0 and 1 samples count together toward the
negative threshold; MTB 2 and uncovered samples count toward neither class.
Positive takes precedence: if the positive threshold is met, label the window
positive; otherwise label it negative if the negative threshold is met. Reject
only windows meeting neither threshold.
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


def _validate_threshold(value: int, *, name: str) -> int:
    """Validate a positive sample-count threshold."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer sample count.")
    return value


def build_targets(
    windows: FeatureWindows,
    annotations: list[Annotation],
    *,
    positive_threshold_samples: int = 4,
    negative_threshold_samples: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Build binary labels from per-sample annotation classes.

    For each feature row, its represented sample interval is determined by
    dividing the window duration evenly by the number of rows. The annotation
    covering the sample midpoint supplies that sample's MTB label.

    MTB 3 contributes to the positive count. MTB 0 and 1 contribute together
    to the negative count. MTB 2 is ignored. A window is retained only when
    exactly one threshold is met; neither-threshold and mixed-threshold
    windows are discarded. Threshold comparisons are inclusive.
    """
    positive_threshold_samples = _validate_threshold(
        positive_threshold_samples, name="positive_threshold_samples"
    )
    negative_threshold_samples = _validate_threshold(
        negative_threshold_samples, name="negative_threshold_samples"
    )

    X_training: list[np.ndarray] = []
    y: list[float] = []
    timestamps = windows.timestamps
    samples_per_window = windows.X.shape[1]

    for window_number in range(windows.X.shape[0]):
        window_start, window_end = timestamps[window_number]
        sample_duration = (window_end - window_start) / samples_per_window
        positive_count = 0
        negative_count = 0

        for sample_number in range(samples_per_window):
            midpoint = window_start + (sample_number + 0.5) * sample_duration
            sample_score = next(
                (
                    annotation.score
                    for annotation in annotations
                    if annotation.start <= midpoint < annotation.end
                ),
                None,
            )
            if sample_score == 3:
                positive_count += 1
            elif sample_score in (0, 1):
                negative_count += 1
            # Uncovered samples and MTB 2 samples count toward neither class.

        if positive_count >= positive_threshold_samples:
            label = 1.0
        elif negative_count >= negative_threshold_samples:
            label = 0.0
        else:
            continue

        X_training.append(windows.X[window_number])
        y.append(label)

    if not X_training:
        return (
            np.empty((0, windows.X.shape[1], windows.X.shape[2]), dtype=np.float32),
            np.empty(0, dtype=np.float32),
        )

    return (
        np.stack(X_training).astype(np.float32, copy=False),
        np.asarray(y, dtype=np.float32),
    )
