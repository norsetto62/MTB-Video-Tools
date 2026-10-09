r""""
Each training window receives an interest target obtained from the
manual annotations covering that window.

Windows which are not covered by an annotation are NOT used.

The flow timestamps follow flow.py semantics: each timestamp is the 
END of the frame-to-frame interval used to calculate that row's 
optical-flow features. The generator therefore derives the
sampling interval from the flow timestamps and maps N selected rows
to the represented interval of N sampling periods.

If a window overlaps more than one annotation, its target is the
duration-weighted mean of the annotation scores.
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

    Target values may be fractional because windows can cross annotation
    boundaries. Every target must be finite and within the inclusive range
    0–3. Target-distribution summaries belong in the dataset audit rather
    than in this structural/value validator.

    Args:
        X: Windowed feature matrix of shape (N, T, F).
        timestamps: Window start/end times of shape (N, 2).
        y: Interestingness targets of shape (N,), in the range [0, 3].

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
        finite_values[(finite_values < 0.0) | (finite_values > 3.0)]
    )
    if invalid_values.size:
        errors.append(
            f"Interest scores must be within [0, 3]; invalid values: "
            f"{invalid_values.tolist()}."
        )

    return errors
def build_targets(
    windows: FeatureWindows,
    annotations: list[Annotation],
) -> tuple[np.ndarray, np.ndarray]:

    X_training: list[np.ndarray] = []
    y: list[float] = []
    timestamps = windows.timestamps

    for window_number in range(windows.X.shape[0]):
        covered_duration = 0.0
        weighted_score = 0.0

        for annotation in annotations:

            overlap =_annotation_overlap(
                timestamps[window_number],
                annotation,
            )

            if overlap <= 0:
                continue

            covered_duration += overlap
            weighted_score += (
                overlap * annotation.score
            )

        if covered_duration > 0:
            X_training.append(windows.X[window_number])
            y.append(
                weighted_score
                / covered_duration
            )

    if not X_training:
        return (
            np.empty((0, windows.X.shape[1], windows.X.shape[2]), dtype=np.float32),
            np.empty(0, dtype=np.float32),
        )
    
    return (
        np.stack(X_training).astype(
            np.float32,
            copy=False,
        ), np.asarray(
            y,
            dtype=np.float32,
        )
    )
