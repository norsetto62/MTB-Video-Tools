"""Build ordinal LightGBM targets from MTB annotation durations."""

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


def build_majority_vote_targets(
    windows: FeatureWindows,
    annotations: list[Annotation],
) -> tuple[FeatureWindows, np.ndarray]:
    """Label each window by duration-weighted majority vote over MTB classes.

    Only annotated portions contribute to the vote. A window with no annotated
    overlap is omitted. Ties are resolved in favor of the numerically highest
    class. The returned FeatureWindows keeps X and timestamps aligned.
    """
    X = np.asarray(windows.X)
    timestamps = np.asarray(windows.timestamps, dtype=np.float64)
    if X.ndim not in (2, 3):
        raise ValueError("windows.X must have shape (N,F) or (N,T,F).")
    if timestamps.shape != (len(X), 2):
        raise ValueError("window timestamps must have shape (N,2).")
    if not np.isfinite(timestamps).all():
        raise ValueError("window timestamps must contain only finite values.")
    for annotation in annotations:
        if annotation.score not in (0, 1, 2, 3):
            raise ValueError(f"Annotation score must be 0..3; got {annotation.score!r}.")
        if not np.isfinite(annotation.start) or not np.isfinite(annotation.end) or annotation.end <= annotation.start:
            raise ValueError("Annotations must have finite bounds with end > start.")

    kept_X: list[np.ndarray] = []
    kept_timestamps: list[np.ndarray] = []
    labels: list[int] = []
    for index, (start, end) in enumerate(timestamps):
        durations = np.zeros(4, dtype=np.float64)
        for annotation in annotations:
            durations[annotation.score] += _annotation_overlap(
                (float(start), float(end)), annotation
            )
        covered = float(durations.sum())
        if covered <= 0.0:
            continue
        label = int(np.flatnonzero(np.isclose(
            durations, durations.max(), rtol=0.0, atol=1e-9
        ))[-1])
        kept_X.append(X[index])
        kept_timestamps.append(timestamps[index])
        labels.append(label)

    if not kept_X:
        empty_shape = (0, *X.shape[1:])
        return (
            FeatureWindows(
                X=np.empty(empty_shape, dtype=np.float32),
                timestamps=np.empty((0, 2), dtype=np.float64),
            ),
            np.empty(0, dtype=np.int64),
        )
    return (
        FeatureWindows(
            X=np.stack(kept_X).astype(np.float32, copy=False),
            timestamps=np.stack(kept_timestamps).astype(np.float64, copy=False),
        ),
        np.asarray(labels, dtype=np.int64),
    )
