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
