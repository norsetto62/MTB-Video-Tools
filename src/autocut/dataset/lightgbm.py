"""End-to-end dataset assembly for the four-class LightGBM path.

The preparation order is intentionally centralized:
canonical continuous features -> continuous engineered signals -> temporal
windows -> 48 tabular statistics -> duration-weighted annotation targets.
"""
from __future__ import annotations

import numpy as np

from ..data_models import Annotation, FeatureSequence, FeatureWindows
from .normalization import normalize_per_video_features
from ..motion.engineered import engineer_lightgbm_sequence
from .targets import build_majority_vote_targets
from .windows import build_lightgbm_windows_from_features


def build_lightgbm_training_data(
    sequence: FeatureSequence,
    annotations: list[Annotation],
    *,
    window: float,
    stride: float,
) -> tuple[FeatureWindows, np.ndarray]:
    """Build aligned LightGBM rows and MTB 0..3 labels from one video sequence.

    The eight signals are engineered over the entire continuous sequence before
    windowing. Each temporal window is then aggregated to 48 columns (eight
    signals times six statistics). Targets are duration-weighted majority votes
    over annotated portions of each window, with ties assigned to the higher
    MTB class. Windows with no annotation overlap are omitted.

    Returns:
        A FeatureWindows object whose X has shape (N, 48), plus an int64 target
        vector of shape (N,). X rows and labels are aligned after unlabeled
        windows have been filtered out.
    """
    if not isinstance(sequence, FeatureSequence):
        raise TypeError("sequence must be a FeatureSequence.")

    signals = engineer_lightgbm_sequence(sequence.features)
    tabular_windows = build_lightgbm_windows_from_features(
        signals,
        sequence.timestamps,
        window=window,
        stride=stride,
    )
    # Normalize over every window from this video before dropping windows
    # that have no unambiguous annotation target. Inference has no labels, so
    # its normalization statistics must be computed over the complete video.
    normalized_windows = FeatureWindows(
        X=normalize_per_video_features(tabular_windows.X),
        timestamps=tabular_windows.timestamps,
    )
    return build_majority_vote_targets(normalized_windows, annotations)
