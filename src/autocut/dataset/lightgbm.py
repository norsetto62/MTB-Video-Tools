"""End-to-end dataset assembly for the four-class LightGBM path.

The preparation order is intentionally centralized:
canonical continuous features -> continuous engineered signals -> temporal
windows -> 48 optical-flow statistics plus pitch-rate RMS -> duration-weighted annotation targets.
"""
from __future__ import annotations

import numpy as np

from ..data_models import Annotation, FeatureSequence, FeatureWindows
from .normalization import normalize_per_video_features
from ..motion.engineered import engineer_lightgbm_sequence
from ..motion.dji import pitch_rate_rms_from_series
from .targets import build_majority_vote_targets
from .windows import append_pitch_rate_rms_feature, build_lightgbm_windows_from_features


def build_lightgbm_training_data(
    sequence: FeatureSequence,
    annotations: list[Annotation],
    *,
    window: float,
    stride: float,
    pitch_timestamps: np.ndarray,
    pitch_rate: np.ndarray,
    pitch_time_offset: float = 0.0,
) -> tuple[FeatureWindows, np.ndarray]:
    """Build aligned LightGBM rows and MTB 0..3 labels from one video sequence.

    The eight signals are engineered over the entire continuous sequence before
    windowing. Each temporal window is aggregated to 48 optical-flow statistics and receives
    one additional DJI pitch-rate RMS feature (49 columns total). Targets are duration-weighted majority votes
    over annotated portions of each window, with ties assigned to the higher
    MTB class. Windows with no annotation overlap are omitted.

    Returns:
        A FeatureWindows object whose X has shape (N, 49), plus an int64 target
        vector of shape (N,). X rows and labels are aligned after unlabeled
        windows have been filtered out.
    """
    if not isinstance(sequence, FeatureSequence):
        raise TypeError("sequence must be a FeatureSequence.")

    signals = engineer_lightgbm_sequence(sequence.features)
    tabular_windows = build_lightgbm_windows_from_features(
        signals, sequence.timestamps, window=window, stride=stride,
    )
    pitch_rms = pitch_rate_rms_from_series(
        pitch_timestamps, pitch_rate, tabular_windows.timestamps,
        time_offset=pitch_time_offset,
    )
    tabular_windows = append_pitch_rate_rms_feature(tabular_windows, pitch_rms)
    # Normalize over every window from this video before dropping windows
    # that have no unambiguous annotation target. Inference has no labels, so
    # its normalization statistics must be computed over the complete video.
    normalized_windows = FeatureWindows(
        X=normalize_per_video_features(tabular_windows.X),
        timestamps=tabular_windows.timestamps,
    )
    return build_majority_vote_targets(normalized_windows, annotations)
