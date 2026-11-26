"""End-to-end dataset assembly for the four-class LightGBM path.

The preparation order is intentionally centralized:
canonical continuous features -> continuous engineered signals -> temporal
windows -> 66 optical-flow statistics plus eight DJI IMU statistics -> normalization -> duration-weighted targets.
"""
from __future__ import annotations

import numpy as np

from ..data_models import Annotation, FeatureSequence, FeatureWindows
from .normalization import normalize_per_video_features
from ..motion.engineered import engineer_lightgbm_sequence
from ..motion.dji import imu_window_statistics
from .targets import build_majority_vote_targets
from .windows import append_imu_features, build_lightgbm_windows_from_features


def build_lightgbm_training_data(
    sequence: FeatureSequence,
    annotations: list[Annotation],
    *,
    window: float,
    stride: float,
    imu_timestamps: np.ndarray,
    imu_angular_velocity: np.ndarray,
    imu_acceleration: np.ndarray,
    pitch_time_offset: float = 0.0,
    normalization_method: str,
) -> tuple[FeatureWindows, np.ndarray]:
    """Build aligned LightGBM rows and MTB 0..3 labels from one video sequence.

    Eleven signals are engineered over the entire continuous sequence before
    windowing. Each temporal window is aggregated to 66 optical-flow statistics;
    eight DJI IMU statistics (three-axis angular-rate RMS/peaks and dynamic
    acceleration RMS/peak) are appended. Targets are duration-weighted majority votes
    over annotated portions of each window, with ties assigned to the higher
    MTB class. Windows with no annotation overlap are omitted.

    Returns:
        A FeatureWindows object whose X has shape (N, 74), plus an int64 target
        vector of shape (N,). X rows and labels are aligned after unlabeled
        windows have been filtered out.
    """
    if not isinstance(sequence, FeatureSequence):
        raise TypeError("sequence must be a FeatureSequence.")

    signals = engineer_lightgbm_sequence(sequence.features)
    tabular_windows = build_lightgbm_windows_from_features(
        signals, sequence.timestamps, window=window, stride=stride,
    )
    imu_statistics = imu_window_statistics(
        imu_timestamps,
        imu_angular_velocity,
        imu_acceleration,
        tabular_windows.timestamps,
        time_offset=pitch_time_offset,
    )
    tabular_windows = append_imu_features(tabular_windows, imu_statistics)
    # Normalize over every window from this video before dropping windows
    # that have no unambiguous annotation target. Inference has no labels, so
    # its normalization statistics must be computed over the complete video.
    normalized_windows = FeatureWindows(
        X=normalize_per_video_features(tabular_windows.X, method=normalization_method),
        timestamps=tabular_windows.timestamps,
    )
    return build_majority_vote_targets(normalized_windows, annotations)
