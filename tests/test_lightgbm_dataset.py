"""Integration tests for the LightGBM continuous-to-tabular target path."""
import numpy as np

from autocut.data_models import Annotation, FeatureSequence
from autocut.dataset.lightgbm import build_lightgbm_training_data
from autocut.dataset.normalization import DEFAULT_FEATURE_NORMALIZATION
from autocut.dataset.windows import LIGHTGBM_TABULAR_FEATURE_NAMES
from autocut.motion.features import FEATURE_NAMES


def _sequence(sample_count=25):
    timestamps = np.arange(sample_count, dtype=np.float64) * 0.5
    features = np.zeros((sample_count, len(FEATURE_NAMES)), dtype=np.float32)
    # Give the flow fields and each grid cell distinct, finite values so all
    # eleven engineered signals are exercised by the real transform.
    for row in range(3):
        for col in range(3):
            features[:, FEATURE_NAMES.index(f"grid_{row}_{col}_u")] = (
                0.5 + row + 0.25 * col + timestamps * (row + 1) * 0.1
            )
            features[:, FEATURE_NAMES.index(f"grid_{row}_{col}_v")] = (
                0.2 + col + 0.1 * row + timestamps * (col + 1) * 0.05
            )
    features[:, FEATURE_NAMES.index("flow_u")] = 0.75 + timestamps * 0.02
    features[:, FEATURE_NAMES.index("flow_v")] = 0.25 + timestamps * 0.01
    imu_timestamps = np.arange(-1.0, 14.0, 0.04, dtype=np.float64)
    angular = np.column_stack((
        2.0 + 0.1 * imu_timestamps,
        1.0 + 0.15 * imu_timestamps + 0.01 * imu_timestamps**2,
        0.5 + 0.05 * imu_timestamps,
    )).astype(np.float64)
    acceleration = np.column_stack((
        0.1 * np.sin(imu_timestamps),
        0.2 * np.cos(imu_timestamps),
        0.05 * np.sin(2 * imu_timestamps),
    )).astype(np.float64)
    return FeatureSequence(
        features=features, timestamps=timestamps,
        imu_timestamps=imu_timestamps,
        imu_angular_velocity=angular,
        imu_acceleration=acceleration,
    )


def _annotation(start, end, score):
    return Annotation(start=start, end=end, score=score, remarks="")


def test_lightgbm_training_data_connects_signals_windows_and_majority_targets():
    sequence = _sequence()
    annotations = [
        _annotation(0.0, 1.0, 0),
        _annotation(1.0, 2.0, 1),
        _annotation(2.0, 4.0, 2),
        _annotation(4.0, 6.0, 3),
        # No annotation after 6s: windows without any covered duration drop.
    ]

    windows, labels = build_lightgbm_training_data(
        sequence, annotations, window=4.0, stride=2.0,
        imu_timestamps=sequence.imu_timestamps,
        imu_angular_velocity=sequence.imu_angular_velocity,
        imu_acceleration=sequence.imu_acceleration,
        normalization_method=DEFAULT_FEATURE_NORMALIZATION,
    )

    assert windows.X.ndim == 2
    assert windows.X.shape[1] == 74
    assert windows.X.shape[0] == len(labels)
    assert len(LIGHTGBM_TABULAR_FEATURE_NAMES) == 74
    assert LIGHTGBM_TABULAR_FEATURE_NAMES[-8:] == (
        "roll_rate_rms", "pitch_rate_rms", "yaw_rate_rms",
        "roll_rate_peak", "pitch_rate_peak", "yaw_rate_peak",
        "accel_dynamic_rms", "accel_dynamic_peak",
    )
    assert labels.dtype == np.int64
    assert np.isfinite(windows.X).all()
    assert np.std(windows.X[:, -8:-2]) > 0.0
    assert np.std(windows.X[:, -2:]) > 0.0
    assert windows.timestamps.shape == (len(labels), 2)
    # The first window is won by class 2; the next three windows are won by
    # class 2 or 3 according to annotated overlap. The last window has no
    # annotated duration and is omitted.
    np.testing.assert_array_equal(labels, [2, 2, 3, 3])
    np.testing.assert_allclose(
        windows.timestamps,
        [[-0.5, 3.5], [1.5, 5.5], [3.5, 7.5], [5.5, 9.5]],
    )


def test_lightgbm_training_data_returns_empty_tabular_dataset_without_annotations():
    windows, labels = build_lightgbm_training_data(
        _sequence(), [], window=4.0, stride=2.0,
        imu_timestamps=_sequence().imu_timestamps,
        imu_angular_velocity=_sequence().imu_angular_velocity,
        imu_acceleration=_sequence().imu_acceleration,
        normalization_method=DEFAULT_FEATURE_NORMALIZATION,
    )
    assert windows.X.shape == (0, 74)
    assert windows.timestamps.shape == (0, 2)
    assert labels.shape == (0,)
