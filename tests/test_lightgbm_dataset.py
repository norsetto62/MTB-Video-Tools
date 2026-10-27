"""Integration tests for the LightGBM continuous-to-tabular target path."""
import numpy as np

from autocut.data_models import Annotation, FeatureSequence
from autocut.dataset.lightgbm import build_lightgbm_training_data
from autocut.dataset.windows import LIGHTGBM_TABULAR_FEATURE_NAMES
from autocut.motion.features import FEATURE_NAMES


def _sequence(sample_count=17):
    timestamps = np.arange(sample_count, dtype=np.float64) * 0.5
    features = np.zeros((sample_count, len(FEATURE_NAMES)), dtype=np.float32)
    # Give the flow fields and each grid cell distinct, finite values so all
    # eight engineered signals are exercised by the real transform.
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
    return FeatureSequence(features=features, timestamps=timestamps)


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
        sequence, annotations, window=4.0, stride=2.0
    )

    assert windows.X.ndim == 2
    assert windows.X.shape[1] == 48
    assert windows.X.shape[0] == len(labels)
    assert len(LIGHTGBM_TABULAR_FEATURE_NAMES) == 48
    assert labels.dtype == np.int64
    assert np.isfinite(windows.X).all()
    assert windows.timestamps.shape == (len(labels), 2)
    # First window has two seconds of class 0/1 combined vs two seconds class
    # 2, so the duration tie is resolved to class 2. Second window has a class
    # 2/class 3 tie, resolved to class 3. The final unannotated window is gone.
    np.testing.assert_array_equal(labels, [2, 3])
    np.testing.assert_allclose(windows.timestamps, [[-0.5, 3.5], [1.5, 5.5]])


def test_lightgbm_training_data_returns_empty_tabular_dataset_without_annotations():
    windows, labels = build_lightgbm_training_data(
        _sequence(), [], window=4.0, stride=2.0
    )
    assert windows.X.shape == (0, 48)
    assert windows.timestamps.shape == (0, 2)
    assert labels.shape == (0,)
