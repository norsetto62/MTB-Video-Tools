"""Focused tests for AutoCut window-level inference."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from autocut.data_models import FeatureWindows
from autocut.ml.checkpoint import build_checkpoint, save_checkpoint
from autocut.ml.inference import InferenceResult, infer_windows
from autocut.ml.model import MTBClassifier
from autocut.ml.scaling import FeatureScaler
from autocut.motion.features import FEATURE_NAMES


def _windows(seed=5, count=7, steps=8):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(count, steps, len(FEATURE_NAMES))).astype(np.float32)
    starts = np.arange(count, dtype=np.float64) * 2.0
    timestamps = np.column_stack((starts, starts + 4.0))
    return FeatureWindows(X=X, timestamps=timestamps)


def _checkpoint(*, scaler=None, feature_mask=None):
    model = MTBClassifier(
        feature_dim=int(sum(feature_mask)) if feature_mask is not None else 54,
        hidden_dim=4,
        classifier_mid_dim=5,
        dropout=0.0,
    )
    return build_checkpoint(
        model,
        scaler=scaler,
        feature_mask=feature_mask,
        data_config={"flow_fps": 2.0, "window": 4.0, "stride": 2.0},
    )


def test_inference_returns_timestamps_and_three_probabilities():
    windows = _windows()
    result = infer_windows(_checkpoint(), windows, batch_size=3)

    assert isinstance(result, InferenceResult)
    np.testing.assert_array_equal(result.timestamps, windows.timestamps)
    assert result.probabilities.shape == (len(windows.X), 3)
    assert result.probabilities.dtype == np.float32
    assert np.isfinite(result.probabilities).all()
    np.testing.assert_allclose(result.probabilities.sum(axis=1), 1.0, atol=1e-6)


def test_inference_accepts_checkpoint_file(tmp_path):
    windows = _windows()
    model = MTBClassifier(hidden_dim=4, classifier_mid_dim=5, dropout=0.0)
    path = tmp_path / "model.pt"
    save_checkpoint(path, model, data_config={"flow_fps": 2.0, "window": 4.0, "stride": 2.0})

    result = infer_windows(path, windows)
    assert result.probabilities.shape == (len(windows.X), 3)


def test_inference_reuses_checkpoint_scaler():
    windows = _windows()
    model = MTBClassifier(hidden_dim=4, classifier_mid_dim=5, dropout=0.0)
    scaler = FeatureScaler(enabled=True).fit(
        np.full((10, 8, 54), 2.0, dtype=np.float32)
    )
    payload = build_checkpoint(
        model,
        scaler=scaler,
        data_config={"flow_fps": 2.0, "window": 4.0, "stride": 2.0},
    )

    result = infer_windows(payload, windows)
    assert result.probabilities.shape == (len(windows.X), 3)


def test_inference_applies_checkpoint_feature_mask():
    windows = _windows()
    mask = [index % 2 == 0 for index in range(54)]
    result = infer_windows(_checkpoint(feature_mask=mask), windows)
    assert result.probabilities.shape == (len(windows.X), 3)


@pytest.mark.parametrize(
    "bad_windows, error",
    [
        (FeatureWindows(np.zeros((0, 8, 54), dtype=np.float32), np.zeros((0, 2))), "empty"),
        (FeatureWindows(np.zeros((2, 7, 54), dtype=np.float32), np.array([[0, 4], [2, 6]])), "time steps"),
        (FeatureWindows(np.zeros((2, 8, 53), dtype=np.float32), np.array([[0, 4], [2, 6]])), "canonical features"),
        (FeatureWindows(np.zeros((2, 8, 54), dtype=np.float32), np.array([[0, 3], [2, 6]])), "durations"),
        (FeatureWindows(np.zeros((2, 8, 54), dtype=np.float32), np.array([[0, 4], [0, 4]])), "strictly increasing"),
        (FeatureWindows(np.full((2, 8, 54), np.nan, dtype=np.float32), np.array([[0, 4], [2, 6]])), "finite"),
    ],
)
def test_inference_rejects_invalid_windows(bad_windows, error):
    with pytest.raises(ValueError, match=error):
        infer_windows(_checkpoint(), bad_windows)


def test_inference_rejects_invalid_batch_size():
    windows = _windows()
    for batch_size in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="batch_size"):
            infer_windows(_checkpoint(), windows, batch_size=batch_size)


def test_inference_rejects_non_feature_windows_input():
    with pytest.raises(TypeError, match="FeatureWindows"):
        infer_windows(_checkpoint(), np.zeros((1, 8, 54), dtype=np.float32))


def test_inference_does_not_return_argmax_or_selection_fields():
    result = infer_windows(_checkpoint(), _windows())
    assert set(vars(result)) == {"timestamps", "probabilities"}
