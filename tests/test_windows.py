import numpy as np
import pytest

from autocut.data_models import FeatureSequence, FeatureWindows
from autocut.dataset.windows import build_windows
from autocut.config import Config


def make_sequence(
    n_samples: int,
    *,
    dt: float = 0.5,
    start: float = 0.5,
    n_features: int = 54,
) -> FeatureSequence:
    """Create a synthetic regularly sampled feature sequence."""
    timestamps = start + np.arange(n_samples) * dt

    features = np.arange(
        n_samples * n_features,
        dtype=np.float32,
    ).reshape(n_samples, n_features)

    return FeatureSequence(
        features=features,
        timestamps=timestamps,
    )


def test_default_constants():
    assert Config.window_duration == 4.0
    assert Config.window_stride == 2.0


def test_default_geometry():
    """Default 4 s / 2 s geometry produces 8-sample windows."""
    sequence = make_sequence(16)

    result = build_windows(sequence, window=Config.window_duration, stride=Config.window_stride)

    assert isinstance(result, FeatureWindows)
    assert result.X.shape == (3, 8, 54)
    assert result.timestamps.shape == (3, 2)


def test_window_timestamps():
    """Window timestamps follow the canonical temporal grid."""
    sequence = make_sequence(16)

    result = build_windows(sequence, window=Config.window_duration, stride=Config.window_stride)

    expected = np.array(
        [
            [0.0, 4.0],
            [2.0, 6.0],
            [4.0, 8.0],
        ]
    )

    np.testing.assert_allclose(
        result.timestamps,
        expected,
        rtol=0.0,
        atol=1e-9,
    )


def test_first_window_uses_first_eight_features():
    """The first window contains exactly the first 8 feature vectors."""
    sequence = make_sequence(16)

    result = build_windows(sequence, window=Config.window_duration, stride=Config.window_stride)

    np.testing.assert_array_equal(
        result.X[0],
        sequence.features[:8],
    )


def test_stride_overlaps_windows():
    """A 2 s stride at 2 FPS advances by four feature vectors."""
    sequence = make_sequence(16)

    result = build_windows(sequence, window=Config.window_duration, stride=Config.window_stride)

    np.testing.assert_array_equal(
        result.X[1],
        sequence.features[4:12],
    )

    np.testing.assert_array_equal(
        result.X[2],
        sequence.features[8:16],
    )


def test_absolute_timestamps_are_preserved():
    """Window times remain absolute when the sequence starts later."""
    sequence = make_sequence(
        16,
        start=420.5,
    )

    result = build_windows(sequence, window=Config.window_duration, stride=Config.window_stride)

    expected = np.array(
        [
            [420.0, 424.0],
            [422.0, 426.0],
            [424.0, 428.0],
        ]
    )

    np.testing.assert_allclose(
        result.timestamps,
        expected,
        rtol=0.0,
        atol=1e-9,
    )


def test_exactly_one_window():
    """Exactly 8 samples at 2 FPS represent one 4-second window."""
    sequence = make_sequence(8)

    result = build_windows(sequence, window=Config.window_duration, stride=Config.window_stride)

    assert result.X.shape == (1, 8, 54)
    assert result.timestamps.shape == (1, 2)

    np.testing.assert_allclose(
        result.timestamps[0],
        [0.0, 4.0],
    )


def test_no_partial_final_window():
    """Incomplete windows at the end are discarded."""
    sequence = make_sequence(10)

    result = build_windows(sequence, window=Config.window_duration, stride=Config.window_stride)

    assert result.X.shape == (1, 8, 54)

    np.testing.assert_array_equal(
        result.X[0],
        sequence.features[:8],
    )


def test_window_and_stride_can_differ_from_defaults():
    """Window geometry is controlled by the arguments."""
    sequence = make_sequence(20)

    result = build_windows(
        sequence,
        window=3.0,
        stride=1.0,
    )

    # 3 s = 6 samples, 1 s = 2 samples.
    assert result.X.shape == (8, 6, 54)

    expected_times = np.array(
        [
            [0.0, 3.0],
            [1.0, 4.0],
            [2.0, 5.0],
            [3.0, 6.0],
            [4.0, 7.0],
            [5.0, 8.0],
            [6.0, 9.0],
            [7.0, 10.0],
        ]
    )

    np.testing.assert_allclose(
        result.timestamps,
        expected_times,
    )


def test_window_must_be_positive():
    sequence = make_sequence(16)

    with pytest.raises(
        ValueError,
        match="window must be greater than zero",
    ):
        build_windows(
            sequence,
            window=0.0,
            stride=Config.window_stride
        )


def test_negative_window_is_rejected():
    sequence = make_sequence(16)

    with pytest.raises(
        ValueError,
        match="window must be greater than zero",
    ):
        build_windows(
            sequence,
            window=-1.0,
            stride=Config.window_stride
        )


def test_stride_must_be_positive():
    sequence = make_sequence(16)

    with pytest.raises(
        ValueError,
        match="stride must be greater than zero",
    ):
        build_windows(
            sequence,
            stride=0.0,
            window=Config.window_duration
        )


def test_negative_stride_is_rejected():
    sequence = make_sequence(16)

    with pytest.raises(
        ValueError,
        match="stride must be greater than zero",
    ):
        build_windows(
            sequence,
            stride=-1.0,
            window=Config.window_duration
        )


def test_at_least_two_timestamps_are_required():
    sequence = make_sequence(1)

    with pytest.raises(
        ValueError,
        match="at least two feature timestamps",
    ):
        build_windows(sequence, window=Config.window_duration, stride=Config.window_stride)


def test_window_must_be_integer_multiple_of_dt():
    sequence = make_sequence(20, dt=0.5)

    with pytest.raises(
        ValueError,
        match="window=.*is not an integer multiple",
    ):
        build_windows(
            sequence,
            window=3.1,
            stride=Config.window_stride
        )


def test_stride_must_be_integer_multiple_of_dt():
    sequence = make_sequence(20, dt=0.5)

    with pytest.raises(
        ValueError,
        match="stride=.*is not an integer multiple",
    ):
        build_windows(
            sequence,
            stride=1.1,
            window=Config.window_duration
        )


def test_window_must_contain_at_least_two_samples():
    sequence = make_sequence(10, dt=1.0)

    with pytest.raises(
        ValueError,
        match="window=.*is too short",
    ):
        build_windows(
            sequence,
            window=1.0,
            stride=Config.window_stride
        )


def test_sequence_shorter_than_window_is_rejected():
    sequence = make_sequence(7)

    with pytest.raises(
        ValueError,
        match="feature duration=.*is shorter",
    ):
        build_windows(sequence, window=Config.window_duration, stride=Config.window_stride)


def test_output_is_float32():
    sequence = make_sequence(16)

    result = build_windows(sequence, window=Config.window_duration, stride=Config.window_stride)

    assert result.X.dtype == np.float32
    assert result.timestamps.dtype == np.float64


def test_input_features_are_not_modified():
    sequence = make_sequence(16)
    original = sequence.features.copy()

    build_windows(sequence, window=Config.window_duration, stride=Config.window_stride)

    np.testing.assert_array_equal(
        sequence.features,
        original,
    )