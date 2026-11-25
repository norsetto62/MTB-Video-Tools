"""Tests for window-level DJI pitch-rate RMS features."""

from pathlib import Path

import numpy as np
import pytest

from autocut.motion import dji


def test_pitch_rate_rms_uses_samples_inside_each_video_window(tmp_path, monkeypatch):
    video = tmp_path / "ride.mp4"
    video.touch()
    times = np.arange(0.0, 8.0, 0.04, dtype=np.float64)
    rates = np.where(times < 4.0, 3.0, 4.0).astype(np.float64)
    monkeypatch.setattr(dji, "_load_pitch_series", lambda *args: (times, rates))

    result = dji.pitch_rate_rms_for_windows(
        video, np.array([[0.0, 4.0], [2.0, 6.0], [4.0, 8.0]])
    )

    np.testing.assert_allclose(result, [3.0, np.sqrt(12.5), 4.0], rtol=1e-6)


def test_pitch_rate_rms_maps_local_windows_to_source_timeline(tmp_path, monkeypatch):
    video = tmp_path / "ride.mp4"
    video.touch()
    times = np.arange(0.0, 12.0, 0.1, dtype=np.float64)
    rates = times.copy()
    monkeypatch.setattr(dji, "_load_pitch_series", lambda *args: (times, rates))

    result = dji.pitch_rate_rms_for_windows(
        video, np.array([[0.0, 2.0]]), time_offset=6.0
    )

    expected = np.sqrt(np.mean(np.square(rates[(times >= 6.0) & (times < 8.0)])))
    np.testing.assert_allclose(result, [expected], rtol=1e-6)


def test_pitch_rate_rms_from_cached_series_uses_each_window():
    times = np.arange(0.0, 8.0, 0.04, dtype=np.float64)
    rates = np.where(times < 4.0, 3.0, 4.0).astype(np.float64)
    result = dji.pitch_rate_rms_from_series(times, rates, np.array([[0.0, 4.0], [4.0, 8.0]]))
    np.testing.assert_allclose(result, [3.0, 4.0], rtol=1e-6)


def test_append_pitch_rate_rms_adds_one_constant_channel_per_window():
    features = np.zeros((2, 8, 7), dtype=np.float32)
    result = dji.append_pitch_rate_rms(
        features, np.array([12.5, 30.0], dtype=np.float32)
    )

    assert result.shape == (2, 8, 8)
    np.testing.assert_array_equal(result[0, :, -1], np.full(8, 12.5))
    np.testing.assert_array_equal(result[1, :, -1], np.full(8, 30.0))


def test_append_pitch_rate_rms_rejects_mismatched_window_count():
    with pytest.raises(ValueError, match="one value per window"):
        dji.append_pitch_rate_rms(
            np.zeros((2, 8, 7), dtype=np.float32),
            np.array([1.0], dtype=np.float32),
        )



def test_imu_window_statistics_include_axis_rms_peaks_and_acceleration_impulses():
    times = np.arange(0.0, 4.0, 0.01, dtype=np.float64)
    angular = np.column_stack((
        np.full_like(times, 2.0),
        np.where(times < 1.0, 3.0, 0.0),
        np.full_like(times, -4.0),
    ))
    acceleration = np.zeros((len(times), 3), dtype=np.float64)
    acceleration[(times >= 1.0) & (times < 1.1), 2] = 5.0
    result = dji.imu_window_statistics(
        times, angular, acceleration, np.array([[0.0, 4.0]])
    )
    assert result.shape == (1, 8)
    np.testing.assert_allclose(result[0, :3], [2.0, 1.5, 4.0], rtol=1e-6)
    np.testing.assert_allclose(result[0, 3:6], [2.0, 3.0, 4.0], rtol=1e-6)
    assert result[0, 6] > 0.0
    assert result[0, 7] == pytest.approx(5.0)
