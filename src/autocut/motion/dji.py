"""DJI quaternion-derived pitch-rate RMS features for AutoCut windows.

The telemetry is read from the same MP4 as the optical-flow sequence. Pitch
rates are expressed in degrees per second and aggregated over each model window.
"""
from __future__ import annotations

from functools import lru_cache
import importlib.util
from pathlib import Path
import sys

import numpy as np


def _telemetry_script() -> Path:
    return Path(__file__).resolve().parents[3] / "tools" / "dji_action4_telemetry.py"


@lru_cache(maxsize=16)
def _load_pitch_series(
    source_path: str, source_size: int, source_mtime_ns: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Load cached (elapsed time, pitch rate) arrays from one DJI MP4."""
    del source_size, source_mtime_ns  # Included in the cache key for invalidation.
    script = _telemetry_script()
    if not script.is_file():
        raise FileNotFoundError(
            f"DJI telemetry parser not found: {script}"
        )
    module_name = "_autocut_dji_action4_telemetry"
    module = sys.modules.get(module_name)
    if module is None:
        spec = importlib.util.spec_from_file_location(module_name, script)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load DJI telemetry parser: {script}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_name, None)
            raise

    samples, _metadata = module.extract_clip(Path(source_path))
    times: list[float] = []
    pitches: list[float] = []
    for sample in samples:
        quaternion = sample.get("quat")
        if quaternion is None:
            continue
        pitch = module.euler(quaternion)[1]
        timestamp = float(sample["t"])
        if np.isfinite(timestamp) and pitch is not None and np.isfinite(pitch):
            times.append(timestamp)
            pitches.append(float(pitch))

    t = np.asarray(times, dtype=np.float64)
    p = np.asarray(pitches, dtype=np.float64)
    if len(t) < 3:
        raise ValueError(
            f"DJI telemetry in {source_path} has fewer than three valid pitch samples."
        )
    if not np.all(np.diff(t) > 0):
        raise ValueError("DJI pitch telemetry timestamps must be strictly increasing.")

    # np.gradient uses the actual (potentially non-uniform) telemetry timestamps.
    rate = np.gradient(p, t)
    if not np.isfinite(rate).all():
        raise ValueError("DJI pitch-rate calculation produced non-finite values.")
    return t, rate



@lru_cache(maxsize=16)
def _load_imu_series(
    source_path: str, source_size: int, source_mtime_ns: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load body-frame angular velocity (deg/s) and dynamic acceleration (g)."""
    del source_size, source_mtime_ns
    script = _telemetry_script()
    if not script.is_file():
        raise FileNotFoundError(f"DJI telemetry parser not found: {script}")
    module_name = "_autocut_dji_action4_telemetry"
    module = sys.modules.get(module_name)
    if module is None:
        spec = importlib.util.spec_from_file_location(module_name, script)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load DJI telemetry parser: {script}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_name, None)
            raise

    samples, _metadata = module.extract_clip(Path(source_path))
    rows, _processing = module.process(samples, gravity_mode="quat", level=False)
    sample_times = np.asarray([float(sample["t"]) for sample in samples], dtype=np.float64)
    quaternions = [module.q_normalize(sample["quat"]) if sample.get("quat") is not None else None for sample in samples]
    dynamic_acceleration = [row.get("processed") for row in rows]

    times: list[float] = []
    angular: list[tuple[float, float, float]] = []
    acceleration: list[tuple[float, float, float]] = []
    for index in range(1, len(samples)):
        q0, q1 = quaternions[index - 1], quaternions[index]
        dt = sample_times[index] - sample_times[index - 1]
        acc = dynamic_acceleration[index]
        if q0 is None or q1 is None or acc is None or dt <= 0:
            continue
        # Relative rotation q0^-1*q1 is expressed in the body frame.
        w0, x0, y0, z0 = q0
        w1, x1, y1, z1 = q1
        delta = (
            w0*w1 + x0*x1 + y0*y1 + z0*z1,
            w0*x1 - x0*w1 - y0*z1 + z0*y1,
            w0*y1 + x0*z1 - y0*w1 - z0*x1,
            w0*z1 - x0*y1 + y0*x1 - z0*w1,
        )
        norm = float(np.linalg.norm(delta))
        if norm < 1e-9:
            continue
        delta = tuple(value / norm for value in delta)
        if delta[0] < 0:
            delta = tuple(-value for value in delta)
        vector_norm = float(np.linalg.norm(delta[1:]))
        if vector_norm < 1e-12:
            omega = (0.0, 0.0, 0.0)
        else:
            angle = 2.0 * np.arctan2(vector_norm, max(-1.0, min(1.0, delta[0])))
            factor = np.degrees(angle / (vector_norm * dt))
            omega = tuple(float(value * factor) for value in delta[1:])
        times.append(float(sample_times[index]))
        angular.append(omega)
        acceleration.append(tuple(float(value) for value in acc[:3]))

    t = np.asarray(times, dtype=np.float64)
    w = np.asarray(angular, dtype=np.float64)
    a = np.asarray(acceleration, dtype=np.float64)
    if len(t) < 3:
        raise ValueError(
            f"DJI telemetry in {source_path} has fewer than three valid combined IMU samples."
        )
    if not np.all(np.diff(t) > 0) or not np.isfinite(w).all() or not np.isfinite(a).all():
        raise ValueError("DJI IMU telemetry must be finite with strictly increasing timestamps.")
    return t, w, a


def load_imu_series(
    video_path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return cached timestamps, body angular rates (deg/s), and dynamic acceleration (g)."""
    source = Path(video_path).expanduser().resolve()
    stat = source.stat()
    times, angular, acceleration = _load_imu_series(
        str(source), stat.st_size, stat.st_mtime_ns
    )
    return times.copy(), angular.copy(), acceleration.copy()


def imu_window_statistics(
    times: np.ndarray,
    angular_velocity: np.ndarray,
    acceleration: np.ndarray,
    window_timestamps: np.ndarray,
    *,
    time_offset: float = 0.0,
) -> np.ndarray:
    """Return per-window axis angular-rate RMS/peaks and acceleration RMS/peak."""
    sample_times = np.asarray(times, dtype=np.float64)
    angular = np.asarray(angular_velocity, dtype=np.float64)
    accel = np.asarray(acceleration, dtype=np.float64)
    windows = np.asarray(window_timestamps, dtype=np.float64)
    offset = float(time_offset)
    if sample_times.ndim != 1 or angular.shape != (len(sample_times), 3) or accel.shape != (len(sample_times), 3):
        raise ValueError("IMU arrays must have shapes (N,), (N,3), and (N,3).")
    if len(sample_times) < 3 or not np.isfinite(sample_times).all() or not np.isfinite(angular).all() or not np.isfinite(accel).all():
        raise ValueError("IMU telemetry must contain at least three finite samples.")
    if not np.all(np.diff(sample_times) > 0):
        raise ValueError("IMU timestamps must be strictly increasing.")
    if windows.ndim != 2 or windows.shape[1] != 2 or len(windows) == 0 or not np.isfinite(windows).all():
        raise ValueError("window_timestamps must be a finite (N,2) array.")
    if not np.isfinite(offset) or np.any(windows[:, 1] <= windows[:, 0]):
        raise ValueError("IMU window timestamps and offset must be valid.")
    output = np.empty((len(windows), 8), dtype=np.float32)
    accel_magnitude = np.linalg.norm(accel, axis=1)
    for index, (start, end) in enumerate(windows):
        mask = (sample_times >= start + offset) & (sample_times < end + offset)
        if not np.any(mask):
            raise ValueError(f"No valid DJI IMU samples in window [{start + offset:.3f}, {end + offset:.3f}).")
        selected_angular = angular[mask]
        selected_accel = accel_magnitude[mask]
        output[index] = np.concatenate((
            np.sqrt(np.mean(selected_angular ** 2, axis=0)),
            np.max(np.abs(selected_angular), axis=0),
            [np.sqrt(np.mean(selected_accel ** 2)), np.max(np.abs(selected_accel))],
        ))
    if not np.isfinite(output).all():
        raise ValueError("Aggregated IMU features contain non-finite values.")
    return output

def load_pitch_rate_series(video_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Return cached DJI telemetry timestamps and pitch rates (degrees/second)."""
    source = Path(video_path).expanduser().resolve()
    stat = source.stat()
    times, rates = _load_pitch_series(str(source), stat.st_size, stat.st_mtime_ns)
    return times.copy(), rates.copy()


def pitch_rate_rms_from_series(
    times: np.ndarray,
    rates: np.ndarray,
    window_timestamps: np.ndarray,
    *,
    time_offset: float = 0.0,
) -> np.ndarray:
    """Aggregate cached pitch-rate samples into [start, end) window RMS values."""
    try:
        sample_times = np.asarray(times, dtype=np.float64)
        sample_rates = np.asarray(rates, dtype=np.float64)
        windows = np.asarray(window_timestamps, dtype=np.float64)
        offset = float(time_offset)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Pitch telemetry, windows and time_offset must be numeric.") from exc
    if sample_times.ndim != 1 or sample_rates.ndim != 1 or len(sample_times) != len(sample_rates) or len(sample_times) < 3:
        raise ValueError("Pitch telemetry arrays must be matching 1D arrays with at least three samples.")
    if not np.isfinite(sample_times).all() or not np.isfinite(sample_rates).all() or not np.all(np.diff(sample_times) > 0):
        raise ValueError("Pitch telemetry arrays must be finite with strictly increasing timestamps.")
    if windows.ndim != 2 or windows.shape[1] != 2 or len(windows) == 0:
        raise ValueError("window_timestamps must have non-empty shape (N, 2).")
    if not np.isfinite(windows).all() or not np.isfinite(offset):
        raise ValueError("Window timestamps and time_offset must be finite.")
    if np.any(windows[:, 1] <= windows[:, 0]):
        raise ValueError("Every window end must be after its start.")
    result = np.empty(len(windows), dtype=np.float32)
    for index, (start, end) in enumerate(windows):
        absolute_start, absolute_end = start + offset, end + offset
        mask = (sample_times >= absolute_start) & (sample_times < absolute_end)
        if not np.any(mask):
            raise ValueError(
                f"No valid DJI pitch-rate samples in video window "
                f"[{absolute_start:.3f}, {absolute_end:.3f})."
            )
        result[index] = np.sqrt(np.mean(np.square(sample_rates[mask])))
    if not np.isfinite(result).all():
        raise ValueError("Pitch-rate RMS contains non-finite values.")
    return result


def pitch_rate_rms_for_windows(
    video_path: str | Path,
    window_timestamps: np.ndarray,
    *,
    time_offset: float = 0.0,
) -> np.ndarray:
    times, rates = load_pitch_rate_series(video_path)
    return pitch_rate_rms_from_series(times, rates, window_timestamps, time_offset=time_offset)


def append_pitch_rate_rms(
    window_features: np.ndarray,
    pitch_rate_rms: np.ndarray,
) -> np.ndarray:
    """Append one window-level RMS value as a constant channel across its steps."""
    try:
        features = np.asarray(window_features, dtype=np.float32)
        rms = np.asarray(pitch_rate_rms, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Window features and pitch-rate RMS must be numeric.") from exc
    if features.ndim != 3 or features.shape[0] == 0 or features.shape[1] == 0:
        raise ValueError("window_features must have shape (N, T, F) with N,T >= 1.")
    if rms.shape != (features.shape[0],):
        raise ValueError("pitch_rate_rms must contain exactly one value per window.")
    if not np.isfinite(features).all() or not np.isfinite(rms).all():
        raise ValueError("Window features and pitch-rate RMS must be finite.")
    channel = np.broadcast_to(
        rms[:, np.newaxis, np.newaxis],
        (features.shape[0], features.shape[1], 1),
    )
    return np.concatenate((features, channel), axis=2).astype(np.float32, copy=False)
