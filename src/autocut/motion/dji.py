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
