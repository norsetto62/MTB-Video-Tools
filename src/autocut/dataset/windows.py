"""Generate fixed-size temporal windows from feature sequences."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..data_models import FeatureSequence

DEFAULT_WINDOW = 4.0
DEFAULT_STRIDE = 2.0


@dataclass(frozen=True)
class FeatureWindows:
    """Fixed-size temporal windows extracted from a feature sequence.

    Attributes:
        X: Windowed feature matrix with shape (N, T, F).
        timestamps: Window start/end times with shape (N, 2).
    """

    X: np.ndarray
    timestamps: np.ndarray


def build_windows(
    sequence: FeatureSequence,
    *,
    window: float = DEFAULT_WINDOW,
    stride: float = DEFAULT_STRIDE,
) -> FeatureWindows:
    """Generate fixed-size temporal windows.

    Feature timestamps represent the END of each frame-to-frame
    optical-flow interval.

    For example, at 2 FPS:

        timestamp=0.5 -> flow over [0.0, 0.5)
        timestamp=1.0 -> flow over [0.5, 1.0)

    Therefore, if the first feature timestamp is 0.5 and ``dt`` is
    0.5, the represented flow interval begins at 0.0.

    Windows are placed on a canonical temporal grid derived from the
    flow sampling interval. Window times are therefore not taken from
    floating-point arithmetic on individual feature timestamps.

    Args:
        sequence: Time-ordered feature sequence.
        window: Window duration in seconds.
        stride: Distance between consecutive window starts in seconds.

    Returns:
        FeatureWindows containing:

        - ``X`` with shape ``(N, samples_per_window, n_features)``
        - ``timestamps`` with shape ``(N, 2)``, containing
          ``[start, end]`` for each window.

    Raises:
        ValueError: If the temporal geometry is invalid or the feature
            sequence is too short to produce a complete window.
    """
    if window <= 0:
        raise ValueError("window must be greater than zero.")

    if stride <= 0:
        raise ValueError("stride must be greater than zero.")

    timestamps = sequence.timestamps
    features = sequence.features

    if len(timestamps) != len(features):
        raise ValueError(
            "timestamps and feature matrix have different lengths."
        )

    if len(timestamps) < 2:
        raise ValueError(
            "at least two feature timestamps are required "
            "to determine the flow sampling interval."
        )

    # Feature timestamps represent the END of each flow interval.
    # Derive dt from the actual timestamps rather than assuming FPS.
    dt = float(np.median(np.diff(timestamps)))

    if dt <= 0:
        raise ValueError("invalid feature sampling interval.")

    window_samples_float = window / dt
    stride_samples_float = stride / dt

    samples_per_window = int(round(window_samples_float))
    samples_per_stride = int(round(stride_samples_float))

    # Do not silently change the requested temporal geometry by
    # rounding window or stride to a different number of samples.
    if not np.isclose(
        window_samples_float,
        samples_per_window,
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError(
            f"window={window:.6f}s is not an integer multiple "
            f"of feature dt={dt:.6f}s."
        )

    if not np.isclose(
        stride_samples_float,
        samples_per_stride,
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError(
            f"stride={stride:.6f}s is not an integer multiple "
            f"of feature dt={dt:.6f}s."
        )

    if samples_per_window < 2:
        raise ValueError(
            f"window={window:.3f}s is too short "
            f"for feature dt={dt:.6f}s."
        )

    # The first feature represents the interval immediately preceding
    # its timestamp. Therefore N feature rows represent:
    #
    #     [first_timestamp - dt, last_timestamp)
    #
    flow_start = float(timestamps[0] - dt)
    flow_end = float(timestamps[-1])

    represented_duration = flow_end - flow_start

    max_start_offset = represented_duration - window

    if max_start_offset < -1e-9:
        raise ValueError(
            f"feature duration={represented_duration:.6f}s is shorter "
            f"than window={window:.6f}s."
        )

    # Number of complete windows on the canonical stride grid.
    n_windows = int(np.floor((max_start_offset + 1e-9) / stride)) + 1

    windows: list[np.ndarray] = []
    window_timestamps: list[tuple[float, float]] = []

    for window_number in range(n_windows):
        index = window_number * samples_per_stride

        windows.append(
            features[index:index + samples_per_window]
        )

        # The first flow row represents [flow_start, flow_start + dt).
        # Therefore row i starts at flow_start + i * dt.
        window_start = flow_start + index * dt
        window_end = window_start + window

        window_timestamps.append((window_start, window_end))

    X = np.stack(windows).astype(
        np.float32,
        copy=False,
    )

    timestamps_array = np.asarray(
        window_timestamps,
        dtype=np.float64,
    )

    return FeatureWindows(
        X=X,
        timestamps=timestamps_array,
    )