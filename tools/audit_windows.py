
"""Audit temporal windows generated from cached feature sequences."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

from autocut.config import Config
from autocut.data_models import FeatureSequence
from autocut.dataset.windows import build_windows
from autocut.annotations import load_training_annotations


RESET = "\033[0m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
CYAN = "\033[36m"


def print_pass(message: str) -> None:
    print(f"{GREEN}[PASS]{RESET} {message}")


def print_fail(message: str) -> None:
    print(f"{RED}[FAIL]{RESET} {message}")


def print_section(title: str) -> None:
    print()
    print(f"{CYAN}{'=' * 60}{RESET}")
    print(f"{CYAN}{title}{RESET}")
    print(f"{CYAN}{'=' * 60}{RESET}")


def audit_windows(
    sequence: FeatureSequence,
    *,
    window_duration: float,
    window_stride: float,
    annotation_number: int,
) -> int:
    """Audit one annotation's window count, geometry and contents.

    Returns:
        Number of windows produced if the audit passes.

    Raises:
        ValueError: If window generation or any audit check fails.
    """
    timestamps = sequence.timestamps
    features = sequence.features

    if len(timestamps) < 2:
        raise ValueError(
            "At least two feature timestamps are required."
        )

    # Independently establish the sampling interval.
    differences = np.diff(timestamps)
    dt = float(np.median(differences))

    if dt <= 0 or not np.allclose(
        differences, dt, rtol=1e-3, atol=1e-3
    ):
        raise ValueError(
            "Feature timestamps are not regularly spaced."
        )

    # The requested geometry must align with whole feature samples.
    window_samples_float = window_duration / dt
    stride_samples_float = window_stride / dt

    window_samples = round(window_samples_float)
    stride_samples = round(stride_samples_float)

    if window_samples < 2 or stride_samples < 1:
        raise ValueError("Invalid window or stride sample count.")

    if not np.isclose(
        window_samples_float, window_samples, rtol=0, atol=1e-6
    ):
        raise ValueError(
            f"Window duration {window_duration}s does not align "
            f"with feature interval {dt:.6f}s."
        )

    if not np.isclose(
        stride_samples_float, stride_samples, rtol=0, atol=1e-6
    ):
        raise ValueError(
            f"Window stride {window_stride}s does not align "
            f"with feature interval {dt:.6f}s."
        )

    # Independent expected count: each window uses W rows,
    # and consecutive windows start S rows apart.
    sample_count = len(features)

    if sample_count < window_samples:
        raise ValueError(
            f"Only {sample_count} feature rows are available; "
            f"{window_samples} are required for one window."
        )

    expected_count = (
        (sample_count - window_samples) // stride_samples
    ) + 1

    # Generate the actual result using production code.
    windows = build_windows(
        sequence,
        window=window_duration,
        stride=window_stride,
    )

    if windows.X.shape != (
        expected_count,
        window_samples,
        features.shape[1],
    ):
        raise ValueError(
            f"Window array shape mismatch: expected "
            f"({expected_count}, {window_samples}, "
            f"{features.shape[1]}), got {windows.X.shape}."
        )

    if windows.timestamps.shape != (expected_count, 2):
        raise ValueError(
            f"Window timestamp shape mismatch: expected "
            f"({expected_count}, 2), got "
            f"{windows.timestamps.shape}."
        )

    if len(windows.X) != expected_count:
        raise ValueError(
            f"Window count mismatch: expected {expected_count}, "
            f"got {len(windows.X)}."
        )

    # Independently derive the expected timestamps. Each feature
    # timestamp marks the END of its flow interval.
    flow_start = float(timestamps[0] - dt)

    expected_starts = (
        flow_start
        + np.arange(expected_count, dtype=np.float64)
        * stride_samples
        * dt
    )
    expected_timestamps = np.column_stack(
        (
            expected_starts,
            expected_starts + window_duration,
        )
    )

    if not np.allclose(
        windows.timestamps,
        expected_timestamps,
        rtol=0,
        atol=1e-6,
    ):
        difference = np.abs(
            windows.timestamps - expected_timestamps
        )
        row, column = np.unravel_index(
            np.argmax(difference), difference.shape
        )
        boundary = ("start", "end")[column]

        raise ValueError(
            f"Timestamp mismatch at window {row + 1}, "
            f"{boundary}: actual="
            f"{windows.timestamps[row, column]:.6f}s, "
            f"expected="
            f"{expected_timestamps[row, column]:.6f}s."
        )

    # Verify every output window contains precisely the expected
    # consecutive feature rows, without omissions or shifts.
    for index in range(expected_count):
        start = index * stride_samples
        expected_features = features[
            start:start + window_samples
        ]

        if not np.array_equal(
            windows.X[index], expected_features
        ):
            raise ValueError(
                f"Feature contents mismatch in window "
                f"{index + 1} (source row {start})."
            )

    # Verify actual durations and stride.
    durations = windows.timestamps[:, 1] - windows.timestamps[:, 0]

    if not np.allclose(
        durations, window_duration, rtol=0, atol=1e-6
    ):
        raise ValueError("One or more windows have incorrect duration.")

    if expected_count > 1:
        actual_strides = np.diff(windows.timestamps[:, 0])

        if not np.allclose(
            actual_strides, window_stride, rtol=0, atol=1e-6
        ):
            raise ValueError("One or more window strides are incorrect.")

    print_pass(
        f"Annotation {annotation_number:03d}: "
        f"{sample_count} feature rows -> "
        f"{expected_count} windows"
    )

    return expected_count


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit window generation using cached feature sequences."
        )
    )
    parser.add_argument(
        "annotations",
        type=Path,
        help="Training annotation file, e.g. data/Ascoli.txt",
    )
    args = parser.parse_args()

    annotation_path = args.annotations.resolve()

    try:
        _, annotations = load_training_annotations(annotation_path)
    except (OSError, ValueError) as exc:
        print_fail(f"Cannot load annotations: {exc}")
        return 1

    if not annotations:
        print_fail("No annotations found.")
        return 1

    config = Config()
    cache_dir = Path("audit") / annotation_path.stem

    print_section("Windowing Audit")
    print(f"Annotations:     {annotation_path}")
    print(f"Cached features:  {cache_dir}")
    print(f"Window duration:  {config.window_duration:.2f}s")
    print(f"Window stride:    {config.window_stride:.2f}s")
    print(f"Annotations:      {len(annotations)}")

    total_frames = 0
    total_windows = 0
    failures = 0

    for index, _ in enumerate(annotations, start=1):
        npz_path = cache_dir / f"annotation_{index:03d}.npz"

        try:
            with np.load(npz_path, allow_pickle=False) as cached:
                sequence = FeatureSequence(
                    features=cached["features"],
                    timestamps=cached["timestamps"],
                )

            count = audit_windows(
                sequence,
                window_duration=config.window_duration,
                window_stride=config.window_stride,
                annotation_number=index,
            )

            total_frames += len(sequence.features)
            total_windows += count

        except (OSError, KeyError, ValueError) as exc:
            failures += 1
            print_fail(f"Annotation {index:03d}: {exc}")

    print_section("Windowing Audit Summary")
    print(f"Annotations checked: {len(annotations)}")
    print(f"Feature frames:      {total_frames}")
    print(f"Windows generated:   {total_windows}")
    print(f"Failures:            {failures}")

    if failures:
        print_fail("Windowing audit failed.")
        return 1

    print_pass("All windowing audits passed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())