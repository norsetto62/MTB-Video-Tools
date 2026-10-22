#!/usr/bin/env python3

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

from autocut.annotations import load_training_annotations
from autocut.data_models import Clip, FeatureSequence
from autocut.dataset.sequence import build_feature_sequence
from autocut.utils import convert_s_to_hms
from autocut.video.probe import get_video_info

SAMPLE_FPS = 2.0
PROCESSING_WIDTH = 480
EXPECTED_FEATURE_DIM = 54

RESET = "\033[0m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
CYAN = "\033[36m"
GRAY = "\033[90m"

def print_pass(message: str) -> None:
    print(f"{GREEN}[PASS]{RESET} {message}")


def print_warning(message: str) -> None:
    print(f"{YELLOW}[WARNING]{RESET} {message}")


def print_fail(message: str) -> None:
    print(f"{RED}[FAIL]{RESET} {message}")

def print_section(title: str) -> None:
    print()
    print(f"{CYAN}{'=' * 60}{RESET}")
    print(f"{CYAN}{title}{RESET}")
    print(f"{CYAN}{'=' * 60}{RESET}")


# ============================================================================
# Audit Functions
# ============================================================================

NEAR_ZERO_TOLERANCE = 1e-8


def calculate_feature_statistics(features: np.ndarray) -> dict:
    """Calculate descriptive statistics for each feature."""

    return {
        "min": np.min(features, axis=0),
        "max": np.max(features, axis=0),
        "mean": np.mean(features, axis=0),
        "std": np.std(features, axis=0),
        "p01": np.percentile(features, 1, axis=0),
        "median": np.percentile(features, 50, axis=0),
        "p99": np.percentile(features, 99, axis=0),
        "near_zero_fraction": np.mean(
            np.abs(features) < NEAR_ZERO_TOLERANCE,
            axis=0,
        ),
    }


NEAR_ZERO_WARNING_FRACTION = 0.95
CONSTANT_FEATURE_TOLERANCE = 1e-8


def report_feature_statistics(features: np.ndarray) -> None:
    """Report descriptive statistics for each feature."""

    statistics = {
        "min": np.min(features, axis=0),
        "max": np.max(features, axis=0),
        "mean": np.mean(features, axis=0),
        "std": np.std(features, axis=0),
        "p01": np.percentile(features, 1, axis=0),
        "median": np.percentile(features, 50, axis=0),
        "p99": np.percentile(features, 99, axis=0),
        "near_zero": np.mean(
            np.abs(features) < NEAR_ZERO_TOLERANCE,
            axis=0,
        ),
    }

    print_section("Feature Statistical Sanity")

    print(f"Feature frames: {features.shape[0]}")
    print(f"Feature count:  {features.shape[1]}")
    print(f"Near-zero tolerance: {NEAR_ZERO_TOLERANCE:g}")
    print()

    print(
        f"{'Feature':<27}"
        f"{'Min':>11}"
        f"{'Max':>11}"
        f"{'Mean':>11}"
        f"{'Std':>11}"
        f"{'P01':>11}"
        f"{'Median':>11}"
        f"{'P99':>11}"
        f"{'Zero %':>9}"
        f"  Status"
    )

    warning_count = 0

    for index, name in enumerate(FEATURE_NAMES):
        std = statistics["std"][index]
        near_zero = statistics["near_zero"][index]

        warnings = []

        if std < CONSTANT_FEATURE_TOLERANCE:
            warnings.append("near-constant")

        if near_zero >= NEAR_ZERO_WARNING_FRACTION:
            warnings.append("mostly zero")

        status = ", ".join(warnings) if warnings else "OK"

        if warnings:
            color = YELLOW
            warning_count += 1
        else:
            color = GREEN

        print(
            f"{color}"
            f"{name:<27}"
            f"{statistics['min'][index]:>11.4g}"
            f"{statistics['max'][index]:>11.4g}"
            f"{statistics['mean'][index]:>11.4g}"
            f"{std:>11.4g}"
            f"{statistics['p01'][index]:>11.4g}"
            f"{statistics['median'][index]:>11.4g}"
            f"{statistics['p99'][index]:>11.4g}"
            f"{near_zero * 100:>8.1f}%"
            f"  {status}"
            f"{RESET}"
        )

    print()

    if warning_count:
        print_warning(
            f"{warning_count} feature(s) have statistical warnings."
        )
    else:
        print_pass("No statistical warnings detected.")


def audit_feature_sequence(
    sequence: FeatureSequence,
    expected_dim: int = EXPECTED_FEATURE_DIM,
    expected_fps: float = SAMPLE_FPS,
) -> None:
    """Run strict mathematical and statistical checks on a FeatureSequence.

    Raises:
        ValueError: If any structural, numerical, or statistical audit fails.
    """
    features = np.asarray(sequence.features)
    timestamps = np.asarray(sequence.timestamps)

    # 1. Structural Checks
    if features.ndim != 2:
        raise ValueError(
            f"[Structure] Feature matrix must be 2D (T, F), got shape {features.shape}."
        )

    if timestamps.ndim != 1:
        raise ValueError(
            f"[Structure] Timestamps must be 1D (T,), got shape {timestamps.shape}."
        )

    num_frames, num_feats = features.shape
    if len(timestamps) != num_frames:
        raise ValueError(
            f"[Structure] Timestamps count ({len(timestamps)}) does not match "
            f"feature frame count ({num_frames})."
        )

    if num_frames < 2:
        raise ValueError(
            f"[Structure] Feature sequence must contain at least 2 frames for flow analysis, got {num_frames}."
        )

    if num_feats != expected_dim:
        raise ValueError(
            f"[Structure] Feature dimension mismatch: expected {expected_dim}, got {num_feats}."
        )

    # 2. Mathematical Invariants & Timing Checks
    dt_series = np.diff(timestamps)
    if np.any(dt_series <= 0):
        invalid_idxs = np.where(dt_series <= 0)[0]
        raise ValueError(
            f"[Invariant] Non-positive temporal step at indices {invalid_idxs.tolist()}."
        )

    expected_dt = 1.0 / expected_fps
    median_dt = float(np.median(dt_series))
    if not np.isclose(median_dt, expected_dt, rtol=1e-3, atol=1e-4):
        raise ValueError(
            f"[Invariant] Temporal step mismatch: expected nominal dt={expected_dt:.4f}s "
            f"({expected_fps} FPS), but observed median dt={median_dt:.4f}s."
        )

    if not np.allclose(dt_series, median_dt, rtol=1e-3, atol=1e-3):
        max_dev = float(np.max(np.abs(dt_series - median_dt)))
        raise ValueError(
            f"[Invariant] Temporal jitter detected! Maximum dt divergence from median ({median_dt:.4f}s) "
            f"is {max_dev:.6f}s."
        )

    # 3. Numerical Sanity
    if not np.isfinite(features).all():
        nans = int(np.isnan(features).sum())
        infs = int(np.isinf(features).sum())
        raise ValueError(
            f"[Numerical] Non-finite values detected in features (NaNs: {nans}, Infs: {infs})."
        )

    if not np.isfinite(timestamps).all():
        raise ValueError("[Numerical] Non-finite values (NaN/Inf) found in timestamps.")

    # 4. Statistical Sanity
    stds = np.std(features, axis=0)
    dead_cols = np.where(stds < 1e-8)[0]
    if len(dead_cols) > 0:
        raise ValueError(
            f"[Statistical] Dead (zero-variance) feature column(s) detected at indices: {dead_cols.tolist()}."
        )

    # 5. Feature-specific mathematical invariants

    feature_index = {
        name: index
        for index, name in enumerate(FEATURE_NAMES)
    }

    flow_mean = features[:, feature_index["flow_mean"]]
    flow_p90 = features[:, feature_index["flow_p90"]]
    flow_net = features[:, feature_index["flow_net"]]

    if np.any(flow_mean < 0):
        raise ValueError("[Invariant] flow_mean contains negative values.")

    if np.any(flow_p90 < 0):
        raise ValueError("[Invariant] flow_p90 contains negative values.")

    if np.any(flow_p90 < flow_mean):
        raise ValueError("[Invariant] flow_p90 is smaller than flow_mean.")

    if np.any(flow_net < 0):
        raise ValueError("[Invariant] flow_net contains negative values.")

    if np.any(flow_net > flow_mean):
        raise ValueError("[Invariant] flow_net is greater than flow_mean.")

    # Grid feature invariants

    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            prefix = f"grid_{row}_{col}"

            magnitude = features[
                :, feature_index[f"{prefix}_magnitude"]
            ]
            net = features[
                :, feature_index[f"{prefix}_net"]
            ]

            if np.any(magnitude < 0):
                raise ValueError(
                    f"[Invariant] {prefix}_magnitude contains negative values."
                )

            if np.any(net < 0):
                raise ValueError(
                    f"[Invariant] {prefix}_net contains negative values."
                )

            if np.any(net > magnitude):
                raise ValueError(
                    f"[Invariant] {prefix}_net is greater than "
                    f"{prefix}_magnitude."
                )
            
    # Structure feature invariants

    div_abs_p90 = features[:, feature_index["div_abs_p90"]]
    div_std = features[:, feature_index["div_std"]]
    div_pos_fraction = features[:, feature_index["div_pos_fraction"]]

    curl_abs_p90 = features[:, feature_index["curl_abs_p90"]]
    curl_std = features[:, feature_index["curl_std"]]

    if np.any(div_abs_p90 < 0):
        raise ValueError(
            "[Invariant] div_abs_p90 contains negative values."
        )

    if np.any(div_std < 0):
        raise ValueError(
            "[Invariant] div_std contains negative values."
        )

    if np.any(curl_abs_p90 < 0):
        raise ValueError(
            "[Invariant] curl_abs_p90 contains negative values."
        )

    if np.any(curl_std < 0):
        raise ValueError(
            "[Invariant] curl_std contains negative values."
        )

    if np.any((div_pos_fraction < 0) | (div_pos_fraction > 1)):
        raise ValueError(
            "[Invariant] div_pos_fraction contains values outside [0, 1]."
        )

    # Temporal feature invariants

    temporal_pairs = (
        ("delta_flow_u", "flow_u"),
        ("delta_flow_v", "flow_v"),
        ("delta_flow_mean", "flow_mean"),
        ("delta_flow_net", "flow_net"),
        ("delta_div_mean", "div_normalized_mean"),
    )

    for delta_name, base_name in temporal_pairs:
        delta = features[:, feature_index[delta_name]]
        base = features[:, feature_index[base_name]]

        # The first feature vector has no previous sample.
        if not np.isclose(delta[0], 0.0, rtol=0.0, atol=1e-6):
            raise ValueError(
                f"[Invariant] {delta_name} first value is not zero: "
                f"{delta[0]:.8f}."
            )

        expected_delta = np.diff(base)

        if not np.allclose(
            delta[1:],
            expected_delta,
            rtol=1e-5,
            atol=1e-6,
        ):
            differences = np.abs(delta[1:] - expected_delta)
            bad_index = int(np.argmax(differences)) + 1

            raise ValueError(
                f"[Invariant] {delta_name} does not match "
                f"consecutive {base_name} values at feature frame "
                f"{bad_index}: stored={delta[bad_index]:.8f}, "
                f"expected={expected_delta[bad_index - 1]:.8f}."
            )

# ============================================================================
# Main Script Execution
# ============================================================================

def parse_arguments() -> Path:
    parser = argparse.ArgumentParser(
        prog="audit_dataset",
        description="Data audit for the MTB autocut project",
        epilog="(c)2026 Anziano in (e)Bicicletta",
        usage="%(prog)s input [options]",
    )

    parser.add_argument("input", type=Path, help="training annotations file absolute path")
    args = parser.parse_args()

    if not args.input.exists():
        print(
            f"Error: input annotation file does not exist:\n{args.input}",
            file=sys.stderr,
        )
        sys.exit(1)

    return args.input


def main() -> int:
    annotation_path = parse_arguments()
    video_path, annotations = load_training_annotations(annotation_path)
    if not annotations:
        print("Error: annotation file contains no intervals.", file=sys.stderr)
        return 1

    clip_start = float(annotations[0].start)
    clip_end = float(annotations[-1].end)
    clip = Clip(
        video_name=video_path.stem,
        start=clip_start,
        end=clip_end,
    )
    print(
        f"Extracting/auditing canonical features for [{clip_start:.3f}, {clip_end:.3f}) "
        f"at {SAMPLE_FPS:g} sampled FPS and {PROCESSING_WIDTH}px processing width."
    )
    sequence = build_feature_sequence(
        video_path,
        clip,
        sample_fps=SAMPLE_FPS,
        processing_width=PROCESSING_WIDTH,
        cache_dir=Path("data/cache/features"),
    )
    fail_counter = 0
    audit_failures: list[tuple[int, str]] = []

    # --------------------------------------------------------------------
    # Run Sequence Audit
    # --------------------------------------------------------------------
    try:
        audit_feature_sequence(sequence)
        report_feature_statistics(sequence.features)
    except ValueError as err:
        fail_counter += 1
        print(f"  [AUDIT FAILED] {fail_counter:03d}: {err}", file=sys.stderr)
        audit_failures.append((fail_counter, str(err)))

    if audit_failures:
        print_fail(f"\n[SUMMARY] {len(audit_failures)} failures failed audit!")
        return 1
    
    print_pass(f"[SUMMARY] All feature sequence audits passed successfully!")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
