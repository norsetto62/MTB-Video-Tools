#!/usr/bin/env python3

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

from autocut.annotations import load_training_annotations
from autocut.data_models import FeatureSequence
from autocut.motion.features import FeatureExtractor, FEATURE_NAMES, GRID_ROWS, GRID_COLS
from autocut.motion.flow import calculate_optical_flow
from autocut.utils import convert_s_to_hms
from autocut.video.probe import get_video_info
from autocut.video.reader import VideoReader

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

    parser.add_argument("input", type=Path, help="annotations file absolute path")
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
    output_dir = Path("audit") / annotation_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Processing", annotation_path)

    video_path, annotations = load_training_annotations(annotation_path)
    video_info = get_video_info(video_path)

    # Calculate ROI spatial geometries
    processing_width = PROCESSING_WIDTH
    processing_height = round(video_info.height * processing_width / video_info.width)
    if processing_height % 2:
        processing_height += 1

    roi_x0 = round(processing_width * 0.15)
    roi_width = round(processing_width * 0.7)
    roi_y0 = round(processing_height * 0.1)
    roi_height = round(processing_height * 0.4)

    total_frames = 0
    tot_annotations = len(annotations)
    processing_start_wall = cv2.getTickCount()

    audit_failures: list[tuple[int, str]] = []

    for index, annotation in enumerate(annotations, start=1):
        filename = f"annotation_{index:03d}.npz"
        npz_path = output_dir / filename

        # --------------------------------------------------------------------
        # Cache Hit: Read sequence directly from .npz
        # --------------------------------------------------------------------
        if npz_path.exists():
            print(f"[{index}/{tot_annotations}] Loading cached NPZ: {npz_path.name}")
            with np.load(npz_path) as data:
                features = data["features"]
                timestamps = data["timestamps"]

            sequence = FeatureSequence(
                features=features,
                timestamps=timestamps,
            )
            total_frames += len(features)

        # --------------------------------------------------------------------
        # Cache Miss: Extract Optical Flow & Features from Video
        # --------------------------------------------------------------------
        else:
            progress = annotation.start * 100 / video_info.duration
            elapsed_wall = (
                cv2.getTickCount() - processing_start_wall
            ) / cv2.getTickFrequency()

            eta = (elapsed_wall * (100 - progress) / progress) if progress > 0 else 0
            print(
                f"[{index}/{tot_annotations}] Extracting video ({progress:5.2f}%) - "
                f"elapsed: {convert_s_to_hms(elapsed_wall)}, ETA: {convert_s_to_hms(eta)}"
            )

            features_list = []
            timestamps_list = []

            with VideoReader(
                video_path=video_path,
                start=annotation.start,
                duration=annotation.end - annotation.start,
                sample_fps=SAMPLE_FPS,
            ) as training_video:
                previous_gray = None
                extractor = FeatureExtractor(roi_width=roi_width, roi_height=roi_height)

                for frame_index, timestamp, frame in training_video:
                    frame_resized = cv2.resize(
                        frame,
                        (processing_width, processing_height),
                        interpolation=cv2.INTER_AREA,
                    )
                    gray_crop = frame_resized[
                        roi_y0 : roi_y0 + roi_height, roi_x0 : roi_x0 + roi_width
                    ]
                    current_gray = cv2.cvtColor(gray_crop, cv2.COLOR_BGR2GRAY)

                    if previous_gray is not None:
                        flow = calculate_optical_flow(
                            previous_gray=previous_gray,
                            current_gray=current_gray,
                        )
                        features_list.append(extractor.extract_vector(flow))
                        timestamps_list.append(timestamp)
                        total_frames += 1

                    previous_gray = current_gray

            features = np.asarray(features_list, dtype=np.float32)
            timestamps = np.asarray(timestamps_list, dtype=np.float64)

            # Cache arrays to disk
            np.savez(npz_path, features=features, timestamps=timestamps)

            sequence = FeatureSequence(
                features=features,
                timestamps=timestamps,
            )
        
            elapsed_wall = (
                cv2.getTickCount() - processing_start_wall
            ) / cv2.getTickFrequency()

            print("\n" + "=" * 60)
            print(f"Processed {total_frames} total feature frames across {tot_annotations} annotations.")
            print(f"Total time: {convert_s_to_hms(elapsed_wall)}")
            print(f"Data directory: {output_dir}")

        # --------------------------------------------------------------------
        # Run Sequence Audit
        # --------------------------------------------------------------------
        try:
            audit_feature_sequence(sequence)
        except ValueError as err:
            print(f"  [AUDIT FAILED] Annotation {index:03d}: {err}", file=sys.stderr)
            audit_failures.append((index, str(err)))

    if audit_failures:
        print_fail(f"\n[SUMMARY] {len(audit_failures)} / {tot_annotations} annotations failed audit!")
        return 1
    
    print_pass(f"[SUMMARY] All feature sequence audits passed successfully!")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())