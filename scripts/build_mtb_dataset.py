#!/usr/bin/env python3
"""
Build temporal MTB-interest datasets from manual annotations.

Annotation format
-----------------

The annotation file contains the source video path on one line followed by:

    Start    End    MTB    Remarks

MTB is an interest score from 0 to 3:

    0 = not interesting
    1 = mildly interesting
    2 = interesting
    3 = very interesting

Example:

    D:\\Prenestini\\Mentorella\\DJI_202507101131.mp4

    Start    End    MTB    Remarks
    00:10    00:18   0     easy trail
    01:23    01:31   3     drop
    02:14    02:29   2     rock garden

The script operates on an optical-flow CSV produced by
analyze_bike_flow.py.

Normal mode
-----------

Creates:

    <output-dir>/<name>_dataset.npz
    <output-dir>/<name>_dataset_manifest.csv

The NPZ contains:

    X              [N, T, F] float32
    y              [N]       float32
    feature_names  [F]
    labels         ["interest"]

Each training window receives an interest target obtained from the
manual annotations covering that window.

Windows which are not covered by an annotation are NOT used.

The flow CSV timestamps follow analyze_bike_flow.py semantics: each
timestamp is the END of the frame-to-frame interval used to calculate
that row's optical-flow features. The generator therefore derives the
sampling interval from the CSV timestamps and maps N selected rows to
the represented interval of N sampling periods.

If a window overlaps more than one annotation, its target is the
duration-weighted mean of the annotation scores.

Interest-report mode
--------------------

    --interest-report

does not create a dataset. It reports:

    * annotation coverage
    * score distribution
    * generated windows
    * duration-weighted targets
    * windows near annotation boundaries

This is intended for inspecting the dataset before training.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


DEFAULT_WINDOW = 4.0
DEFAULT_STRIDE = 2.0
DEFAULT_DT_TOLERANCE = 0.15


@dataclass(frozen=True)
class Annotation:
    start: float
    end: float
    score: int
    remarks: str


def parse_time(value: str) -> float:
    """Convert seconds, MM:SS or HH:MM:SS to seconds."""

    value = value.strip()

    if not value:
        raise ValueError("empty time value")

    parts = value.split(":")

    try:
        if len(parts) == 1:
            return float(parts[0])

        if len(parts) == 2:
            minutes = float(parts[0])
            seconds = float(parts[1])
            return minutes * 60.0 + seconds

        if len(parts) == 3:
            hours = float(parts[0])
            minutes = float(parts[1])
            seconds = float(parts[2])
            return hours * 3600.0 + minutes * 60.0 + seconds

    except ValueError as exc:
        raise ValueError(f"invalid time: {value!r}") from exc

    raise ValueError(f"invalid time: {value!r}")


def is_windows_path(value: str) -> bool:
    """Return True for a normal Windows absolute path."""

    return (
        len(value) >= 3
        and value[1] == ":"
        and value[2] in ("\\", "/")
    )


def load_annotations(
    path: Path,
) -> tuple[Path, list[Annotation]]:
    """
    Load the current simplified annotation format.

    Expected:

        source-video-path

        Start End MTB Remarks

    Remarks are optional.
    """

    lines = path.read_text(encoding="utf-8-sig").splitlines()

    video_path: Path | None = None
    annotations: list[Annotation] = []

    for line_no, raw in enumerate(lines, start=1):
        line = raw.strip()

        if not line or line.startswith("#"):
            continue

        lower = line.lower()

        # Header.
        if "start" in lower and "end" in lower and "mtb" in lower:
            continue

        # Source video path.
        if video_path is None and is_windows_path(line):
            video_path = Path(line)
            continue

        parts = line.split(maxsplit=3)

        if len(parts) < 3:
            raise ValueError(
                f"{path}:{line_no}: expected "
                "Start End MTB [Remarks]"
            )

        try:
            start = parse_time(parts[0])
            end = parse_time(parts[1])
            score = int(parts[2])
        except ValueError as exc:
            raise ValueError(
                f"{path}:{line_no}: {exc}"
            ) from exc

        if end <= start:
            raise ValueError(
                f"{path}:{line_no}: End must be greater than Start"
            )

        if not 0 <= score <= 3:
            raise ValueError(
                f"{path}:{line_no}: MTB score must be 0..3"
            )

        remarks = parts[3].strip() if len(parts) == 4 else ""

        annotations.append(
            Annotation(
                start=start,
                end=end,
                score=score,
                remarks=remarks,
            )
        )

    if video_path is None:
        raise ValueError(
            f"{path}: source video path not found"
        )

    annotations.sort(key=lambda annotation: annotation.start)

    # Overlapping annotations make the meaning of the manual labels
    # ambiguous. Do not silently accept them.
    previous: Annotation | None = None

    for annotation in annotations:
        if (
            previous is not None
            and annotation.start < previous.end - 1e-9
        ):
            raise ValueError(
                f"{path}: overlapping annotations: "
                f"{previous.start:.3f}-{previous.end:.3f}s and "
                f"{annotation.start:.3f}-{annotation.end:.3f}s"
            )

        previous = annotation

    return video_path, annotations


def load_flow_csv(
    path: Path,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """
    Load an analyze_bike_flow.py CSV.

    Returns:

        timestamps
        feature_names
        feature_matrix
    """

    with path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise ValueError(
                f"{path}: missing CSV header"
            )

        fields = [
            field.strip()
            for field in reader.fieldnames
        ]

        if "time" in fields:
            time_field = "time"
        elif "timestamp" in fields:
            time_field = "timestamp"
        else:
            raise ValueError(
                f"{path}: missing time column"
            )

        feature_names = [
            field
            for field in fields
            if field != time_field
        ]

        if not feature_names:
            raise ValueError(
                f"{path}: no feature columns"
            )

        timestamps: list[float] = []
        rows: list[list[float]] = []

        for line_no, row in enumerate(reader, start=2):
            try:
                timestamp = float(row[time_field])

                values = [
                    float(row[name])
                    for name in feature_names
                ]

            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{path}:{line_no}: "
                    f"non-numeric value: {exc}"
                ) from exc

            if (
                not np.isfinite(timestamp)
                or not np.all(np.isfinite(values))
            ):
                raise ValueError(
                    f"{path}:{line_no}: "
                    "NaN/Inf values are not allowed"
                )

            timestamps.append(timestamp)
            rows.append(values)

    if len(timestamps) < 2:
        raise ValueError(
            f"{path}: need at least two rows"
        )

    timestamps_array = np.asarray(
        timestamps,
        dtype=np.float64,
    )

    features = np.asarray(
        rows,
        dtype=np.float32,
    )

    return timestamps_array, feature_names, features


def check_sampling(
    timestamps: np.ndarray,
    tolerance: float,
) -> float:
    """Check flow sampling and return median timestep."""

    dt = np.diff(timestamps)

    if np.any(dt <= 0):
        raise ValueError(
            "flow timestamps must be strictly increasing"
        )

    median_dt = float(np.median(dt))

    if median_dt <= 0:
        raise ValueError(
            "invalid flow sampling interval"
        )

    relative_error = np.abs(
        dt - median_dt
    ) / median_dt

    maximum_error = float(
        np.max(relative_error)
    )

    if maximum_error > tolerance:
        print(
            "WARNING: non-uniform flow sampling: "
            f"median dt={median_dt:.6f}s, "
            f"maximum relative deviation="
            f"{maximum_error:.3f}"
        )

    return median_dt


def annotation_overlap(
    window_start: float,
    window_end: float,
    annotation: Annotation,
) -> float:
    """Return overlap duration in seconds."""

    return max(
        0.0,
        min(window_end, annotation.end)
        - max(window_start, annotation.start),
    )


def build_windows(
    timestamps: np.ndarray,
    features: np.ndarray,
    annotations: list[Annotation],
    window: float,
    stride: float,
    flow_start: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """
    Generate fixed-size temporal windows.

    A window is included only if it has at least some annotation coverage.

    If multiple annotations overlap a window, the target is their
    duration-weighted mean score.
    """

    if window <= 0:
        raise ValueError("window must be > 0")

    if stride <= 0:
        raise ValueError("stride must be > 0")

    if len(timestamps) != len(features):
        raise ValueError(
            "timestamps and feature matrix have different lengths"
        )

    # analyze_bike_flow.py writes the END timestamp of each
    # frame-to-frame flow interval. For example, at 2 FPS:
    #
    #   time=0.5 -> flow over [0.0, 0.5)
    #   time=1.0 -> flow over [0.5, 1.0)
    #
    # Therefore N rows represent N * dt seconds, beginning one dt
    # before the first row timestamp. Derive dt from the actual CSV
    # timestamps; never assume a hard-coded FPS.
    dt = float(
        np.median(np.diff(timestamps))
    )

    if dt <= 0:
        raise ValueError(
            "invalid flow sampling interval"
        )
    
    window_samples_float = window / dt
    stride_samples_float = stride / dt

    samples_per_window = int(round(window_samples_float))
    samples_per_stride = int(round(stride_samples_float))

    # Do not silently change the requested temporal geometry by rounding
    # window/stride to a different number of flow samples.
    if not np.isclose(
        window_samples_float,
        samples_per_window,
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError(
            f"window={window:.6f}s is not an integer multiple of "
            f"flow dt={dt:.6f}s"
        )

    if not np.isclose(
        stride_samples_float,
        samples_per_stride,
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError(
            f"stride={stride:.6f}s is not an integer multiple of "
            f"flow dt={dt:.6f}s"
        )
    
    if samples_per_window < 2:
        raise ValueError(
            f"window={window:.3f}s is too short "
            f"for flow dt={dt:.6f}s"
        )

    if samples_per_stride < 1:
        raise ValueError(
            f"stride={stride:.3f}s is too short "
            f"for flow dt={dt:.6f}s"
        )

    examples: list[np.ndarray] = []
    targets: list[float] = []
    manifest: list[dict] = []    # The flow CSV can end at slightly different times at different
    # FPS because the final frame-to-frame interval is quantized
    # differently.  Do not let that determine the training-window
    # grid.
    #
    # The represented flow interval is:
    #
    #   [flow_start, flow_start + len(timestamps) * dt)
    #
    # We deliberately truncate the usable duration to the largest
    # complete stride boundary.  This makes independently generated
    # datasets at different FPS use the same canonical window grid.
    represented_duration = (
        len(timestamps) * dt
    )

    canonical_duration = (
        np.floor(
            (represented_duration + 1e-9) / stride
        )
        * stride
    )
    flow_relative_start = float(timestamps[0] - dt)
    flow_relative_end = float(timestamps[-1])

    represented_duration = (
        flow_relative_end - flow_relative_start
    )

    max_start_offset = (
        represented_duration - window
    )

    if max_start_offset < -1e-9:
        raise ValueError(
            f"flow duration={represented_duration:.6f}s is shorter "
            f"than window={window:.6f}s"
        )

    # Number of complete canonical windows.
    n_windows = (
        int(
            np.floor(
                (max_start_offset + 1e-9) / stride
            )
        )
        + 1
    )

    for window_number in range(n_windows):
        index = (
            window_number
            * samples_per_stride
        )

        sample_indices = np.arange(
            index,
            index + samples_per_window,
        )

        # Window times come from the canonical temporal grid, NOT
        # from floating-point arithmetic on CSV timestamps.
        #
        # The first flow row represents [flow_start, flow_start+dt),
        # so row index i represents a window starting at
        # flow_start + i*dt.
        video_start = (
            flow_start
            + index * dt
        )

        video_end = (
            video_start + window
        )

        covered_duration = 0.0
        weighted_score = 0.0

        matching_annotations: list[
            tuple[int, Annotation, float]
        ] = []

        for annotation_id, annotation in enumerate(
            annotations,
            start=1,
        ):
            overlap = annotation_overlap(
                video_start,
                video_end,
                annotation,
            )

            if overlap <= 0:
                continue

            matching_annotations.append(
                (
                    annotation_id,
                    annotation,
                    overlap,
                )
            )

            covered_duration += overlap
            weighted_score += (
                overlap * annotation.score
            )

        if covered_duration > 0:
            target = (
                weighted_score
                / covered_duration
            )

            annotation_ids = [
                str(item[0])
                for item in matching_annotations
            ]

            scores = [
                str(item[1].score)
                for item in matching_annotations
            ]

            remarks = [
                item[1].remarks
                for item in matching_annotations
                if item[1].remarks
            ]

            examples.append(
                features[sample_indices]
            )

            targets.append(target)

            manifest.append(
                {
                    "example_id": len(manifest),
                    "start": video_start,
                    "end": video_end,
                    "duration": (
                        video_end - video_start
                    ),
                    "target": target,
                    "annotation_id": ";".join(
                        annotation_ids
                    ),
                    "annotation_scores": ";".join(
                        scores
                    ),
                    "covered_duration": covered_duration,
                    "coverage": (
                        covered_duration / window
                    ),
                    "remarks": " | ".join(remarks),
                    "n_samples": samples_per_window,
                }
            )

    if not examples:
        raise ValueError(
            "no annotated training windows were generated"
        )

    X = np.stack(examples).astype(
        np.float32
    )

    y = np.asarray(
        targets,
        dtype=np.float32,
    )

    return X, y, manifest


def format_clock(seconds: float) -> str:
    """Format seconds as MM:SS or HH:MM:SS."""

    total = max(
        0,
        int(round(seconds)),
    )

    hours, remainder = divmod(
        total,
        3600,
    )

    minutes, seconds = divmod(
        remainder,
        60,
    )

    if hours:
        return (
            f"{hours:02d}:"
            f"{minutes:02d}:"
            f"{seconds:02d}"
        )

    return (
        f"{minutes:02d}:"
        f"{seconds:02d}"
    )


def write_manifest(
    path: Path,
    manifest: list[dict],
) -> None:
    """Write the generated-window manifest."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fields = [
        "example_id",
        "start",
        "end",
        "duration",
        "target",
        "annotation_id",
        "annotation_scores",
        "covered_duration",
        "coverage",
        "remarks",
        "n_samples",
    ]

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(manifest)


def write_example_audit(
    path: Path,
    manifest: list[dict],
) -> None:
    """Write one inspectable row per generated example."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fields = [
        "example_id",
        "start",
        "end",
        "duration",
        "target",
        "expected_target",
        "target_error",
        "annotation_id",
        "expected_annotation_id",
        "annotation_scores",
        "expected_annotation_scores",
        "covered_duration",
        "expected_covered_duration",
        "coverage",
        "expected_coverage",
        "duration_error",
        "coverage_error",
        "remarks",
        "n_samples",
        "sample_count_expected",
        "sample_count_ok",
        "annotation_ids_ok",
        "target_ok",
        "valid",
    ]

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )

        writer.writeheader()

        for i, row in enumerate(manifest):
            start = float(row["start"])
            end = float(row["end"])
            target = float(row["target"])

            output = dict(row)
            writer.writerow(output)


def validate_generated_examples(
    manifest: list[dict],
    annotations: list[Annotation],
    window: float,
    flow_dt: float,
    tolerance: float = 1e-6,
) -> tuple[list[dict], int]:
    """
    Independently validate every generated example against the annotations.

    This deliberately does not call build_windows(). It reconstructs the
    expected annotation coverage and duration-weighted target directly from
    each generated window's start/end timestamps.
    """

    expected_samples = int(round(window / flow_dt))

    if expected_samples < 1:
        raise ValueError(
            f"invalid expected sample count for window={window:.6f}s "
            f"and flow dt={flow_dt:.6f}s"
        )

    validated = []
    invalid_count = 0

    for row in manifest:
        start = float(row["start"])
        end = float(row["end"])
        duration = float(row["duration"])
        target = float(row["target"])
        covered_duration = float(row["covered_duration"])
        coverage = float(row["coverage"])
        n_samples = int(row["n_samples"])

        expected_matches = []
        expected_covered = 0.0
        expected_weighted_score = 0.0

        for annotation_id, annotation in enumerate(
            annotations,
            start=1,
        ):
            overlap = annotation_overlap(
                start,
                end,
                annotation,
            )

            if overlap <= 0:
                continue

            expected_matches.append(
                (annotation_id, annotation, overlap)
            )

            expected_covered += overlap
            expected_weighted_score += (
                overlap * annotation.score
            )

        expected_annotation_ids = ";".join(
            str(item[0])
            for item in expected_matches
        )

        expected_annotation_scores = ";".join(
            str(item[1].score)
            for item in expected_matches
        )

        expected_target = (
            expected_weighted_score / expected_covered
            if expected_covered > 0
            else float("nan")
        )

        expected_coverage = (
            expected_covered / duration
            if duration > 0
            else float("nan")
        )

        expected_duration = end - start

        duration_error = abs(
            duration - expected_duration
        )

        covered_duration_error = abs(
            covered_duration - expected_covered
        )

        coverage_error = (
            abs(coverage - expected_coverage)
            if np.isfinite(expected_coverage)
            else float("inf")
        )

        target_error = (
            abs(target - expected_target)
            if np.isfinite(expected_target)
            else float("inf")
        )

        generated_annotation_ids = str(
            row["annotation_id"]
        )

        generated_annotation_scores = str(
            row["annotation_scores"]
        )

        expected_duration_from_samples = (
            expected_samples * flow_dt
        )

        geometry_error = abs(
            expected_duration_from_samples
            - expected_duration
        )

        checks = {
            "duration_ok": duration_error <= tolerance,
            "sample_count_ok": (
                n_samples == expected_samples
            ),
            "sample_geometry_ok": (
                geometry_error <= tolerance
            ),
            "annotation_ids_ok": (
                generated_annotation_ids
                == expected_annotation_ids
            ),
            "annotation_scores_ok": (
                generated_annotation_scores
                == expected_annotation_scores
            ),
            "covered_duration_ok": (
                covered_duration_error <= tolerance
            ),
            "coverage_ok": (
                coverage_error <= tolerance
            ),
            "target_ok": (
                target_error <= tolerance
            ),
            "target_range_ok": (
                -tolerance <= target <= 3.0 + tolerance
            ),
        }

        valid = all(checks.values())

        if not valid:
            invalid_count += 1

        output = dict(row)

        output.update(
            {
                "expected_target": (
                    expected_target
                    if np.isfinite(expected_target)
                    else ""
                ),
                "target_error": target_error,
                "expected_annotation_id": (
                    expected_annotation_ids
                ),
                "expected_annotation_scores": (
                    expected_annotation_scores
                ),
                "expected_covered_duration": (
                    expected_covered
                ),
                "expected_coverage": (
                    expected_coverage
                    if np.isfinite(expected_coverage)
                    else ""
                ),
                "duration_error": duration_error,
                "coverage_error": coverage_error,
                "sample_count_expected": expected_samples,
                "sample_count_ok": int(
                    checks["sample_count_ok"]
                ),
                "annotation_ids_ok": int(
                    checks["annotation_ids_ok"]
                ),
                "target_ok": int(
                    checks["target_ok"]
                ),
                "valid": int(valid),
            }
        )

        validated.append(output)

    return validated, invalid_count


def print_annotation_summary(
    annotations: list[Annotation],
) -> None:
    """Print the manual annotation distribution."""

    print()
    print("=" * 80)
    print("ANNOTATIONS")
    print("=" * 80)

    total_duration = sum(
        annotation.end - annotation.start
        for annotation in annotations
    )

    print(
        f"Count:           {len(annotations)}"
    )

    print(
        f"Annotated time:  {total_duration:.1f}s"
    )

    for score in range(4):
        duration = sum(
            annotation.end - annotation.start
            for annotation in annotations
            if annotation.score == score
        )

        percentage = (
            100.0 * duration / total_duration
            if total_duration > 0
            else 0.0
        )

        print(
            f"MTB {score}: "
            f"{duration:7.1f}s "
            f"({percentage:5.1f}%)"
        )


def print_dataset_summary(
    X: np.ndarray,
    y: np.ndarray,
    manifest: list[dict],
) -> None:
    """Print a concise generated-dataset summary."""

    print()
    print("=" * 80)
    print("DATASET")
    print("=" * 80)

    print(
        f"Examples:        {len(manifest)}"
    )

    print(
        f"X shape:          {X.shape}"
    )

    print(
        f"y shape:          {y.shape}"
    )

    print(
        f"Target range:     "
        f"{float(np.min(y)):.3f} .. "
        f"{float(np.max(y)):.3f}"
    )

    print(
        f"Target mean:      "
        f"{float(np.mean(y)):.3f}"
    )

    for score in range(4):
        count = int(
            np.sum(
                np.isclose(
                    y,
                    score,
                    atol=1e-6,
                )
            )
        )

        print(
            f"Exact MTB {score}:  "
            f"{count:5d} windows"
        )


def print_preview(
    manifest: list[dict],
    count: int,
) -> None:
    """Print generated windows for inspection."""

    print()
    print("=" * 80)
    print("GENERATED WINDOWS")
    print("=" * 80)

    print(
        "id       interval       target   coverage   annotations   remarks"
    )

    print("-" * 80)

    for row in manifest[:count]:
        interval = (
            f"{format_clock(float(row['start']))}-"
            f"{format_clock(float(row['end']))}"
        )

        print(
            f"{int(row['example_id']):3d}  "
            f"{interval:>17s}  "
            f"{float(row['target']):7.3f}  "
            f"{float(row['coverage']):8.1%}  "
            f"{row['annotation_id']:>12s}   "
            f"{row['remarks']}"
        )

    if len(manifest) > count:
        print(
            f"... {len(manifest) - count} more windows"
        )

def print_audit_transition_windows(
    manifest: list[dict],
    annotations: list[Annotation],
    preview_context: float,
) -> None:
    """Print generated windows surrounding every annotation boundary."""

    print()
    print("=" * 80)
    print("ANNOTATION TRANSITIONS")
    print("=" * 80)

    boundaries: list[tuple[float, str]] = []

    for index in range(len(annotations) - 1):
        left = annotations[index]
        right = annotations[index + 1]

        if left.end <= right.start:
            boundaries.append(
                (
                    right.start,
                    f"MTB {left.score} -> MTB {right.score}",
                )
            )

    for annotation in annotations:
        boundaries.append(
            (
                annotation.start,
                f"start MTB {annotation.score}",
            )
        )

        boundaries.append(
            (
                annotation.end,
                f"end MTB {annotation.score}",
            )
        )

    # Avoid printing duplicate boundaries at exactly the same timestamp.
    unique_boundaries: dict[float, list[str]] = {}

    for timestamp, description in boundaries:
        unique_boundaries.setdefault(
            timestamp,
            [],
        ).append(description)

    for timestamp in sorted(unique_boundaries):
        descriptions = unique_boundaries[timestamp]

        print()
        print(
            f"{format_clock(timestamp)} "
            f"({timestamp:.3f}s): "
            f"{'; '.join(descriptions)}"
        )

        nearby = [
            row
            for row in manifest
            if (
                float(row["start"]) <= timestamp + preview_context
                and float(row["end"]) >= timestamp - preview_context
            )
        ]

        if not nearby:
            print("  NO GENERATED WINDOWS")
            continue

        for row in nearby:
            print(
                f"  "
                f"{format_clock(float(row['start']))}-"
                f"{format_clock(float(row['end']))}  "
                f"target={float(row['target']):.3f}  "
                f"coverage={float(row['coverage']):.1%}  "
                f"annotations={row['annotation_id']}"
            )


def print_annotation_coverage_audit(
    annotations: list[Annotation],
) -> None:
    """Report gaps and overlaps between annotation intervals."""

    print()
    print("=" * 80)
    print("ANNOTATION CONTINUITY")
    print("=" * 80)

    if not annotations:
        print("No annotations.")
        return

    gaps: list[tuple[float, float]] = []
    overlaps: list[tuple[float, float]] = []

    for previous, current in zip(
        annotations,
        annotations[1:],
    ):
        delta = current.start - previous.end

        if delta > 1e-9:
            gaps.append(
                (
                    previous.end,
                    current.start,
                )
            )
        elif delta < -1e-9:
            overlaps.append(
                (
                    current.start,
                    previous.end,
                )
            )

    if gaps:
        print(f"GAPS: {len(gaps)}")
        for start, end in gaps:
            print(
                f"  {format_clock(start)}-"
                f"{format_clock(end)} "
                f"({end - start:.3f}s)"
            )
    else:
        print("GAPS:     none")

    if overlaps:
        print(f"OVERLAPS: {len(overlaps)}")
        for start, end in overlaps:
            print(
                f"  {format_clock(start)}-"
                f"{format_clock(end)} "
                f"({end - start:.3f}s)"
            )
    else:
        print("OVERLAPS: none")

def audit_pair(
    annotations_path: Path,
    flow_csv: Path,
    window: float,
    stride: float,
    flow_start: float,
    dt_tolerance: float,
    preview: int,
) -> tuple[int, dict]:
    """Audit one annotation/flow pair without writing dataset files."""

    video_path, annotations = load_annotations(
        annotations_path
    )

    timestamps, feature_names, features = load_flow_csv(
        flow_csv
    )

    median_dt = check_sampling(
        timestamps,
        dt_tolerance,
    )

    X, y, manifest = build_windows(
        timestamps,
        features,
        annotations,
        window,
        stride,
        flow_start,
    )

    # Flow timestamps are END timestamps. The represented flow
    # therefore starts one sampling interval before the first row.
    flow_start_time = float(
        timestamps[0] - median_dt + flow_start
    )

    flow_end_time = float(
        timestamps[-1] + flow_start
    )

    annotated_start = min(
        annotation.start
        for annotation in annotations
    )

    annotated_end = max(
        annotation.end
        for annotation in annotations
    )

    print()
    print("#" * 80)
    print(
        f"AUDIT: {annotations_path.stem}"
    )
    print("#" * 80)

    print(
        f"Source video: {video_path}"
    )

    print(
        f"Annotations:  {annotations_path}"
    )

    print(
        f"Flow CSV:     {flow_csv}"
    )

    print(
        f"Flow range:   "
        f"{flow_start_time:.3f}s - "
        f"{flow_end_time:.3f}s"
    )

    print(
        f"Flow rows:    {len(timestamps)}"
    )

    print(
        f"Features:     {len(feature_names)}"
    )

    print(
        f"Median dt:    {median_dt:.6f}s"
    )

    print(
        f"Window:       {window:.2f}s"
    )

    print(
        f"Stride:       {stride:.2f}s"
    )

    print_annotation_summary(
        annotations
    )

    print_annotation_coverage_audit(
        annotations
    )

    print()
    print("=" * 80)
    print("GENERATED DATASET")
    print("=" * 80)

    print(
        f"Examples:     {len(manifest)}"
    )

    print(
        f"X shape:      {X.shape}"
    )

    print(
        f"Target range: "
        f"{float(np.min(y)):.3f} .. "
        f"{float(np.max(y)):.3f}"
    )

    print(
        f"Target mean:  "
        f"{float(np.mean(y)):.3f}"
    )

    intermediate = int(
        np.sum(
            ~np.isclose(y, np.round(y), atol=1e-6)
        )
    )

    print(
        f"Intermediate targets: "
        f"{intermediate}"
    )

    for score in range(4):
        count = int(
            np.sum(
                np.isclose(
                    y,
                    score,
                    atol=1e-6,
                )
            )
        )

        print(
            f"Exact target {score}: "
            f"{count}"
        )

    print_preview(
        manifest,
        preview,
    )

    print_audit_transition_windows(
        manifest,
        annotations,
        preview_context=max(
            window,
            stride * 2.0,
        ),
    )

    print()
    print(
        f"Annotation range: "
        f"{annotated_start:.3f}s - "
        f"{annotated_end:.3f}s"
    )

    audit_path = Path(
        "output/datasets"
    ) / f"{annotations_path.stem.lower()}_dataset_audit.csv"

    validated_manifest, validation_failures = (
        validate_generated_examples(
            manifest=manifest,
            annotations=annotations,
            window=window,
            flow_dt=median_dt,
        )
    )

    write_example_audit(
        audit_path,
        validated_manifest,
    )

    print(
        f"Validation:   "
        f"{len(manifest) - validation_failures} valid / "
        f"{validation_failures} invalid"
    )

    print(
        f"Audit CSV:   {audit_path}"
    )

    if validation_failures:
        raise ValueError(
            f"{annotations_path}: "
            f"{validation_failures} generated examples failed "
            "independent validation"
        )
    
    return len(manifest), {
        "examples": len(manifest),
        "targets": y,
        "annotations": annotations,
    }

def audit_pairs(
    pairs: list[tuple[Path, Path]],
    window: float,
    stride: float,
    flow_start: float,
    dt_tolerance: float,
    preview: int,
) -> None:
    """Audit multiple annotation/flow pairs and print global totals."""

    total_examples = 0
    total_annotations = 0
    total_duration = np.zeros(
        4,
        dtype=np.float64,
    )
    total_exact_windows = np.zeros(
        4,
        dtype=np.int64,
    )
    total_intermediate = 0

    print()
    print("#" * 80)
    print("MTB DATASET AUDIT")
    print("#" * 80)
    print(
        f"Pairs: {len(pairs)}"
    )
    print(
        f"Window: {window:.2f}s"
    )
    print(
        f"Stride: {stride:.2f}s"
    )

    for annotations_path, flow_csv in pairs:
        examples, result = audit_pair(
            annotations_path=annotations_path,
            flow_csv=flow_csv,
            window=window,
            stride=stride,
            flow_start=flow_start,
            dt_tolerance=dt_tolerance,
            preview=preview,
        )

        total_examples += examples

        annotations = result["annotations"]
        targets = result["targets"]

        total_annotations += len(annotations)

        for annotation in annotations:
            total_duration[annotation.score] += (
                annotation.end - annotation.start
            )

        for score in range(4):
            total_exact_windows[score] += int(
                np.sum(
                    np.isclose(
                        targets,
                        score,
                        atol=1e-6,
                    )
                )
            )

        total_intermediate += int(
            np.sum(
                ~np.isclose(
                    targets,
                    np.round(targets),
                    atol=1e-6,
                )
            )

        )

    total_annotated_duration = float(
        np.sum(total_duration)
    )

    print()
    print("#" * 80)
    print("GLOBAL AUDIT")
    print("#" * 80)

    print(
        f"Pairs:             {len(pairs)}"
    )

    print(
        f"Annotations:       {total_annotations}"
    )

    print(
        f"Generated windows: {total_examples}"
    )

    print(
        f"Annotated time:    "
        f"{total_annotated_duration:.1f}s"
    )

    print()

    for score in range(4):
        percentage = (
            100.0
            * total_duration[score]
            / total_annotated_duration
            if total_annotated_duration > 0
            else 0.0
        )

        print(
            f"MTB {score}: "
            f"{total_duration[score]:7.1f}s "
            f"({percentage:5.1f}%)"
        )

    print()
    print("Generated targets:")

    for score in range(4):
        print(
            f"  Exact {score}: "
            f"{total_exact_windows[score]}"
        )

    print(
        f"  Intermediate: "
        f"{total_intermediate}"
    )

    print()
    print("AUDIT COMPLETE")

def print_interest_report(
    flow_csv: Path,
    annotations_path: Path,
    window: float,
    stride: float,
    flow_start: float,
    preview: int,
) -> None:
    """
    Report how the current 0..3 annotations translate into temporal windows.

    No dataset files are written.
    """

    video_path, annotations = load_annotations(
        annotations_path
    )

    timestamps, feature_names, features = (
        load_flow_csv(flow_csv)
    )

    median_dt = check_sampling(
        timestamps,
        DEFAULT_DT_TOLERANCE,
    )

    X, y, manifest = build_windows(
        timestamps,
        features,
        annotations,
        window,
        stride,
        flow_start,
    )

    print()
    print("#" * 80)
    print(
        f"MTB INTEREST REPORT: "
        f"{annotations_path.stem}"
    )
    print("#" * 80)

    print(
        f"Source video: {video_path}"
    )

    print(
        f"Flow CSV:     {flow_csv}"
    )

    print(
        f"Flow range:   "
        f"{format_clock(float(timestamps[0] - median_dt + flow_start))}"
        f" - "
        f"{format_clock(float(timestamps[-1] + flow_start))}"
    )

    print(
        f"Flow rows:    {len(timestamps)}"
    )

    print(
        f"Features:     {len(feature_names)}"
    )

    print(
        f"Median dt:    {median_dt:.6f}s"
    )

    print(
        f"Window:       {window:.2f}s"
    )

    print(
        f"Stride:       {stride:.2f}s"
    )

    print_annotation_summary(
        annotations
    )

    print()
    print("=" * 80)
    print("WINDOW COVERAGE")
    print("=" * 80)

    # Include the interval represented by the first flow row.
    total_flow_duration = (
        float(timestamps[-1] - timestamps[0] + median_dt)
    )

    annotated_duration = sum(
        annotation.end - annotation.start
        for annotation in annotations
    )

    generated_duration = sum(
        float(row["duration"])
        for row in manifest
    )

    print(
        f"Flow duration:          "
        f"{total_flow_duration:.1f}s"
    )

    print(
        f"Annotated duration:     "
        f"{annotated_duration:.1f}s"
    )

    print(
        f"Generated-window time:  "
        f"{generated_duration:.1f}s"
    )

    print(
        f"Generated windows:      "
        f"{len(manifest)}"
    )

    print(
        f"Target mean:             "
        f"{float(np.mean(y)):.3f}"
    )

    print_preview(
        manifest,
        preview,
    )

    print()
    print("=" * 80)
    print("BOUNDARY WINDOWS")
    print("=" * 80)

    boundary_windows: list[dict] = []

    for row in manifest:
        start = float(
            row["start"]
        )

        end = float(
            row["end"]
        )

        near_boundary = any(
            abs(start - annotation.start)
            <= stride + 1e-9
            or
            abs(end - annotation.end)
            <= stride + 1e-9
            for annotation in annotations
        )

        if near_boundary:
            boundary_windows.append(row)

    print(
        "Windows within one stride of an "
        f"annotation boundary: "
        f"{len(boundary_windows)}"
    )

    for row in boundary_windows[:preview]:
        print(
            f"{format_clock(float(row['start']))}-"
            f"{format_clock(float(row['end']))}  "
            f"target={float(row['target']):.3f}  "
            f"coverage={float(row['coverage']):.1%}  "
            f"annotations={row['annotation_id']}  "
            f"{row['remarks']}"
        )

    if len(boundary_windows) > preview:
        print(
            f"... {len(boundary_windows) - preview} more"
        )

    print()
    print("=" * 80)
    print("REPORT COMPLETE")
    print("=" * 80)


def build_dataset(
    flow_csv: Path,
    annotations_path: Path,
    output_dir: Path,
    window: float,
    stride: float,
    flow_start: float,
    dt_tolerance: float,
    preview: int,
) -> None:
    """Build and save the NPZ dataset."""

    video_path, annotations = load_annotations(
        annotations_path
    )

    timestamps, feature_names, features = (
        load_flow_csv(flow_csv)
    )

    median_dt = check_sampling(
        timestamps,
        dt_tolerance,
    )

    X, y, manifest = build_windows(
        timestamps,
        features,
        annotations,
        window,
        stride,
        flow_start,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    stem = annotations_path.stem.lower()

    npz_path = (
        output_dir
        / f"{stem}_dataset.npz"
    )

    manifest_path = (
        output_dir
        / f"{stem}_dataset_manifest.csv"
    )

    metadata = {
        "source_annotations": str(annotations_path),
        "source_flow_csv": str(flow_csv),
        "window": window,
        "stride": stride,
        "flow_start": flow_start,
        "flow_dt": median_dt,
    }

    np.savez_compressed(
        npz_path,
        X=X,
        y=y,
        feature_names=np.asarray(
            feature_names
        ),
        labels=np.asarray(
            ["interest"]
        ),
        metadata=np.asarray(metadata, dtype=object),
    )

    _, validation_failures = validate_generated_examples(
        manifest=manifest,
        annotations=annotations,
        window=window,
        flow_dt=median_dt,
    )

    if validation_failures:
        raise ValueError(
            f"{annotations_path}: "
            f"{validation_failures} generated examples failed "
            "independent validation"
        )

    write_manifest(
        manifest_path,
        manifest,
    )

    print()
    print("=" * 80)
    print("BUILD COMPLETE")
    print("=" * 80)

    print(
        f"Source video: {video_path}"
    )

    print(
        f"Flow CSV:     {flow_csv}"
    )

    print(
        f"Flow rows:    {len(timestamps)}"
    )

    print(
        f"Features:     {len(feature_names)}"
    )

    print(
        f"Median dt:    {median_dt:.6f}s"
    )

    print(
        f"Window:       {window:.2f}s"
    )

    print(
        f"Stride:       {stride:.2f}s"
    )

    print_dataset_summary(
        X,
        y,
        manifest,
    )

    print_annotation_summary(
        annotations
    )

    print_preview(
        manifest,
        preview,
    )

    print()
    print(
        f"Dataset:  {npz_path}"
    )

    print(
        f"Manifest: {manifest_path}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a temporal MTB-interest dataset "
            "from simplified 0..3 annotations."
        )
    )

    parser.add_argument(
        "--flow-csv",
        type=Path,
        default=None,
        help=(
            "Optical-flow CSV produced by "
            "analyze_bike_flow.py"
        ),
    )

    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path(
            "data/annotations/Mentorella.txt"
        ),
        help=(
            "Simplified MTB annotation file "
            "(default: data/annotations/Mentorella.txt)"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "output/datasets"
        ),
        help=(
            "Output directory "
            "(default: output/datasets)"
        ),
    )

    parser.add_argument(
        "--window",
        type=float,
        default=DEFAULT_WINDOW,
        help=(
            "Window duration in seconds "
            "(default: 4)"
        ),
    )

    parser.add_argument(
        "--stride",
        type=float,
        default=DEFAULT_STRIDE,
        help=(
            "Window stride in seconds "
            "(default: 2)"
        ),
    )

    parser.add_argument(
        "--flow-start",
        type=float,
        default=0.0,
        help=(
            "Start time of the analyzed flow segment "
            "in the source video (default: 0)"
        ),
    )

    parser.add_argument(
        "--dt-tolerance",
        type=float,
        default=DEFAULT_DT_TOLERANCE,
        help=(
            "Maximum relative sampling deviation "
            "before warning (default: 0.15)"
        ),
    )

    parser.add_argument(
        "--preview",
        type=int,
        default=40,
        help=(
            "Number of generated windows to print "
            "(default: 40)"
        ),
    )

    parser.add_argument(
        "--interest-report",
        action="store_true",
        help=(
            "Report how the 0..3 annotations map to "
            "temporal windows without writing a dataset."
        ),
    )

    parser.add_argument(
        "--audit-only",
        action="store_true",
        help=(
            "Audit one or more annotation/flow pairs without "
            "writing datasets."
        ),
    )

    parser.add_argument(
        "--audit-pair",
        nargs=2,
        action="append",
        metavar=("ANNOTATIONS", "FLOW_CSV"),
        help=(
            "Annotation file and matching flow CSV. "
            "May be supplied multiple times."
        ),
    )

    args = parser.parse_args()

    if args.audit_only:
        if not args.audit_pair:
            parser.error(
                "--audit-only requires at least one --audit-pair"
            )

        audit_pairs(
            pairs=[
                (
                    Path(annotation),
                    Path(flow_csv),
                )
                for annotation, flow_csv in args.audit_pair
            ],
            window=args.window,
            stride=args.stride,
            flow_start=args.flow_start,
            dt_tolerance=args.dt_tolerance,
            preview=args.preview,
        )

        return 0

    if args.audit_pair:
        parser.error(
            "--audit-pair requires --audit-only"
        )

    if args.flow_csv is None:
        parser.error(
            "--flow-csv is required unless --audit-only is used"
        )
        
    if args.window <= 0:
        parser.error("--window must be > 0")

    if args.stride <= 0:
        parser.error("--stride must be > 0")

    if args.preview < 0:
        parser.error("--preview must be >= 0")

    if args.interest_report:
        print_interest_report(
            flow_csv=args.flow_csv,
            annotations_path=args.annotations,
            window=args.window,
            stride=args.stride,
            flow_start=args.flow_start,
            preview=args.preview,
        )

        return 0

    build_dataset(
        flow_csv=args.flow_csv,
        annotations_path=args.annotations,
        output_dir=args.output_dir,
        window=args.window,
        stride=args.stride,
        flow_start=args.flow_start,
        dt_tolerance=args.dt_tolerance,
        preview=args.preview,
    )

    return 0

if __name__ == "__main__":
    raise SystemExit(main())